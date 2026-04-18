
import torch
import numpy as np
import faiss
from tqdm import tqdm
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import ColPaliForRetrieval, ColPaliProcessor, BitsAndBytesConfig


class ColPaliIndexer:
    def __init__(self, model_name: str = "vidore/colpali-v1.3-hf"):
        print(f"Loading ColPali: {model_name}")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = ColPaliForRetrieval.from_pretrained(
            model_name,
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        self.page_embeddings: List[Dict] = []
        print("ColPali loaded (4-bit quantized)")

    def _to_device(self, inputs: dict) -> dict:
        return {
            k: v.to(self.model.device).to(torch.bfloat16)
            if torch.is_floating_point(v) else v.to(self.model.device)
            for k, v in inputs.items()
        }

    @torch.no_grad()
    def embed_pages(self, page_images: list, batch_size: int = 2):
        
        print(f"\n Embedding {len(page_images)} pages with ColPali...")
        self.page_embeddings = []

        for i in tqdm(range(0, len(page_images), batch_size), desc="ColPali pages"):
            batch  = page_images[i: i + batch_size]
            images = [b[2] for b in batch]
            try:
               
                if hasattr(self.processor, "process_images"):
                    inputs = self.processor.process_images(images)
                else:
                    inputs = self.processor(images=images, return_tensors="pt")

                inputs = self._to_device(inputs)
                outputs = self.model(**inputs)

                for idx, emb in enumerate(outputs.embeddings):
                    pnum, sdoc, _ = batch[idx]
                    self.page_embeddings.append({
                        "page_num":   pnum,
                        "source_doc": sdoc,
                        "embedding":  emb.detach().cpu().float(),
                    })
            except Exception as exc:
                print(f"  [ColPali page warn] batch {i}: {exc}")

        print(f" {len(self.page_embeddings)} page embeddings stored")

    @torch.no_grad()
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
     
        if not self.page_embeddings:
            return []
        try:
            if hasattr(self.processor, "process_queries"):
                inputs = self.processor.process_queries([query_text])
            else:
                inputs = self.processor(text=[query_text], return_tensors="pt")

            inputs = self._to_device(inputs)
            q_emb  = self.model(**inputs).embeddings[0].detach().cpu().float()
        except Exception as exc:
            print(f"  [ColPali query warn]: {exc}")
            return []

        scores = []
        for pd_ in self.page_embeddings:
          sim = torch.matmul(q_emb, pd_["embedding"].T)
          score = sim.max(dim=1).values.sum().item()
          scores.append((pd_["page_num"], pd_["source_doc"], score))
        scores.sort(key=lambda x: x[2], reverse=True)
        return [
            {"page_num": p, "source_doc": s, "score": sc}
            for p, s, sc in scores[:top_k]
        ]


class TextIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading text embedder: {model_name}")
        self.model  = SentenceTransformer(model_name)
        self.dim    = self.model.get_sentence_embedding_dimension()
        self.index  = None
        self.chunks = []
        print(f"Text embedder loaded (dim={self.dim})")

    def build_index(self, chunks, batch_size: int = 64):
        self.chunks = list(chunks)
        texts = [c.content for c in self.chunks]
        print(f"\n Embedding {len(texts)} chunks with MiniLM...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        n = len(self.chunks)
        if n > 1000:
            nlist     = max(4, min(100, int(n ** 0.5)))
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings)
            self.index.nprobe = max(1, nlist // 4)
        else:
            self.index = faiss.IndexFlatIP(self.dim)

        self.index.add(embeddings)
        print(f"FAISS index built: {self.index.ntotal} vectors")

    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        if self.index is None or not self.chunks:
            return []
        q_emb    = self.model.encode([query_text], normalize_embeddings=True).astype(np.float32)
        k_actual = min(top_k, len(self.chunks))
        scores, indices = self.index.search(q_emb, k_actual)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            c = self.chunks[idx]
            results.append({
                "chunk":      c,
                "score":      float(score),
                "page_num":   c.page_num,
                "chunk_type": c.chunk_type,
                "source_doc": c.source_doc,
            })
        return results

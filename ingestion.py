
import os
import re
import fitz
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

MAX_PAGES = 20
MAX_CAPTIONS_PER_PAGE = 2
CHUNK_MAX_TOKENS = 300   
CHUNK_OVERLAP_PARAS= 1

@dataclass
class DocumentChunk:
    chunk_id:   str
    page_num:   int
    chunk_type: str 
    content:    str
    source_doc: str
    metadata:   Dict[str, Any] = field(default_factory=dict)


class MultiModalIngestionPipeline:
    CAPTION_RE = re.compile(
        r"^(Figure|Fig\.|Chart|Graph|Exhibit|Box)\s*[\dIVX\.]+",
        re.IGNORECASE,
    )

    HEADING_PATTERNS = [
        re.compile(r"^[IVX]+\.\s+[A-Z][A-Za-z\s]{2,50}$"),
        re.compile(r"^\d+\.\s+[A-Z][A-Z\s]{4,60}$"),
        re.compile(r"^[A-Z][A-Z\s]{6,60}$"),
        re.compile(r"^(Box|Figure|Table|Annex|Appendix)\s+[\dIVX]"),
        re.compile(r"^(Chapter|CHAPTER|SECTION|Section)\s+\d+"),
        re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$"), 
    ]

    def __init__(self, dpi: int = 150):
        self.dpi = dpi

    def _render_page_image(self, page) -> Image.Image:
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        return Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")

    def _is_heading(self, text: str) -> bool:
        t = text.strip()
        if not t or len(t) > 100:
            return False
        if t.endswith(".") and len(t.split()) > 4:
            return False
        if len(t.split()) > 10:  
            return False
        return any(p.match(t) for p in self.HEADING_PATTERNS)

    def _extract_tables(self, page, page_num: int, source_doc: str) -> List[DocumentChunk]:
        chunks = []
        try:
            for t_idx, table in enumerate(page.find_tables()):
                raw = table.extract()
                if not raw or len(raw) < 2:
                    continue
                df = pd.DataFrame(raw)
                first_row = [str(c).strip() for c in df.iloc[0].fillna("")]
                if any(first_row):
                    df.columns = [h or f"Col{i}" for i, h in enumerate(first_row)]
                    df = df.iloc[1:].reset_index(drop=True)
                df = df.fillna("")
                if df.shape[0] < 1 or df.shape[1] < 2:
                    continue
                table_text = (
                    f"[TABLE | {source_doc} | Page {page_num}]\n"
                    + df.to_string(index=False)
                )
                chunks.append(DocumentChunk(
                    chunk_id=f"{source_doc}_table_p{page_num}_t{t_idx}",
                    page_num=page_num,
                    chunk_type="table",
                    content=table_text,
                    source_doc=source_doc,
                    metadata={"rows": df.shape[0], "cols": df.shape[1]},
                ))
        except Exception as exc:
            print(f"  [table warn] p{page_num}: {exc}")
        return chunks

    def _extract_image_captions(self, page, page_num: int, source_doc: str) -> List[DocumentChunk]:
        seen_labels: set = set()
        raw_candidates: List[DocumentChunk] = []
        try:
            text  = page.get_text("text")
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            for i, line in enumerate(lines):
                if not self.CAPTION_RE.match(line):
                    continue
                label_key = re.sub(r"\s+", " ", line[:50]).lower().strip()
                if label_key in seen_labels:
                    continue
                seen_labels.add(label_key)
                caption_lines = [line] + lines[i + 1: i + 5]
                caption_text  = (
                    f"[FIGURE/CHART | {source_doc} | Page {page_num}]\n"
                    + " ".join(caption_lines)
                )
                raw_candidates.append(DocumentChunk(
                    chunk_id=f"{source_doc}_fig_p{page_num}_l{i}",
                    page_num=page_num,
                    chunk_type="image_caption",
                    content=caption_text,
                    source_doc=source_doc,
                    metadata={"caption_label": line},
                ))
        except Exception:
            pass
        raw_candidates.sort(key=lambda c: len(c.content), reverse=True)
        return raw_candidates[:MAX_CAPTIONS_PER_PAGE]

    def _chunk_text(
        self,
        text: str,
        page_num: int,
        section: str,
        source_doc: str,
        max_tokens: int = CHUNK_MAX_TOKENS,
    ) -> List[DocumentChunk]:
      
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text.strip()) if p.strip()]
        if not paragraphs:
            return []

        chunks: List[DocumentChunk] = []
        chunk_idx      = 0
        i              = 0
        last_chunk_end = 0   

        while i < len(paragraphs):
            current_paras: List[str] = []
            cur_len = 0

            if chunk_idx > 0 and last_chunk_end > 0:
                overlap_start = max(0, last_chunk_end - CHUNK_OVERLAP_PARAS)
                for op in paragraphs[overlap_start:last_chunk_end]:
                    current_paras.append(op)
                    cur_len += len(op.split())

            chunk_start = i
            while i < len(paragraphs):
                para = paragraphs[i]
                plen = len(para.split())
                if cur_len + plen > max_tokens and current_paras:
                    break
                current_paras.append(para)
                cur_len += plen
                i += 1

            last_chunk_end = i   

            combined = " ".join(current_paras).strip()
            if len(combined) < 40:
                chunk_idx += 1
                continue

            chunks.append(DocumentChunk(
                chunk_id=f"{source_doc}_text_p{page_num}_c{chunk_idx}",
                page_num=page_num,
                chunk_type="heading" if self._is_heading(combined) else "text",
                content=combined,
                source_doc=source_doc,
                metadata={"section": section},
            ))
            chunk_idx += 1

        return chunks

    def ingest(self, pdf_paths: List[str]) -> Tuple[List[DocumentChunk], list]:
        all_chunks: List[DocumentChunk] = []
        page_images: list = []

        for pdf_path in pdf_paths:
            source_doc       = os.path.splitext(os.path.basename(pdf_path))[0]
            doc              = fitz.open(pdf_path)
            total_pages      = len(doc)
            pages_to_process = min(total_pages, MAX_PAGES)
            print(
                f"\n {source_doc}: {total_pages} pages total — "
                f"processing first {pages_to_process}"
            )

            section = "Introduction"
            for page_num in tqdm(range(pages_to_process), desc=f"  {source_doc}"):
                page = doc[page_num]
                pnum = page_num + 1

                page_images.append((pnum, source_doc, self._render_page_image(page)))

                text = page.get_text("text")

                for line in text.split("\n"):
                    stripped = line.strip()
                    if stripped and self._is_heading(stripped):
                        section = stripped[:80]
                        break

                all_chunks.extend(self._extract_tables(page, pnum, source_doc))
                all_chunks.extend(self._extract_image_captions(page, pnum, source_doc))
                all_chunks.extend(self._chunk_text(text, pnum, section, source_doc))

            doc.close()

        type_counts: Dict[str, int] = {}
        for c in all_chunks:
            type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1

        print(f"\n Ingestion complete — {len(all_chunks)} chunks from {len(pdf_paths)} PDFs")
        for t, n in sorted(type_counts.items()):
            print(f"   {t}: {n}")
        print(f"   Page images: {len(page_images)}")
        return all_chunks, page_images

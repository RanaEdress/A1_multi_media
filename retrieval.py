
import re
from typing import List, Dict, Tuple, Set, Optional

CHUNK_TYPE_BUDGET = {
    "text": 4,
    "heading": 2,
    "table": 4,
    "image_caption": 5,
}

TABLE_QUERY_SIGNALS = re.compile(
    r"\b(percent|%|share|ratio|rate|gdp|growth|figure|table|how much|"
    r"how many|total|amount|number|population|billion|million|"
    r"classification|income|low.income|high.income|upper|lower|"
    r"extreme poverty|emissions|co2|projection|forecast|outlook)\b",
    re.IGNORECASE,
)

VISUAL_QUERY_SIGNALS = re.compile(
    r"\b(figure|fig\.|chart|graph|caption|box|image|diagram|"
    r"show|illustrate|depict|visual|plot)\b",
    re.IGNORECASE,
)

TABLE_PAGE_BONUS= 0.35
DOC_TARGET_BONUS = 0.20
TEXT_WEIGHT_DEFAULT = 1
VISUAL_WEIGHT_DEFAULT = 2
TEXT_WEIGHT_VISUAL = 1
VISUAL_WEIGHT_VISUAL  = 3

DOC_TARGET_PATTERNS: List[Tuple] = [
    (re.compile(r"\b(imf|weo|world economic outlook)\b", re.IGNORECASE), "imf_weo"),
    (re.compile(r"\b(world bank|wb report|middle.income trap)\b", re.IGNORECASE), "world_bank"),
]


class DualModeRetriever:
    def __init__(self, colpali, text_indexer, chunks, page_images):
        self.colpali      = colpali
        self.text_indexer = text_indexer
        self.chunks       = list(chunks)

        self.page_image_map: Dict[Tuple, object] = {
            (p[0], p[1]): p[2] for p in page_images
        }

        self._table_pages: Set[Tuple] = {
            (c.page_num, c.source_doc)
            for c in self.chunks if c.chunk_type == "table"
        }

        self._chunks_by_page: Dict[Tuple, List] = {}
        for c in self.chunks:
            key = (c.page_num, c.source_doc)
            self._chunks_by_page.setdefault(key, []).append(c)

    def _is_table_query(self, query: str) -> bool:
        return bool(TABLE_QUERY_SIGNALS.search(query))

    def _is_visual_query(self, query: str) -> bool:
        return bool(VISUAL_QUERY_SIGNALS.search(query))

    def _detect_target_doc(self, query: str) -> Optional[str]:
        for pattern, doc_key in DOC_TARGET_PATTERNS:
            if pattern.search(query):
                return doc_key
        return None

    def _rrf(self, rank_lists: List[List], weights: List[int], k: int = 60) -> Dict:
        scores: Dict = {}
        for rank_list, w in zip(rank_lists, weights):
            for rank, key in enumerate(rank_list):
                scores[key] = scores.get(key, 0.0) + w * (1.0 / (k + rank + 1))
        return scores

    def _apply_table_bonus(self, rrf_scores: Dict, query: str) -> Dict:
        if not self._is_table_query(query):
            return rrf_scores
        boosted = dict(rrf_scores)
        for key in boosted:
            if key in self._table_pages:
                boosted[key] += TABLE_PAGE_BONUS
        return boosted

    def _apply_doc_target_bonus(self, rrf_scores: Dict, target_doc: Optional[str]) -> Dict:
        if target_doc is None:
            return rrf_scores
        boosted = dict(rrf_scores)
        for (page_num, source_doc) in boosted:
            if target_doc in source_doc:
                boosted[(page_num, source_doc)] += DOC_TARGET_BONUS
        return boosted

    def _apply_type_budget(self, chunks: List, rank_order: List[Tuple]) -> List:
        rank_map = {key: i for i, key in enumerate(rank_order)}
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (rank_map.get((c.page_num, c.source_doc), 999), c.page_num),
        )
        counts: Dict[str, int] = {}
        kept = []
        for c in sorted_chunks:
            budget = CHUNK_TYPE_BUDGET.get(c.chunk_type, 4)
            used   = counts.get(c.chunk_type, 0)
            if used < budget:
                kept.append(c)
                counts[c.chunk_type] = used + 1
        return kept

    def retrieve(self, query: str, top_k: int = 5) -> Dict:
        is_visual = self._is_visual_query(query)

        if is_visual:
            tw, vw = TEXT_WEIGHT_VISUAL, VISUAL_WEIGHT_VISUAL
        else:
            tw, vw = TEXT_WEIGHT_DEFAULT, VISUAL_WEIGHT_DEFAULT

        text_results   = self.text_indexer.query(query, top_k=top_k * 2)
        visual_results = self.colpali.query(query,      top_k=top_k * 2)

        seen: Set = set()
        text_keys: List[Tuple] = []
        for r in text_results:
            key = (r["page_num"], r["source_doc"])
            if key not in seen:
                seen.add(key)
                text_keys.append(key)

        seen_v: Set = set()
        visual_keys: List[Tuple] = []
        for r in visual_results:
            key = (r["page_num"], r["source_doc"])
            if key not in seen_v:
                seen_v.add(key)
                visual_keys.append(key)

        rrf_scores = self._rrf([text_keys, visual_keys], weights=[tw, vw])
        rrf_scores = self._apply_table_bonus(rrf_scores, query)
        target_doc = self._detect_target_doc(query)
        rrf_scores = self._apply_doc_target_bonus(rrf_scores, target_doc)

        top_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]

        candidate_chunks = []
        for key in top_keys:
            candidate_chunks.extend(self._chunks_by_page.get(key, []))

        retrieved_chunks = self._apply_type_budget(candidate_chunks, top_keys)

        context_parts = []
        for chunk in retrieved_chunks:
            header = (
                f"[{chunk.source_doc} | Page {chunk.page_num} "
                f"| {chunk.chunk_type.upper()}]"
            )
            if chunk.metadata.get("section"):
                header += f" — {chunk.metadata['section'][:60]}"
            context_parts.append(f"{header}\n{chunk.content}")

        context = "\n\n---\n\n".join(context_parts)

        # visual_pages: for visual queries trust ColPali ranking; else use RRF top
        if is_visual and visual_keys:
            visual_page_keys = visual_keys[:3]
        else:
            visual_page_keys = top_keys[:3]

        visual_pages = []
        for key in visual_page_keys:
            if key in self.page_image_map:
                visual_pages.append((key[0], key[1], self.page_image_map[key]))

        modality_counts: Dict[str, int] = {}
        for c in retrieved_chunks:
            modality_counts[c.chunk_type] = modality_counts.get(c.chunk_type, 0) + 1

        return {
            "context":          context,
            "source_pages":     top_keys,
            "retrieved_chunks": retrieved_chunks,
            "visual_pages":     visual_pages,
            "modality_counts":  modality_counts,
        }

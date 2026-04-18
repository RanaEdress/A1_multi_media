
import re
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

semantic_eval_model = SentenceTransformer("all-MiniLM-L6-v2")

MAX_PAGES = 20

EVAL_QUERIES: List[Dict] = [
    {
        "query": "What is the central concept introduced in the World Bank 2024 report overview?",
        "type":  "text",
        "expected_keywords": ["middle", "income", "trap", "growth", "development"],
    },
    {
        "query": "What does the IMF say about inflation in advanced economies in 2024?",
        "type":  "text",
        "expected_keywords": ["inflation", "advanced", "economies", "disinflation", "percent"],
    },
    {
        "query": "What output growth figures for advanced economies appear in the IMF WEO projections table?",
        "type":  "table",
        "expected_keywords": ["advanced", "economies", "output", "growth", "projection"],
    },
    {
        "query": "What inflation rates for emerging market economies are shown in the IMF WEO overview table?",
        "type":  "table",
        "expected_keywords": ["inflation", "emerging", "market", "percent", "2024"],
    },
    {
        "query": "What does Figure 1.1 in the IMF WEO show about global growth or inflation trends?",
        "type":  "visual",
        "expected_keywords": ["figure", "global", "growth", "inflation", "trend"],
    },
    {
        "query": "What does Figure O.1 in the World Bank report show about middle-income countries?",
        "type":  "visual",
        "expected_keywords": ["figure", "middle", "income", "country", "growth"],
    },
    {
        "query": "What captions describe charts about economic risks in the IMF WEO opening chapter?",
        "type":  "image",
        "expected_keywords": ["figure", "risk", "outlook", "economic", "downside"],
    },
    {
        "query": "What are the captions of figures or boxes in the World Bank report overview pages?",
        "type":  "image",
        "expected_keywords": ["figure", "box", "overview", "income", "trap"],
    },
]

CITATION_RE = re.compile(r"\[Source[s]?\s*:[^\]]+\]", re.IGNORECASE)


def evaluate_system(queries, retriever, answer_gen, verbose=True) -> pd.DataFrame:
    results         = []
    page_violations = []

    for i, q in enumerate(queries):
        query       = q["query"]
        expected_kw = q["expected_keywords"]
        qtype       = q["type"]

        print(f"\n[{i+1}/{len(queries)}] ({qtype.upper()}) {query[:70]}...")

        retrieval    = retriever.retrieve(query)
        visual_pages = retrieval.get("visual_pages", [])
        source_pages = retrieval.get("source_pages", [])

        out_of_range = []
        for item in source_pages:
            page_num = item[0] if isinstance(item, tuple) else item
            if page_num > MAX_PAGES:
                out_of_range.append(page_num)
        if out_of_range:
            print(f"   PAGE RANGE WARNING: pages {out_of_range} > MAX_PAGES={MAX_PAGES}")
            page_violations.append({"query_idx": i + 1, "query": query, "bad_pages": out_of_range})
        else:
            print(f" Page range OK")

        answer = answer_gen.generate(
            query, retrieval["context"], visual_pages, source_pages=source_pages
        )

        ctx_lower    = retrieval["context"].lower()
        kw_ctx       = sum(1 for kw in expected_kw if kw.lower() in ctx_lower)
        ctx_kw_score = round(kw_ctx / max(1, len(expected_kw)), 2)

        ans_lower    = answer.lower()
        kw_ans       = sum(1 for kw in expected_kw if kw.lower() in ans_lower)
        ans_kw_score = round(kw_ans / max(1, len(expected_kw)), 2)

        expected_text  = " ".join(expected_kw)
        ans_emb        = semantic_eval_model.encode(answer,        convert_to_tensor=True)
        exp_emb        = semantic_eval_model.encode(expected_text, convert_to_tensor=True)
        semantic_score = float(max(0.0, min(1.0, util.cos_sim(ans_emb, exp_emb).item())))

        has_citation = bool(CITATION_RE.search(answer))

        if source_pages and isinstance(source_pages[0], tuple):
            pages_retrieved = [f"{s}:p{p}" for p, s in source_pages]
        else:
            pages_retrieved = [f"p{p}" for p in source_pages]

        modality_counts = retrieval.get("modality_counts", {})
        modalities_hit  = ", ".join(f"{k}:{v}" for k, v in modality_counts.items()) or "none"

        result = {
            "query":               query,
            "type":                qtype,
            "pages_retrieved":     pages_retrieved,
            "n_pages":             len(source_pages),
            "pages_in_range":      len(out_of_range) == 0,
            "modalities_hit":      modalities_hit,
            "context_kw_coverage": ctx_kw_score,
            "answer_kw_coverage":  ans_kw_score,
            "semantic_score":      round(semantic_score, 2),
            "has_citation":        has_citation,
            "answer_words":        len(answer.split()),
            "answer_preview":      answer[:250].replace("\n", " "),
        }
        results.append(result)

        if verbose:
            print(f"  Answer  : {answer[:200]}")
            print(f"  Pages   : {result['pages_retrieved']}")
            print(f"  Mods    : {modalities_hit}")
            print(f"  CtxKW:{ctx_kw_score:.0%} AnsKW:{ans_kw_score:.0%} Sem:{semantic_score:.0%} Cite:{has_citation}")

    df = pd.DataFrame(results)

    print(f"Queries evaluated        : {len(results)}")
    print(f"Pages-in-range rate      : {df['pages_in_range'].mean():.1%}")
    print(f"Avg context KW coverage  : {df['context_kw_coverage'].mean():.1%}")
    print(f"Avg answer KW coverage   : {df['answer_kw_coverage'].mean():.1%}")
    print(f"Avg semantic score       : {df['semantic_score'].mean():.1%}")
    print(f"Citation rate            : {df['has_citation'].mean():.1%}")
    print(f"Avg pages retrieved      : {df['n_pages'].mean():.1f}")
    print(f"Avg answer length (words): {df['answer_words'].mean():.0f}")

    if page_violations:
        print(f"\n {len(page_violations)} page-range violation(s):")
        for v in page_violations:
            print(f"   Query {v['query_idx']}: pages {v['bad_pages']} — {v['query'][:60]}")

    print("\nBy modality type:")
    print(
        df.groupby("type")[
            ["context_kw_coverage", "answer_kw_coverage", "semantic_score",
             "has_citation", "pages_in_range"]
        ].mean().round(2).to_string()
    )
    return df

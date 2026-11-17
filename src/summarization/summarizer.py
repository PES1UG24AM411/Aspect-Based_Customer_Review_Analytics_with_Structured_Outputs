"""Product summaries: a deterministic extractive baseline (no API) and an LLM abstractive summariser.

Both are built ONLY from validated extraction evidence. The LLM judge in judge.py is a
supplementary consistency check; it is NOT a substitute for human-annotated ground truth.
"""
from pathlib import Path

import pandas as pd

from src.extraction.config import MODEL, PROJECT_ROOT
from .schemas import Claim, ProductSummary

SUMMARY_PROMPT = (PROJECT_ROOT / "prompts" / "product_summary_v1.txt")


def build_evidence(aspects_df, product_id, max_items=40):
    """Evidence items E1..En for one product (one per review/aspect, no duplicates)."""
    d = aspects_df[(aspects_df.product_id == product_id) & (~aspects_df.dup_in_review)]
    d = d.sort_values(["review_id", "aspect_id"]).head(max_items)
    return [{"id": f"E{i}", "review_id": r.review_id, "aspect": r.aspect, "sentiment": r.sentiment,
             "complaint": bool(r.complaint_present), "quote": r.evidence} for i, r in enumerate(d.itertuples(), 1)]


def extractive_summary(aspects_df, product_id, *, top_k=3, min_mentions=2):
    """Rule-based summary: top positive/negative aspects by mention count, each citing real evidence."""
    ev = build_evidence(aspects_df, product_id, max_items=10_000)
    if not ev:
        return ProductSummary(product_id=product_id, overview="No extracted aspects for this product.",
                              strengths=[], weaknesses=[])
    df = pd.DataFrame(ev)
    n_rev = df.review_id.nunique()

    def claims(sentiment, label):
        sub = df[df.sentiment == sentiment].groupby("aspect")
        ranked = sorted(sub, key=lambda kv: -len(kv[1]))
        out = []
        for aspect, g in ranked:
            if len(g) < min_mentions and len(out) >= 1:
                continue
            ids = g.id.tolist()[:3]
            out.append(Claim(claim=f"{aspect.capitalize()} is {label} in {len(g)} of {n_rev} analysed review(s).",
                             evidence_ids=ids))
            if len(out) >= top_k:
                break
        return out

    return ProductSummary(product_id=product_id,
                          overview=f"Based on {n_rev} analysed review(s) with {len(df)} aspect mention(s).",
                          strengths=claims("positive", "praised"), weaknesses=claims("negative", "criticised"))


def llm_summary(aspects_df, product_id, *, client=None, model=None):
    from openai import OpenAI
    client = client or OpenAI(timeout=60.0, max_retries=0)
    ev = build_evidence(aspects_df, product_id)
    if not ev:
        return extractive_summary(aspects_df, product_id), ev
    lines = "\n".join(f"{e['id']} [{e['sentiment']}{', complaint' if e['complaint'] else ''}] {e['aspect']}: \"{e['quote']}\"" for e in ev)
    resp = client.responses.parse(model=model or MODEL, text_format=ProductSummary,
                                  input=f"{SUMMARY_PROMPT.read_text(encoding='utf-8')}\n\nproduct_id: {product_id}\nEVIDENCE:\n{lines}")
    s = resp.output_parsed
    if s is None:
        raise ValueError("No structured summary returned")
    return s, ev

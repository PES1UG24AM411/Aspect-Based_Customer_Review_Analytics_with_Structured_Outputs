"""Product-level aggregation of validated extraction records.

Stable key = product_id (ASIN). Reviews are de-duplicated by review_id so a review that
appears in both the pilot and full prediction files is counted once.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.extraction.config import PROJECT_ROOT
from src.extraction.json_utils import read_jsonl
from src.extraction.schemas import ExtractionRecord
from .aspect_normalizer import normalize_aspect

DEFAULT_PRED_FILES = ("data/extracted/predictions.jsonl", "data/extracted/pilot_predictions.jsonl")


def load_records(paths):
    """Valid records, de-duplicated by review_id (first file wins)."""
    seen, out, skipped = set(), [], 0
    for p in paths:
        for r in read_jsonl(p):
            try:
                rec = ExtractionRecord.model_validate(r)
            except Exception:
                skipped += 1
                continue
            if rec.review_id in seen:
                continue
            seen.add(rec.review_id)
            out.append(rec)
    return out, skipped


def aspect_frame(records, reviews_df=None):
    """One row per (review, aspect). Reviews with zero aspects are kept in review_frame."""
    rows = []
    for rec in records:
        for a in rec.aspects:
            rows.append({"review_id": rec.review_id, "product_id": rec.product_id,
                         "aspect_id": a.aspect_id, "aspect_term": a.aspect_term,
                         "aspect": normalize_aspect(a.aspect_term), "sentiment": a.sentiment,
                         "evidence": a.evidence, "complaint_present": a.complaint_present,
                         "complaint_evidence": a.complaint_evidence})
    cols = ["review_id", "product_id", "aspect_id", "aspect_term", "aspect", "sentiment",
            "evidence", "complaint_present", "complaint_evidence"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        # a review mentioning the same canonical aspect twice counts once per sentiment-majority; keep both rows
        # but flag duplicates so review counts are never inflated
        df["dup_in_review"] = df.duplicated(["review_id", "aspect"], keep="first")
    else:
        df["dup_in_review"] = pd.Series(dtype=bool)
    return df


def review_frame(records, reviews_df=None):
    df = pd.DataFrame([{"review_id": r.review_id, "product_id": r.product_id, "n_aspects": len(r.aspects),
                        "n_complaints": sum(a.complaint_present for a in r.aspects)} for r in records],
                      columns=["review_id", "product_id", "n_aspects", "n_complaints"])
    if reviews_df is not None and not df.empty:
        meta = reviews_df[["review_id", "product_name", "rating"]].copy()
        meta["review_id"] = meta.review_id.astype(str)
        df = df.merge(meta, on="review_id", how="left")
    return df


def aggregate(records, reviews_df=None, *, min_reviews_recurring=3, recurring_neg_share=0.5):
    """Returns a dict of DataFrames (all derived from the same de-duplicated records)."""
    af, rf = aspect_frame(records), review_frame(records, reviews_df)
    uniq = af[~af.dup_in_review] if not af.empty else af
    out = {"aspects": af, "reviews": rf}
    if uniq.empty:
        for k in ("aspect_frequency", "sentiment_distribution", "complaints", "recurring_negative",
                  "aspect_comparison", "product_overview"):
            out[k] = pd.DataFrame()
        return out
    out["aspect_frequency"] = (uniq.groupby(["product_id", "aspect"]).review_id.nunique()
                               .rename("n_reviews").reset_index().sort_values(["product_id", "n_reviews"], ascending=[True, False]))
    sd = uniq.groupby(["product_id", "aspect", "sentiment"]).size().unstack("sentiment", fill_value=0)
    for s in ("positive", "negative", "neutral"):
        if s not in sd:
            sd[s] = 0
    sd = sd[["positive", "negative", "neutral"]]
    sd["total"] = sd.sum(axis=1)
    sd["net_sentiment"] = ((sd.positive - sd.negative) / sd.total).round(3)
    out["sentiment_distribution"] = sd.reset_index()
    comp = uniq[uniq.complaint_present]
    out["complaints"] = (comp.groupby(["product_id", "aspect"]).review_id.nunique().rename("n_complaint_reviews")
                         .reset_index().sort_values("n_complaint_reviews", ascending=False))
    # recurring negative: product-aspect with >= min_reviews mentions and negative share >= threshold
    sdf = out["sentiment_distribution"].copy()
    sdf["negative_share"] = (sdf.negative / sdf.total).round(3)
    out["recurring_negative"] = sdf[(sdf.total >= min_reviews_recurring) & (sdf.negative_share >= recurring_neg_share)] \
        .sort_values(["negative_share", "total"], ascending=False).reset_index(drop=True)
    out["aspect_comparison"] = sdf.pivot(index="aspect", columns="product_id", values="net_sentiment")
    names = {}
    if reviews_df is not None:
        names = reviews_df.drop_duplicates("product_id").set_index("product_id").product_name.to_dict()
    po = (rf.groupby("product_id").agg(n_reviews=("review_id", "nunique"), n_aspects=("n_aspects", "sum"),
                                         n_complaints=("n_complaints", "sum")).reset_index())
    po["product_name"] = po.product_id.map(names)
    out["product_overview"] = po.sort_values("n_reviews", ascending=False).reset_index(drop=True)
    return out


def write_outputs(tables, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(out_dir / f"{name}.csv", index=name == "aspect_comparison")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="*", default=[str(PROJECT_ROOT / p) for p in DEFAULT_PRED_FILES])
    ap.add_argument("--reviews", default=str(PROJECT_ROOT / "data/processed/amazon_reviews_cleaned.csv"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data/analysis"))
    a = ap.parse_args(argv)
    recs, skipped = load_records(a.predictions)
    if not recs:
        raise SystemExit("No valid extraction records found; run extraction first.")
    reviews = pd.read_csv(a.reviews)
    write_outputs(aggregate(recs, reviews), a.out)
    print(f"Aggregated {len(recs)} unique reviews ({skipped} invalid lines skipped) -> {a.out}")


if __name__ == "__main__":
    main()

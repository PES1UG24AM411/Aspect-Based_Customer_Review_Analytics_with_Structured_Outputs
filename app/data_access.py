"""Data loading for the UI. Reads existing pipeline outputs; never re-implements the pipeline."""
import json
from pathlib import Path

import pandas as pd

from src.analysis.aggregate import aggregate, load_records, DEFAULT_PRED_FILES
from src.extraction.config import PROJECT_ROOT, API_KEY_PRESENT
from src.extraction.json_utils import read_jsonl
from src.extraction.mock_client import mock_extract

REVIEWS = PROJECT_ROOT / "data/processed/amazon_reviews_cleaned.csv"


def load_reviews(root=PROJECT_ROOT):
    p = Path(root) / "data/processed/amazon_reviews_cleaned.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    df["review_id"] = df.review_id.astype(str)
    return df


def load_extraction(root=PROJECT_ROOT, use_mock=False, reviews=None):
    """Returns (records, source_label). Real predictions take priority; mock only if asked and none exist."""
    root = Path(root)
    paths = [root / p for p in DEFAULT_PRED_FILES]
    recs, _ = load_records(paths)
    if recs:
        return recs, "model predictions"
    if use_mock and reviews is not None:
        eng = reviews[reviews.language.astype(str).str.casefold().eq("english")].head(300)
        return [mock_extract(str(r.review_id), str(r.product_id), str(r.review_text)) for r in eng.itertuples()], "MOCK"
    return [], "none"


def load_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def pipeline_status(root=PROJECT_ROOT):
    root = Path(root)
    n_pilot = len(read_jsonl(root / "data/extracted/pilot_predictions.jsonl"))
    n_full = len(read_jsonl(root / "data/extracted/predictions.jsonl"))
    n_fail = len(read_jsonl(root / "data/extracted/failures.jsonl")) + len(read_jsonl(root / "data/extracted/pilot_failures.jsonl"))
    return {"api_key_configured": API_KEY_PRESENT, "pilot_predictions": n_pilot, "full_predictions": n_full,
            "logged_failures": n_fail,
            "pilot_approved": (root / "reports/pilot_approval.json").is_file(),
            "full_manifest": load_json(root / "reports/full_run_manifest.json"),
            "gold_present": (root / "data/evaluation/gold_annotations.jsonl").is_file(),
            "evaluation_report": (root / "reports/evaluation/evaluation_metrics.json").is_file()}

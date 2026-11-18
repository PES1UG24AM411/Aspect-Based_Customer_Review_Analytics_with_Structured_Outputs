"""Deterministic dev / evaluation split and blank annotation template.

* dev set  : pilot cohort + extra reviews; the ONLY data allowed for prompt tuning.
* eval set : disjoint stratified sample (by star bucket) that humans annotate; never used
             for prompt tuning. Star rating is used only to stratify the sample, never shown
             to the extractor.
The template contains NO model predictions, to avoid anchoring annotators.
"""
import json
import random
from pathlib import Path

import pandas as pd

from src.extraction.config import PROJECT_ROOT

HUMAN_EVALUATION_SIZE = 15


def select_human_evaluation_ids(
    splits,
    assignment,
    *,
    target_size=HUMAN_EVALUATION_SIZE,
):
    """Select a stable subset of the existing shared evaluation cohort.

    The assignment file already identifies reviews independently annotated by
    both annotators. Taking its first ``target_size`` IDs preserves that
    allocation and does not create or rewrite annotation records.
    """
    eval_ids = [str(review_id) for review_id in splits.get("eval_ids", [])]
    dev_ids = {str(review_id) for review_id in splits.get("dev_ids", [])}
    shared_ids = [
        str(review_id) for review_id in assignment.get("shared_review_ids", [])
    ]

    if len(eval_ids) != len(set(eval_ids)):
        raise ValueError("Evaluation split contains duplicate review IDs.")
    if len(shared_ids) != len(set(shared_ids)):
        raise ValueError("Shared annotation assignment contains duplicate IDs.")

    eval_set = set(eval_ids)
    invalid_shared = [review_id for review_id in shared_ids if review_id not in eval_set]
    if invalid_shared:
        raise ValueError(
            "Shared annotation IDs are not in the evaluation split: "
            + ", ".join(invalid_shared[:5])
        )

    selected = shared_ids[:target_size]
    if len(selected) != target_size:
        raise ValueError(
            f"Need {target_size} shared evaluation reviews; found {len(selected)}."
        )
    if dev_ids.intersection(selected):
        raise ValueError("Selected human evaluation IDs overlap the dev split.")
    return selected


def bucket(rating):
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return "unknown"
    return "negative" if r <= 2 else "neutral" if r < 4 else "positive"


def make_splits(df, *, pilot_ids, seed=42, dev_extra=10, eval_size=100, max_chars=2500):
    pool = df[df.review_text.astype(str).str.len().le(max_chars)].copy()
    pool["_b"] = pool.rating.map(bucket)
    pool = pool.sort_values("review_id")
    pilot = [str(i) for i in pilot_ids]
    rest = pool[~pool.review_id.astype(str).isin(pilot)]
    rng = random.Random(seed)
    ids = rest.review_id.astype(str).tolist()
    dev_extra_ids = sorted(rng.sample(ids, min(dev_extra, len(ids))))
    rest = rest[~rest.review_id.astype(str).isin(dev_extra_ids)]
    groups = {b: g.review_id.astype(str).tolist() for b, g in rest.groupby("_b")}
    total = sum(len(v) for v in groups.values())
    eval_ids = []
    for b in sorted(groups):
        k = max(1, round(eval_size * len(groups[b]) / total)) if total else 0
        eval_ids += rng.sample(groups[b], min(k, len(groups[b])))
    eval_ids = sorted(eval_ids)[:eval_size] if len(eval_ids) > eval_size else sorted(eval_ids)
    dev_ids = sorted(set(pilot) | set(dev_extra_ids))
    assert not set(dev_ids) & set(eval_ids)
    return dev_ids, eval_ids


def write_splits(df, *, pilot_ids, out_dir=None, seed=42, dev_extra=10, eval_size=100):
    out_dir = Path(out_dir) if out_dir else PROJECT_ROOT / "data/evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_ids, eval_ids = make_splits(df, pilot_ids=pilot_ids, seed=seed, dev_extra=dev_extra, eval_size=eval_size)
    (out_dir / "splits.json").write_text(json.dumps(
        {"seed": seed, "dev_ids": dev_ids, "eval_ids": eval_ids,
         "note": "dev_ids may be used for prompt tuning; eval_ids must never be."}, indent=2))
    by_id = df.set_index(df.review_id.astype(str))
    tpl = out_dir / "gold_annotations_template.jsonl"
    with tpl.open("w", encoding="utf-8") as f:
        for rid in eval_ids:
            r = by_id.loc[rid]
            f.write(json.dumps({"review_id": rid, "product_id": str(r.product_id),
                                "review_text": str(r.review_text), "annotator": None,
                                "status": "todo", "aspects": [], "notes": None}, ensure_ascii=False) + "\n")
    return dev_ids, eval_ids, tpl

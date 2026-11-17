"""Write per-product summaries (+ judge results) to data/analysis/summaries.jsonl."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.analysis.aggregate import aspect_frame, load_records, DEFAULT_PRED_FILES
from src.extraction.config import PROJECT_ROOT, API_KEY_PRESENT
from .judge import structural_check, llm_judge
from .summarizer import build_evidence, extractive_summary, llm_summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", nargs="*", default=[str(PROJECT_ROOT / p) for p in DEFAULT_PRED_FILES])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data/analysis/summaries.jsonl"))
    ap.add_argument("--llm", action="store_true", help="Use the LLM summariser and judge (needs OPENAI_API_KEY)")
    ap.add_argument("--min-reviews", type=int, default=3, help="Only summarise products with at least this many reviews")
    ap.add_argument("--max-products", type=int, default=50)
    a = ap.parse_args(argv)
    if a.llm and not API_KEY_PRESENT:
        raise SystemExit("OPENAI_API_KEY missing; run without --llm for the extractive summaries.")
    recs, _ = load_records(a.predictions)
    if not recs:
        raise SystemExit("No valid extraction records; run extraction first.")
    af = aspect_frame(recs)
    counts = pd.Series([r.product_id for r in recs]).value_counts()
    pids = [p for p in counts.index if counts[p] >= a.min_reviews][: a.max_products]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for pid in pids:
            ev = build_evidence(af, pid)
            if a.llm:
                s, ev = llm_summary(af, pid)
                judge = llm_judge(s, ev).model_dump()
            else:
                s, judge = extractive_summary(af, pid), None
            f.write(json.dumps({"product_id": pid, "method": "llm" if a.llm else "extractive",
                                "summary": s.model_dump(), "evidence": ev,
                                "structural_check": structural_check(s, ev), "llm_judge": judge}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pids)} product summaries -> {a.out}")


if __name__ == "__main__":
    main()

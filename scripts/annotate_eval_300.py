"""Generate AI silver annotations for exactly the 300 evaluation reviews.

Dry-run is the default. API requests require --execute.
Never writes to human gold or annotator files.
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.extraction.config import MODEL
from src.extraction.llm_client import extract_one
from src.extraction.schemas import ExtractionRecord
from src.extraction.validators import validate_record

SPLIT = ROOT / "data/evaluation/splits.json"
DATASET = ROOT / "data/processed/amazon_reviews_cleaned.csv"
OUT = ROOT / "data/evaluation/ai_annotations.jsonl"
FAIL = ROOT / "data/evaluation/ai_annotation_failures.jsonl"
REPORT = ROOT / "reports/ai_annotation_300/manifest.json"

INPUT_RATE = 0.10
OUTPUT_RATE = 0.50


def load_cohort():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    ids = [str(x) for x in split["eval_ids"]]

    if len(ids) != 300 or len(set(ids)) != 300:
        raise SystemExit("Expected exactly 300 unique evaluation IDs.")

    df = pd.read_csv(DATASET, dtype={"review_id": str, "product_id": str})
    df = df[df.review_id.isin(ids)].copy()

    if df.review_id.duplicated().any():
        raise SystemExit("Duplicate review IDs found in source dataset.")

    if set(df.review_id) != set(ids):
        raise SystemExit("Source dataset does not contain exactly the evaluation cohort.")

    if not df.language.astype(str).str.casefold().eq("english").all():
        raise SystemExit("Evaluation cohort contains non-English reviews.")

    by_id = {str(row.review_id): row for _, row in df.iterrows()}
    return [by_id[rid] for rid in ids]


def load_completed(rows):
    source = {str(r.review_id): r for r in rows}
    done = set()

    if not OUT.exists():
        return done

    for number, line in enumerate(OUT.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = ExtractionRecord.model_validate_json(line)
            row = source.get(rec.review_id)
            if row is None:
                raise ValueError("ID not in evaluation cohort")
            validate_record(
                rec,
                expected_review_id=str(row.review_id),
                expected_product_id=str(row.product_id),
                expected_review_text=str(row.review_text),
            )
            if rec.review_id in done:
                raise ValueError("Duplicate completed output ID")
            done.add(rec.review_id)
        except Exception as exc:
            raise SystemExit(f"Invalid existing output line {number}: {exc}")

    return done


def append(path, obj):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Make paid API requests. Without this flag, only dry-run.")
    parser.add_argument("--max-cost-usd", type=float, default=None,
                        help="Required spending estimate threshold for execution.")
    args = parser.parse_args()

    rows = load_cohort()
    done = load_completed(rows)
    todo = [r for r in rows if str(r.review_id) not in done]

    print("Model:", MODEL)
    print("Evaluation cohort:", len(rows))
    print("Already completed:", len(done))
    print("Remaining:", len(todo))
    print("Output:", OUT.relative_to(ROOT))
    print("Failures:", FAIL.relative_to(ROOT))
    print("Mode:", "EXECUTION" if args.execute else "DRY RUN")

    if not args.execute:
        print("\nDRY RUN ONLY. No API calls were made.")
        print("Estimated remaining cost based on pilot averages:")
        input_est = len(todo) * 7622 / 5
        output_est = len(todo) * 7969 / 5
        cost = (input_est * INPUT_RATE + output_est * OUTPUT_RATE) / 1_000_000
        print(f"Input tokens:  {input_est:.0f}")
        print(f"Output tokens: {output_est:.0f}")
        print(f"Estimated cost: ${cost:.4f}")
        print("\nTo execute, supply --execute and an explicit --max-cost-usd.")
        return

    if args.max_cost_usd is None or args.max_cost_usd <= 0:
        raise SystemExit("Execution requires a positive --max-cost-usd.")

    if OUT.exists() and not done:
        raise SystemExit("Existing output is invalid or empty; refusing to append.")

    if OUT.exists() and len(done) == 300:
        print("All 300 reviews already have valid annotations.")
        return

    # Check estimate against the user's explicit spending threshold.
    input_est = len(todo) * 7622 / 5
    output_est = len(todo) * 7969 / 5
    estimate = (input_est * INPUT_RATE + output_est * OUTPUT_RATE) / 1_000_000

    print(f"\nEstimated remaining cost: ${estimate:.4f}")
    if estimate > args.max_cost_usd:
        raise SystemExit("Estimate exceeds --max-cost-usd. No API calls made.")

    # A second explicit confirmation is required at execution time.
    confirmation = input(
        f"Type RUN {len(todo)} REVIEWS to authorize API requests: "
    )
    if confirmation != f"RUN {len(todo)} REVIEWS":
        raise SystemExit("Confirmation did not match. No API calls made.")

    from openai import OpenAI
    client = OpenAI(timeout=60.0, max_retries=0)

    successes = failures = 0
    input_tokens = output_tokens = total_tokens = 0
    started = datetime.now(timezone.utc).isoformat()

    for index, row in enumerate(todo, 1):
        item = {
            "review_id": str(row.review_id),
            "product_id": str(row.product_id),
            "review_text": str(row.review_text),
        }
        try:
            record, usage = extract_one(item, client=client, strategy="few_shot_v2")
            validate_record(
                record,
                expected_review_id=item["review_id"],
                expected_product_id=item["product_id"],
                expected_review_text=item["review_text"],
            )
            append(OUT, record.model_dump())
            successes += 1

            if usage:
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)

        except Exception as exc:
            append(FAIL, {
                "review_id": item["review_id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            failures += 1

        if index % 10 == 0 or index == len(todo):
            print(f"Progress: {index}/{len(todo)} | success={successes} | failed={failures}")

    actual_cost = (
        input_tokens * INPUT_RATE + output_tokens * OUTPUT_RATE
    ) / 1_000_000

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "prompt_strategy": "few_shot_v2",
        "cohort_size": 300,
        "previously_completed": len(done),
        "processed_this_run": len(todo),
        "successes_this_run": successes,
        "failures_this_run": failures,
        "completed_total": len(done) + successes,
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
        "estimated_cost_usd": round(actual_cost, 6),
        "output": str(OUT.relative_to(ROOT)),
        "failure_log": str(FAIL.relative_to(ROOT)),
        "annotation_type": "AI-generated silver annotations; not human gold labels",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

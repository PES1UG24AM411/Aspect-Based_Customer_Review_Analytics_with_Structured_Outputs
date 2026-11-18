"""Safely resume failed AI silver annotations for the evaluation cohort."""
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from openai import RateLimitError, APITimeoutError, APIConnectionError
from src.extraction.config import MODEL
from src.extraction.llm_client import extract_one
from src.extraction.schemas import ExtractionRecord
from src.extraction.validators import validate_record

SPLIT = ROOT / "data/evaluation/splits.json"
DATASET = ROOT / "data/processed/amazon_reviews_cleaned.csv"
OUT = ROOT / "data/evaluation/ai_annotations.jsonl"
OLD_FAIL = ROOT / "data/evaluation/ai_annotation_failures.jsonl"
FAIL = ROOT / "data/evaluation/ai_recovery_failures.jsonl"
REPORT = ROOT / "reports/ai_annotation_recovery/manifest.json"

INPUT_RATE = 0.10
OUTPUT_RATE = 0.50
REQUEST_INTERVAL = 7.2
DEFAULT_BATCH_LIMIT = 10
DEFAULT_MAX_REQUESTS = 30


class RequestBudgetReached(BaseException):
    """Stop cleanly before exceeding the API request budget."""


def read_jsonl(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


class PacedResponses:
    """Apply pacing to every Responses API parse call."""
    def __init__(self, responses, max_requests):
        self.responses = responses
        self.last_request = 0.0
        self.max_requests = max_requests
        self.request_count = 0

    def parse(self, **kwargs):
        if self.request_count >= self.max_requests:
            raise RequestBudgetReached("Maximum API request budget reached.")
        elapsed = time.monotonic() - self.last_request
        if self.last_request and elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        self.last_request = time.monotonic()
        self.request_count += 1
        return self.responses.parse(**kwargs)


class PacedClient:
    def __init__(self, client, max_requests):
        self.responses = PacedResponses(client.responses, max_requests)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--limit", type=int, default=DEFAULT_BATCH_LIMIT)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    args = parser.parse_args()

    if args.limit < 1 or args.max_requests < 1:
        raise SystemExit("--limit and --max-requests must be positive.")

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    eval_ids = [str(x) for x in split["eval_ids"]]
    if len(eval_ids) != 300 or len(set(eval_ids)) != 300:
        raise SystemExit("Evaluation split is not exactly 300 unique IDs.")

    df = pd.read_csv(DATASET, dtype={"review_id": str, "product_id": str})
    df = df[df.review_id.isin(eval_ids)].copy()
    if df.review_id.duplicated().any() or set(df.review_id) != set(eval_ids):
        raise SystemExit("Dataset IDs do not match the evaluation cohort.")
    if not df.language.astype(str).str.casefold().eq("english").all():
        raise SystemExit("Non-English review found.")

    by_id = {str(r.review_id): r for _, r in df.iterrows()}

    completed = set()
    for line_no, record in enumerate(read_jsonl(OUT), 1):
        rec = ExtractionRecord.model_validate(record)
        row = by_id.get(rec.review_id)
        if row is None or rec.review_id in completed:
            raise SystemExit(f"Invalid or duplicate output record at line {line_no}.")
        validate_record(
            rec,
            expected_review_id=str(row.review_id),
            expected_product_id=str(row.product_id),
            expected_review_text=str(row.review_text),
        )
        completed.add(rec.review_id)

    failed_ids = []
    for record in read_jsonl(OLD_FAIL):
        rid = str(record["review_id"])
        if rid in by_id and rid not in failed_ids and rid not in completed:
            failed_ids.append(rid)

    recovery_done = set()
    for record in read_jsonl(FAIL):
        if record.get("status") == "success":
            recovery_done.add(str(record["review_id"]))

    todo_ids = [
        rid for rid in failed_ids
        if rid not in completed and rid not in recovery_done
    ]

    print("Model:", MODEL)
    print("Original successful annotations:", len(completed))
    print("Original failed IDs:", len(failed_ids))
    print("Remaining recovery candidates:", len(todo_ids))
    batch_ids = todo_ids[:args.limit]
    print("Selected batch size:", len(batch_ids))
    print("Maximum API requests:", args.max_requests)
    print("Mode:", "EXECUTION" if args.execute else "DRY RUN")
    print("Recovery output:", OUT.relative_to(ROOT))
    print("Recovery log:", FAIL.relative_to(ROOT))

    if not args.execute:
        print("\nDRY RUN ONLY. No API calls made.")
        print("Estimated pacing time:", round(args.max_requests * REQUEST_INTERVAL / 60, 1), "minutes maximum pacing allowance.")
        print("This excludes API processing time and quota waiting.")
        return

    if args.max_cost_usd is None or args.max_cost_usd <= 0:
        raise SystemExit("Supply a positive --max-cost-usd.")

    if not batch_ids:
        print("No failed reviews remain to recover.")
        return

    # Estimate from actual successful usage in the previous run.
    manifest_path = ROOT / "reports/ai_annotation_300/manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    usage = previous.get("token_usage", {})
    previous_count = previous.get("successes_this_run", 0)

    if not previous_count:
        raise SystemExit("Cannot estimate cost: previous successful usage unavailable.")

    in_avg = usage.get("input_tokens", 0) / previous_count
    out_avg = usage.get("output_tokens", 0) / previous_count
    estimate = len(batch_ids) * (
        in_avg * INPUT_RATE + out_avg * OUTPUT_RATE
    ) / 1_000_000

    print(f"Estimated cost for this recovery: ${estimate:.4f}")
    if estimate > args.max_cost_usd:
        raise SystemExit("Estimate exceeds spending threshold. No API calls made.")

    confirmation = input(
        f"Type RECOVER {len(batch_ids)} REVIEWS to authorize: "
    )
    if confirmation != f"RECOVER {len(batch_ids)} REVIEWS":
        raise SystemExit("Confirmation mismatch. No API calls made.")

    from openai import OpenAI
    client = PacedClient(
        OpenAI(timeout=60.0, max_retries=0),
        max_requests=args.max_requests,
    )

    successes = 0
    budget_stopped = False
    failures = 0
    input_tokens = 0
    output_tokens = 0
    started = datetime.now(timezone.utc).isoformat()

    for index, rid in enumerate(batch_ids, 1):
        row = by_id[rid]
        item = {
            "review_id": rid,
            "product_id": str(row.product_id),
            "review_text": str(row.review_text),
        }

        for attempt in range(1, 4):
            try:
                record, usage = extract_one(
                    item, client=client, strategy="few_shot_v2"
                )
                validate_record(
                    record,
                    expected_review_id=rid,
                    expected_product_id=item["product_id"],
                    expected_review_text=item["review_text"],
                )

                append(OUT, record.model_dump())
                append(FAIL, {
                    "review_id": rid,
                    "status": "success",
                    "recovered_at": datetime.now(timezone.utc).isoformat(),
                })
                successes += 1

                if usage:
                    input_tokens += usage.get("input_tokens", 0)
                    output_tokens += usage.get("output_tokens", 0)
                break

            except RequestBudgetReached:
                print("\\nRequest budget reached. Stopping without marking this review as failed.")
                budget_stopped = True
                break

            except RateLimitError as exc:
                append(FAIL, {
                    "review_id": rid,
                    "status": "rate_limited",
                    "error": str(exc),
                    "at": datetime.now(timezone.utc).isoformat(),
                })
                print("\nAPI rate limit reached. Stopping safely.")
                print("Successful recoveries this run:", successes)
                print("Run again later after the provider quota resets.")
                return

            except (APITimeoutError, APIConnectionError) as exc:
                if attempt < 3:
                    wait = 2 ** attempt * 5
                    print(f"{rid}: temporary connection error; retrying in {wait}s.")
                    time.sleep(wait)
                    continue
                append(FAIL, {
                    "review_id": rid,
                    "status": "temporary_failure",
                    "error": str(exc),
                    "at": datetime.now(timezone.utc).isoformat(),
                })
                failures += 1
                break

            except Exception as exc:
                append(FAIL, {
                    "review_id": rid,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "at": datetime.now(timezone.utc).isoformat(),
                })
                failures += 1
                break

        if index % 5 == 0 or index == len(batch_ids):
            print(
                f"Progress: {index}/{len(batch_ids)} | "
                f"recovered={successes} | failed={failures}"
            )

        if budget_stopped:
            break

    actual_cost = (
        input_tokens * INPUT_RATE + output_tokens * OUTPUT_RATE
    ) / 1_000_000

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "prompt_strategy": "few_shot_v2",
        "recovered_this_run": successes,
        "failures_this_run": failures,
        "token_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "estimated_cost_usd": round(actual_cost, 6),
        "annotation_type": "AI-generated silver annotations",
    }, indent=2), encoding="utf-8")

    print("Recovery finished.")
    print("Recovered:", successes)
    print("Failed:", failures)
    print("Estimated cost:", round(actual_cost, 6))


if __name__ == "__main__":
    main()

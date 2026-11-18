"""Resumable extraction CLI; defaults to a 10-review pilot, never a full run."""
import argparse, hashlib, json, logging, random, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from .config import (PROJECT_ROOT, MODEL, PROMPT_VERSION, SCHEMA_VERSION, API_KEY_PRESENT,
                     MODEL_INPUT_USD_PER_MILLION, MODEL_OUTPUT_USD_PER_MILLION)
from .llm_client import extract_one
from .schemas import ExtractionRecord
from .validators import validate_record
from .retry_handler import ExtractionFailure

DATASET = PROJECT_ROOT / "data/processed/amazon_reviews_cleaned.csv"
OUT_DIR = PROJECT_ROOT / "data/extracted"

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False)+"\n")


def export_predictions_csv(jsonl_path, csv_path):
    """Export validated extraction records as one CSV row per aspect.

    Reviews with no extracted aspects are retained with empty aspect fields.
    """
    jsonl_path = Path(jsonl_path)
    csv_path = Path(csv_path)

    columns = [
        "review_id", "product_id", "review_text", "extraction_status",
        "aspect_id", "aspect_term", "sentiment", "evidence",
        "complaint_present", "complaint_evidence",
    ]

    rows = []
    if jsonl_path.exists():
        for line_number, line in enumerate(
            jsonl_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue

            try:
                record = ExtractionRecord.model_validate_json(line)
            except Exception as exc:
                raise ValueError(
                    f"Invalid extraction record at {jsonl_path}:{line_number}: {exc}"
                ) from exc

            base = {
                "review_id": record.review_id,
                "product_id": record.product_id,
                "review_text": record.review_text,
                "extraction_status": record.extraction_status,
            }

            if not record.aspects:
                rows.append({
                    **base,
                    "aspect_id": None,
                    "aspect_term": None,
                    "sentiment": None,
                    "evidence": None,
                    "complaint_present": None,
                    "complaint_evidence": None,
                })
            else:
                for aspect in record.aspects:
                    rows.append({
                        **base,
                        "aspect_id": aspect.aspect_id,
                        "aspect_term": aspect.aspect_term,
                        "sentiment": aspect.sentiment,
                        "evidence": aspect.evidence,
                        "complaint_present": aspect.complaint_present,
                        "complaint_evidence": aspect.complaint_evidence,
                    })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False)
    return csv_path

def preflight(limit=10, reports_dir=None):
    """Validate local inputs and pilot selection without creating an API client."""
    if not 1 <= limit <= 10:
        raise SystemExit("Pilot limit must be between 1 and 10 reviews.")
    if not DATASET.is_file():
        raise SystemExit(f"Processed dataset not found: {DATASET}")
    df = pd.read_csv(DATASET)
    required = {"review_id", "product_id", "review_text", "language"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset missing columns: {sorted(missing)}")
    english = df[df.language.astype(str).str.casefold().eq("english")].copy()
    english = english.dropna(subset=["review_id", "product_id", "review_text"])
    english = english[english.review_text.astype(str).str.strip().ne("")]
    selected = english.head(limit)
    if len(selected) != min(limit, len(english)):
        raise SystemExit(f"Unable to select the expected {limit}-review pilot")
    prompt_path = PROJECT_ROOT / "prompts" / f"{PROMPT_VERSION}.txt"
    if not prompt_path.is_file() or not prompt_path.read_text(encoding="utf-8").strip():
        raise SystemExit(f"Versioned prompt missing or empty: {prompt_path}")
    result = {
        "status": "offline_preflight_passed",
        "api_call_made": False,
        "api_key_present": API_KEY_PRESENT,
        "dataset_path": str(DATASET.relative_to(PROJECT_ROOT)),
        "dataset_sha256": sha256(DATASET),
        "english_rows_available": int(len(english)),
        "pilot_size": int(len(selected)),
        "requested_limit": limit,
        "pilot_review_ids": selected.review_id.astype(str).tolist(),
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "model_configured": MODEL,
    }
    report_path = Path(reports_dir) if reports_dir else PROJECT_ROOT / "reports"
    manifest_path = report_path / "offline_preflight_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def english_eligible(dataset=None):
    """English, non-empty reviews in file order (the extraction universe)."""
    df = pd.read_csv(dataset or DATASET)
    required = {"review_id", "product_id", "review_text", "language"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset missing columns: {sorted(missing)}")
    df = df[df.language.astype(str).str.casefold().eq("english")].copy()
    df = df.dropna(subset=["review_id", "product_id", "review_text"])
    return df[df.review_text.astype(str).str.strip().ne("")]


def completed_ids(paths, source_by_id):
    """Review IDs with a schema- and source-valid record in any of ``paths``."""
    done = set()
    for path in paths:
        if not Path(path).exists():
            continue
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = ExtractionRecord.model_validate_json(line)
                src = source_by_id.get(rec.review_id)
                if src is None:
                    continue
                validate_record(rec, expected_review_id=str(src.review_id),
                                expected_product_id=str(src.product_id),
                                expected_review_text=str(src.review_text))
                done.add(rec.review_id)
            except Exception:
                continue
    return done



def _extract_with_transient_retry(row, client, sleep, strategy=None):
    """Returns ('ok', record, usage) | ('fail', failure_dict, None)."""
    for attempt in range(1, 4):
        try:
            kwargs = {"strategy": strategy} if strategy else {}
            record, usage = (
                extract_one(row, client=client, **kwargs)
                if client is not None
                else extract_one(row, **kwargs)
            )
            return "ok", record, usage

        except ExtractionFailure as exc:
            return "fail", {
                "review_id": row["review_id"],
                "category": exc.category,
                "attempts": exc.attempts,
                "error_type": type(exc).__name__,
                "error": str(exc)
            }, None

        except Exception as exc:
            name = type(exc).__name__
            status = getattr(exc, "status_code", None)

            if name in {"AuthenticationError", "PermissionDeniedError"} or status in (401, 403):
                raise SystemExit(
                    f"Authentication/permission failed; stop run: {exc}"
                ) from exc

            transient = (
                name in {"RateLimitError", "APITimeoutError", "APIConnectionError"}
                or status == 429
                or (isinstance(status, int) and status >= 500)
            )

            if not transient or attempt >= 3:
                return "fail", {
                    "review_id": row["review_id"],
                    "category": "transient_api" if transient else "api_or_extraction_error",
                    "attempts": attempt,
                    "error_type": name,
                    "error": str(exc)
                }, None

            sleep(min(20, 2 ** attempt + random.random()))

    return "fail", {
        "review_id": row["review_id"],
        "category": "unexpected",
        "attempts": 0,
        "error_type": "Unreachable",
        "error": "retry loop exited"
    }, None



def run(limit=10, pilot=True, *, full=False, client=None, dataset=None, out_dir=None,
        reports_dir=None, workers=1, sleep=time.sleep, require_key=True,
        strategy=None):
    """Pilot (fixed 10 reviews) or, once approved, full resumable extraction.

    ``client``/``dataset``/``out_dir``/``reports_dir`` are injectable so the whole
    path can be tested offline with a fake client.
    """
    from .prompt_strategies import load_prompt, PROMPT_FILES

    if strategy is not None:
        if strategy not in PROMPT_FILES:
            raise SystemExit(f"Unknown strategy: {strategy}")
        load_prompt(strategy)
    dataset = Path(dataset) if dataset else DATASET
    base_out_dir = Path(out_dir) if out_dir else OUT_DIR
    if strategy is not None:
        out_dir = base_out_dir / strategy
    else:
        out_dir = base_out_dir
    base_reports_dir = Path(reports_dir) if reports_dir else PROJECT_ROOT / "reports"
    reports_dir = base_reports_dir / strategy if strategy is not None else base_reports_dir
    if full:
        pilot = False
    elif not pilot or not 1 <= limit <= 10:
        raise SystemExit("Pilot limit must be between 1 and 10 reviews; use --full after pilot approval.")
    if require_key and client is None and not API_KEY_PRESENT:
        raise SystemExit("OPENAI_API_KEY missing. Add it to .env; no API call was made.")
    if full:
        from .approval import require_pilot_approval
        approval_out_dir = base_out_dir if strategy is None else base_out_dir
        require_pilot_approval(
            reports_dir=reports_dir,
            out_dir=approval_out_dir,
            dataset=dataset,
            strategy=strategy,
        )

    df = english_eligible(dataset)
    cohort = df.head(limit).copy() if pilot else df.copy()
    stem = "pilot" if pilot else "full"
    out = out_dir / ("pilot_predictions.jsonl" if pilot else "predictions.jsonl")
    failures = out_dir / ("pilot_failures.jsonl" if pilot else "failures.jsonl")
    source_by_id = {str(r.review_id): r for _, r in cohort.iterrows()}
    paths = [out] if pilot else [out, out_dir / "pilot_predictions.jsonl"]
    completed = completed_ids(paths, source_by_id)
    pilot_reused = (len(completed_ids([out_dir / "pilot_predictions.jsonl"], source_by_id))
                    if not pilot else 0)
    todo = cohort[~cohort.review_id.astype(str).isin(completed)]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.time(); started_iso = datetime.now(timezone.utc).isoformat()
    successes = no_aspects = failed = 0
    tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    rows = [{"review_id": str(r.review_id), "product_id": str(r.product_id),
             "review_text": str(r.review_text)} for _, r in todo.iterrows()]

    def handle(kind, payload, usage):
        nonlocal successes, no_aspects, failed
        if kind == "ok":
            append_jsonl(out, payload.model_dump()); successes += 1
            if not payload.aspects: no_aspects += 1
            for k in tokens: tokens[k] += (usage or {}).get(k) or 0
        else:
            append_jsonl(failures, payload); failed += 1

    if workers <= 1:
        for i, row in enumerate(rows, 1):
            handle(*_extract_with_transient_retry(row, client, sleep, strategy))
            if i % 25 == 0: logging.info("processed %d/%d", i, len(rows))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
    ex.submit(_extract_with_transient_retry, r, client, sleep, strategy)
    for r in rows
]
            try:
                for i, f in enumerate(as_completed(futs), 1):
                    handle(*f.result())
                    if i % 25 == 0: logging.info("processed %d/%d", i, len(rows))
            except BaseException:
                for f in futs: f.cancel()
                raise
    elapsed = round(time.time() - started, 2)
    done_now = completed_ids(paths, source_by_id)
    outstanding = sorted(set(source_by_id) - done_now)
    csv_out = export_predictions_csv(out, out.with_suffix(".csv"))
    manifest = {"mode": stem, "started_at_utc": started_iso, "model": MODEL,
        "prompt_version": (
    PROMPT_FILES[strategy] if strategy else PROMPT_VERSION
),
"prompt_strategy": strategy or "baseline",
"schema_version": SCHEMA_VERSION,
        "dataset_path": str(dataset.relative_to(PROJECT_ROOT)) if dataset.is_relative_to(PROJECT_ROOT) else str(dataset),
        "dataset_sha256": sha256(dataset),
        "english_rows_available": len(df), "requested_limit": limit if pilot else None,
        "eligible_reviews": len(cohort), "skipped_already_completed": len(completed),
        "reused_from_pilot": pilot_reused, "processed_this_run": len(todo),
        "successes_this_run": successes, "failures_this_run": failed,
        "no_aspect_successes": no_aspects, "completed_total": len(done_now),
        "outstanding_review_ids": outstanding, "complete": not outstanding,
        "token_usage_this_run": tokens,
        "estimated_api_cost_usd_at_configured_prices": round((tokens["input_tokens"]*MODEL_INPUT_USD_PER_MILLION
            + tokens["output_tokens"]*MODEL_OUTPUT_USD_PER_MILLION)/1_000_000, 6),
        "elapsed_seconds": elapsed,
        "average_latency_seconds": round(elapsed/len(todo), 2) if len(todo) else None,
        "workers": workers,
        "output_jsonl": str(out.relative_to(PROJECT_ROOT)) if out.is_relative_to(PROJECT_ROOT) else str(out),
        "output_csv": str(csv_out.relative_to(PROJECT_ROOT)) if csv_out.is_relative_to(PROJECT_ROOT) else str(csv_out),
        "failure_jsonl": str(failures.relative_to(PROJECT_ROOT)) if failures.is_relative_to(PROJECT_ROOT) else str(failures)}
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{stem}_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__=="__main__":
    ap=argparse.ArgumentParser(description="ABSA extraction: pilot by default; --full only after pilot approval")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out-dir", default="data/ai_annotation_pilot_5")
    ap.add_argument("--reports-dir", default="reports/ai_annotation_pilot_5")
    ap.add_argument("--preflight",action="store_true",help="Validate local setup without making API calls")
    ap.add_argument("--full",action="store_true",help="Process all remaining English reviews after matching pilot approval")
    ap.add_argument("--workers",type=int,default=1,help="Parallel requests for --full")
    ap.add_argument(
    "--strategy",
    choices=["zero_shot", "few_shot", "few_shot_v2", "cot"],
    default=None,
    help="Prompt strategy; default uses the original baseline prompt"
)

    ap.add_argument("--approve-pilot",action="store_true",help="Record human approval of the reviewed pilot")
    ap.add_argument("--reviewer",default=None,help="Name of the person approving the pilot")
    ap.add_argument("--notes",default="",help="Approval notes")
    args=ap.parse_args()
    if args.strategy is not None:
        from .prompt_strategies import load_prompt
        try:
            load_prompt(args.strategy)
        except (ValueError, FileNotFoundError) as exc:
            raise SystemExit(str(exc))

    if args.preflight:
        preflight(limit=args.limit, reports_dir=args.reports_dir)
        raise SystemExit(0)
    if args.approve_pilot:
        from .approval import approve_pilot
        if not args.reviewer: raise SystemExit("--reviewer NAME is required")
        if args.strategy:
            from pathlib import Path
            approve_pilot(
                reviewer=args.reviewer,
                notes=args.notes,
                reports_dir=Path(args.reports_dir) / args.strategy,
                out_dir=Path(args.out_dir) / args.strategy,
                strategy=args.strategy,
            )
        else:
            approve_pilot(reviewer=args.reviewer, notes=args.notes)
        raise SystemExit(0)
    if args.full:
        if args.strategy:
            run(
                full=True,
                workers=max(1, args.workers),
                strategy=args.strategy,
                out_dir=args.out_dir,
                reports_dir=args.reports_dir,
            )
        else:
            run(full=True, workers=max(1, args.workers))
        raise SystemExit(0)

    if not 1 <= args.limit <= 10:
        print("Pilot limit must be between 1 and 10 reviews.")
        raise SystemExit(2)

    run(limit=args.limit, pilot=True, strategy=args.strategy, out_dir=args.out_dir, reports_dir=args.reports_dir)
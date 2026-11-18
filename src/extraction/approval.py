"""Human pilot-approval gate: --full refuses to run until a reviewed pilot is approved."""
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import PROJECT_ROOT, MODEL, PROMPT_VERSION, SCHEMA_VERSION

APPROVAL_FILE = "pilot_approval.json"


def _pipeline():
    from . import pipeline  # late import: avoids circular import
    return pipeline


def _prompt_version(strategy=None):
    if strategy is None:
        return PROMPT_VERSION
    from .prompt_strategies import PROMPT_FILES, load_prompt

    load_prompt(strategy)
    return PROMPT_FILES[strategy]


def approve_pilot(
    *, reviewer, notes="", reports_dir=None, out_dir=None, dataset=None,
    strategy=None,
):
    """Record approval. Requires 10 valid pilot predictions and no outstanding pilot failures."""
    pl = _pipeline()
    reports_dir = Path(reports_dir) if reports_dir else PROJECT_ROOT / "reports"
    out_dir = Path(out_dir) if out_dir else pl.OUT_DIR
    dataset = Path(dataset) if dataset else pl.DATASET
    df = pl.english_eligible(dataset).head(10)
    by_id = {str(r.review_id): r for _, r in df.iterrows()}
    done = pl.completed_ids([out_dir / "pilot_predictions.jsonl"], by_id)
    if done != set(by_id):
        raise SystemExit(f"Cannot approve: only {len(done)}/{len(by_id)} pilot reviews have valid predictions.")
    record = {"approved_by": reviewer, "approved_at_utc": datetime.now(timezone.utc).isoformat(),
              "notes": notes, "model": MODEL, "prompt_version": _prompt_version(strategy),
              "prompt_strategy": strategy or "baseline",
              "schema_version": SCHEMA_VERSION, "dataset_sha256": pl.sha256(dataset),
              "pilot_review_ids": sorted(by_id), "pilot_valid_records": len(done)}
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / APPROVAL_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return record


def require_pilot_approval(*, reports_dir=None, out_dir=None, dataset=None, strategy=None):
    pl = _pipeline()
    reports_dir = Path(reports_dir) if reports_dir else PROJECT_ROOT / "reports"
    dataset = Path(dataset) if dataset else pl.DATASET
    path = reports_dir / APPROVAL_FILE
    if not path.is_file():
        if strategy:
            guidance = (
                f"--strategy {strategy} --approve-pilot --reviewer NAME "
                "with the same --out-dir and --reports-dir used for its pilot"
            )
        else:
            guidance = (
                "--approve-pilot --reviewer NAME after reviewing reports/pilot_review.md"
            )
        raise SystemExit(
            "Full extraction blocked: no pilot approval matching this prompt. Review that "
            f"strategy's pilot, then run `python -m src.extraction.pipeline {guidance}`."
        )
    rec = json.loads(path.read_text(encoding="utf-8"))
    current = {"model": MODEL, "prompt_version": _prompt_version(strategy),
               "schema_version": SCHEMA_VERSION, "dataset_sha256": pl.sha256(dataset)}
    stale = [k for k, v in current.items() if rec.get(k) != v]
    if stale:
        raise SystemExit(f"Pilot approval is stale (changed: {stale}). Re-run and re-approve the pilot.")
    return rec

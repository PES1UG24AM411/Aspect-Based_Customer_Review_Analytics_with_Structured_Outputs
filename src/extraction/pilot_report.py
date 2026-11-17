"""Generate reports/pilot_review.md: automated checks plus a human review checklist.

Semantic correctness (hallucinated aspects, missed aspects, wrong sentiment, wrong
complaint flag) cannot be verified automatically; this report lays out each review
next to its extraction so a person can judge them, then approve via the CLI.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from .config import PROJECT_ROOT, MODEL, PROMPT_VERSION
from .json_utils import read_jsonl
from .schemas import ExtractionRecord
from .validators import validate_record


def build_pilot_report(out_dir=None, reports_dir=None):
    out_dir = Path(out_dir) if out_dir else PROJECT_ROOT / "data/extracted"
    reports_dir = Path(reports_dir) if reports_dir else PROJECT_ROOT / "reports"
    rows = read_jsonl(out_dir / "pilot_predictions.jsonl")
    fails = read_jsonl(out_dir / "pilot_failures.jsonl")
    if not rows:
        raise SystemExit("No pilot predictions found; run the pilot first.")
    recs, bad = [], []
    for r in rows:
        try:
            rec = ExtractionRecord.model_validate(r)
            validate_record(rec, expected_review_id=rec.review_id, expected_product_id=rec.product_id,
                            expected_review_text=rec.review_text)
            recs.append(rec)
        except Exception as exc:
            bad.append((r.get("review_id"), str(exc)))
    sent = Counter(a.sentiment for r in recs for a in r.aspects)
    n_asp = sum(len(r.aspects) for r in recs)
    n_cmp = sum(a.complaint_present for r in recs for a in r.aspects)
    terms = Counter(a.aspect_term.lower() for r in recs for a in r.aspects)
    L = [f"# Pilot review ({len(recs)} reviews)", "",
         f"Model: `{MODEL}` | Prompt: `{PROMPT_VERSION}`", "",
         "## Automated checks (structural only)", "",
         f"- Schema-valid and source-faithful records: {len(recs)}/{len(rows)}",
         f"- Invalid records: {len(bad)}", f"- Logged failures: {len(fails)}",
         f"- Reviews with zero aspects: {sum(not r.aspects for r in recs)}",
         f"- Total aspects: {n_asp}; complaints flagged: {n_cmp}",
         f"- Sentiment distribution: {dict(sent)}",
         f"- Most common aspect terms: {terms.most_common(10)}", ""]
    for rid, err in bad: L.append(f"- INVALID `{rid}`: {err}")
    L += ["", "## Manual review (fill in before approving)", "",
          "For each review mark: hallucinated aspect? missed aspect? wrong sentiment? wrong complaint flag?", ""]
    for r in recs:
        L += [f"### {r.review_id} (product {r.product_id})", "", f"> {r.review_text[:1500]}", ""]
        if not r.aspects: L.append("_No aspects extracted._")
        for a in r.aspects:
            L.append(f"- **{a.aspect_term}** — {a.sentiment}; complaint={a.complaint_present}; evidence: \"{a.evidence}\"")
        L += ["", "- [ ] hallucination  - [ ] missed aspect  - [ ] wrong sentiment  - [ ] wrong complaint  - Notes:", ""]
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "pilot_review.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    build_pilot_report()

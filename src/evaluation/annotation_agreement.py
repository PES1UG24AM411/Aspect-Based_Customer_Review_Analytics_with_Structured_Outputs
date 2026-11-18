"""Inter-annotator agreement for independently labelled ABSA reviews."""

import argparse
import json
from pathlib import Path

from src.evaluation.metrics import match_review, cohen_kappa, confusion


SENTIMENT_LABELS = ["positive", "negative", "neutral"]
COMPLAINT_LABELS = [False, True]


def load_jsonl(path):
    """Load JSONL records keyed by review_id; reject duplicate IDs."""
    records = {}

    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            review_id = record["review_id"]

            if review_id in records:
                raise ValueError(
                    f"Duplicate review_id {review_id} in {path}, "
                    f"line {line_number}"
                )

            records[review_id] = record

    return records


def calculate_agreement(
    annotator_1,
    annotator_2,
    shared_review_ids,
    mode="lenient",
):
    """Calculate agreement on completed, shared reviews only."""

    if len(shared_review_ids) != len(set(shared_review_ids)):
        raise ValueError("Shared review IDs contain duplicates.")

    missing = []

    for rid in shared_review_ids:
        if rid not in annotator_1:
            missing.append(f"{rid}: missing from annotator 1")
        elif rid not in annotator_2:
            missing.append(f"{rid}: missing from annotator 2")
        else:
            if annotator_1[rid]["status"] != "complete":
                missing.append(f"{rid}: annotator 1 has not completed it")
            if annotator_2[rid]["status"] != "complete":
                missing.append(f"{rid}: annotator 2 has not completed it")

    if missing:
        raise ValueError(
            "Agreement cannot be calculated yet. Incomplete shared reviews:\n"
            + "\n".join(missing)
        )

    sentiment_1, sentiment_2 = [], []
    complaint_1, complaint_2 = [], []

    per_review = []
    disagreements = []

    total_matched = 0
    total_unmatched_1 = 0
    total_unmatched_2 = 0

    for rid in shared_review_ids:
        a = annotator_1[rid]
        b = annotator_2[rid]

        if a["review_text"] != b["review_text"]:
            raise ValueError(f"Review text differs for {rid}.")

        pairs, unmatched_2, unmatched_1 = match_review(
            a["review_text"],
            a["aspects"],
            b["aspects"],
            mode=mode,
        )

        total_matched += len(pairs)
        total_unmatched_1 += len(unmatched_1)
        total_unmatched_2 += len(unmatched_2)

        per_review.append({
            "review_id": rid,
            "matched_aspects": len(pairs),
            "unmatched_annotator_1": len(unmatched_1),
            "unmatched_annotator_2": len(unmatched_2),
        })

        for i, j in pairs:
            aspect_a = a["aspects"][i]
            aspect_b = b["aspects"][j]

            s1 = aspect_a["sentiment"]
            s2 = aspect_b["sentiment"]

            c1 = bool(aspect_a["complaint_present"])
            c2 = bool(aspect_b["complaint_present"])

            if s1 not in SENTIMENT_LABELS or s2 not in SENTIMENT_LABELS:
                raise ValueError(f"Invalid sentiment label in review {rid}.")

            sentiment_1.append(s1)
            sentiment_2.append(s2)
            complaint_1.append(c1)
            complaint_2.append(c2)

            if s1 != s2 or c1 != c2:
                disagreements.append({
                    "review_id": rid,
                    "aspect_annotator_1": aspect_a["aspect_term"],
                    "aspect_annotator_2": aspect_b["aspect_term"],
                    "sentiment_annotator_1": s1,
                    "sentiment_annotator_2": s2,
                    "complaint_annotator_1": c1,
                    "complaint_annotator_2": c2,
                })

    sentiment_n = len(sentiment_1)
    complaint_n = len(complaint_1)

    sentiment_accuracy = (
        round(
            sum(x == y for x, y in zip(sentiment_1, sentiment_2))
            / sentiment_n,
            4,
        )
        if sentiment_n else None
    )

    complaint_accuracy = (
        round(
            sum(x == y for x, y in zip(complaint_1, complaint_2))
            / complaint_n,
            4,
        )
        if complaint_n else None
    )

    return {
        "method": {
            "aspect_matching": "existing match_review function",
            "matching_mode": mode,
            "sentiment_metric": "Cohen's kappa",
            "matching_unit": "matched aspects within shared reviews",
        },
        "review_summary": {
            "shared_reviews": len(shared_review_ids),
            "completed_reviews": len(shared_review_ids),
        },
        "aspect_summary": {
            "matched_aspects": total_matched,
            "unmatched_annotator_1": total_unmatched_1,
            "unmatched_annotator_2": total_unmatched_2,
        },
        "sentiment_agreement": {
            "n": sentiment_n,
            "accuracy": sentiment_accuracy,
            "cohen_kappa": cohen_kappa(
                sentiment_1, sentiment_2, SENTIMENT_LABELS
            ),
            "confusion_matrix": confusion(
                sentiment_1, sentiment_2, SENTIMENT_LABELS
            ),
        },
        "complaint_agreement": {
            "n": complaint_n,
            "accuracy": complaint_accuracy,
            "cohen_kappa": cohen_kappa(
                complaint_1, complaint_2, COMPLAINT_LABELS
            ),
            "confusion_matrix": confusion(
                complaint_1, complaint_2, COMPLAINT_LABELS
            ),
        },
        "per_review": per_review,
        "disagreements": disagreements,
        "note": (
            "Agreement scores do not establish gold labels. "
            "Unmatched aspects and disagreements require human review "
            "before adjudication."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate inter-annotator agreement."
    )
    parser.add_argument(
        "--mode",
        choices=["strict", "lenient"],
        default="lenient",
    )
    parser.add_argument(
        "--output",
        default="reports/annotation_agreement/agreement_report.json",
    )
    args = parser.parse_args()

    base = Path("data/evaluation")

    assignment = json.loads(
        (base / "annotation_assignment.json").read_text(encoding="utf-8")
    )

    annotator_1 = load_jsonl(base / assignment["annotator_1_file"])
    annotator_2 = load_jsonl(base / assignment["annotator_2_file"])

    report = calculate_agreement(
        annotator_1,
        annotator_2,
        assignment["shared_review_ids"],
        mode=args.mode,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Agreement report created:", output)
    print("Shared reviews:", report["review_summary"]["shared_reviews"])
    print("Matched aspects:", report["aspect_summary"]["matched_aspects"])
    print("Sentiment accuracy:", report["sentiment_agreement"]["accuracy"])
    print("Sentiment Cohen's kappa:",
          report["sentiment_agreement"]["cohen_kappa"])
    print("Complaint accuracy:", report["complaint_agreement"]["accuracy"])
    print("Complaint Cohen's kappa:",
          report["complaint_agreement"]["cohen_kappa"])


if __name__ == "__main__":
    main()


import json
from pathlib import Path
from collections import Counter

path = Path("data/evaluation/dev_annotations.jsonl")

rows = []
errors = []

# Read and parse JSONL
for line_number, line in enumerate(
    path.read_text(encoding="utf-8").splitlines(), 1
):
    if not line.strip():
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError as exc:
        errors.append(f"Line {line_number}: invalid JSON - {exc}")

# Check record count and IDs
if len(rows) != 20:
    errors.append(f"Expected 20 records; found {len(rows)}")

ids = [r.get("review_id") for r in rows]

if len(set(ids)) != len(ids):
    errors.append("Duplicate review IDs found")

# Validate each record
for i, record in enumerate(rows, 1):
    if record.get("status") != "complete":
        errors.append(f"Record {i}: status is not complete")

    review_text = record.get("review_text")

    if not isinstance(review_text, str):
        errors.append(f"Record {i}: review_text is missing or invalid")
        review_text = ""

    aspects = record.get("aspects")

    if not isinstance(aspects, list):
        errors.append(f"Record {i}: aspects is not a list")
        continue

    for j, aspect in enumerate(aspects, 1):
        if not isinstance(aspect, dict):
            errors.append(f"Record {i}, aspect {j}: invalid aspect object")
            continue

        required = (
            "aspect_term",
            "sentiment",
            "evidence",
            "complaint_present",
        )

        for field in required:
            if field not in aspect:
                errors.append(
                    f"Record {i}, aspect {j}: missing {field}"
                )

        if not isinstance(aspect.get("aspect_term"), str) or not aspect.get("aspect_term"):
            errors.append(f"Record {i}, aspect {j}: invalid aspect_term")

        if aspect.get("sentiment") not in (
            "positive", "negative", "neutral"
        ):
            errors.append(f"Record {i}, aspect {j}: invalid sentiment")

        if type(aspect.get("complaint_present")) is not bool:
            errors.append(
                f"Record {i}, aspect {j}: complaint_present is not boolean"
            )

        evidence = aspect.get("evidence")

        if (
            not isinstance(evidence, str)
            or not evidence
            or evidence not in review_text
        ):
            errors.append(
                f"Record {i}, aspect {j}: evidence is not an exact review-text span"
            )

# Report
print("\nDEVELOPMENT ANNOTATION AUDIT")
print("----------------------------")
print("Records:", len(rows))
print("Unique review IDs:", len(set(ids)))
print("Completed records:", sum(
    r.get("status") == "complete" for r in rows
))
print("Total aspects:", sum(
    len(r.get("aspects", []))
    for r in rows
    if isinstance(r.get("aspects"), list)
))

annotators = Counter(
    str(r.get("annotator"))
    for r in rows
)
print("Annotator field counts:", dict(annotators))

print("\nAudit errors:", len(errors))

for error in errors:
    print("-", error)

if errors:
    print("\nAUDIT NEEDS ATTENTION")
else:
    print("\nAUDIT PASSED")

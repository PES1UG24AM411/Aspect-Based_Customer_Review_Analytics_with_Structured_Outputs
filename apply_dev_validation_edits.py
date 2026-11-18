import json
import os
from pathlib import Path

folder = Path("data/evaluation")
active_file = folder / "dev_annotations.jsonl"
backup_file = folder / "dev_annotations_before_validation.jsonl"
temp_file = folder / "dev_annotations.jsonl.tmp"

if not active_file.exists() or not backup_file.exists():
    raise FileNotFoundError("The active file or validation backup is missing.")

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

rows = load_jsonl(active_file)
backup_rows = load_jsonl(backup_file)

if len(rows) != 20 or len(backup_rows) != 20:
    raise ValueError("Expected exactly 20 development records.")

if [r["review_id"] for r in rows] != [r["review_id"] for r in backup_rows]:
    raise ValueError("Review IDs or record order differ from the backup.")

# Ensure the original review text has not changed.
for current, original in zip(rows, backup_rows):
    if current["review_text"] != original["review_text"]:
        raise ValueError(f"Review text changed: {current['review_id']}")

def find_aspect(record, term):
    matches = [
        a for a in record["aspects"]
        if a["aspect_term"].strip().casefold() == term.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{record['review_id']}: expected one '{term}' aspect, "
            f"found {len(matches)}"
        )
    return matches[0]

def remove_aspect(record, term):
    aspect = find_aspect(record, term)
    record["aspects"].remove(aspect)
    print(f"{record['review_id']}: removed {term}")

by_id = {r["review_id"]: r for r in rows}

# AMZ-000001: use a shorter exact evidence span.
a = find_aspect(by_id["AMZ-000001"], "Battery health")
a["evidence"] = "it's now at 90% battery health"
print("AMZ-000001: shortened Battery health evidence")

# AMZ-000002: remove the broad Phone condition aspect;
# the specific Screen condition aspect remains.
remove_aspect(by_id["AMZ-000002"], "Phone condition")

# AMZ-000003: remove the broad aspect that overlaps with specifics.
remove_aspect(by_id["AMZ-000003"], "Overall phone quality")

# AMZ-000005: standardize aspect-term spelling.
find_aspect(by_id["AMZ-000005"], "Battery Health")[
    "aspect_term"
] = "Battery health"

find_aspect(by_id["AMZ-000005"], "Price / value")[
    "aspect_term"
] = "Price/value"

print("AMZ-000005: standardized two aspect names")

# Validate evidence for every non-empty evidence span.
evidence_problems = []
for record in rows:
    for aspect in record["aspects"]:
        evidence = aspect.get("evidence")
        if evidence and evidence not in record["review_text"]:
            evidence_problems.append(
                (record["review_id"], aspect["aspect_term"], evidence)
            )

# Write to a temporary file first, then replace the active file.
with temp_file.open("w", encoding="utf-8", newline="\n") as f:
    for record in rows:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

os.replace(temp_file, active_file)

# Reload the saved file and verify it is valid JSONL.
saved_rows = load_jsonl(active_file)
assert len(saved_rows) == 20
assert len({r["review_id"] for r in saved_rows}) == 20

for current, original in zip(saved_rows, backup_rows):
    assert current["review_id"] == original["review_id"]
    assert current["review_text"] == original["review_text"]

print("\nSUCCESS: development file saved and verified.")
print("Records:", len(saved_rows))
print("Evidence spans not found in original text:", len(evidence_problems))

for problem in evidence_problems:
    print("CHECK:", problem)
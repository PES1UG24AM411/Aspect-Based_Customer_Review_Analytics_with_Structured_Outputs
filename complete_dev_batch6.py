
import json
import shutil
from pathlib import Path

DATA = Path("data/evaluation/dev_annotations.jsonl")
BACKUP = Path("data/evaluation/dev_annotations_before_batch6.jsonl")

annotations = {
    "AMZ-001126": [
        (
            "Device blacklist status",
            "negative",
            'the phone has been reported "lost/stolen" in their system',
            True,
        ),
        (
            "Device unlocking",
            "negative",
            "the only person authorized to unlock it is the original account holder.",
            True,
        ),
        (
            "Network connectivity",
            "negative",
            "no cell service",
            True,
        ),
        (
            "Seller responsiveness",
            "positive",
            "The seller has been very responsive over Amazon chat and provided a return shipping label",
            False,
        ),
    ],
    "AMZ-001292": [
        (
            "App usability",
            "positive",
            "The app has nice features and is easy to use.",
            False,
        ),
        (
            "Sound quality",
            "positive",
            "Tuning them to your ears seems to improve SQ a bit.",
            False,
        ),
        (
            "Noise cancellation",
            "positive",
            "Noise cancellation is effective and smart.",
            False,
        ),
        (
            "Battery life",
            "positive",
            "Battery easily lasts 3 hours",
            False,
        ),
        (
            "Fit and comfort",
            "positive",
            "Fit could be better but is very good considering it's the first product I found, the price, and the fact that they're made to fit Everyone.",
            False,
        ),
        (
            "Charging case design",
            "negative",
            "Plopping these yucky drenched earbuds into their charging container and then shutting it lets them sit in their stew.",
            True,
        ),
    ],
    "AMZ-001300": [
        (
            "Product version accuracy",
            "negative",
            "They say it's the US version (SM-928U) but it's a SM-928W",
            True,
        ),
        (
            "Digital wallet compatibility",
            "negative",
            "digital wallet issues",
            True,
        ),
        (
            "Software update compatibility",
            "negative",
            "software update issues",
            True,
        ),
        (
            "Network band compatibility",
            "negative",
            "mmwave bands missing",
            True,
        ),
        (
            "Cosmetic condition",
            "positive",
            "it looked brand new",
            False,
        ),
    ],
}

with DATA.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

assert len(rows) == 20, f"Expected 20 records, found {len(rows)}"

assert not BACKUP.exists(), (
    f"Backup already exists: {BACKUP}. "
    "Do not overwrite an existing backup."
)

by_id = {r["review_id"]: r for r in rows}
assert len(by_id) == 20, "Duplicate review IDs detected"

# Validate records and evidence before changing anything.
for review_id, aspects in annotations.items():
    assert review_id in by_id, f"Missing review: {review_id}"

    record = by_id[review_id]

    assert record["status"] == "todo", (
        f"{review_id} is not todo; refusing to overwrite"
    )
    assert record.get("annotator") is None, (
        f"{review_id} already has an annotator"
    )

    for name, sentiment, evidence, complaint in aspects:
        assert evidence in record["review_text"], (
            f"Evidence not found: {review_id} / {name}: {evidence}"
        )
        assert sentiment in ("positive", "negative", "neutral")
        assert isinstance(complaint, bool)

# Create a fresh backup.
shutil.copy2(DATA, BACKUP)

# Apply annotations, leaving annotator blank.
for review_id, aspects in annotations.items():
    record = by_id[review_id]

    record["annotator"] = None
    record["status"] = "complete"

    record["aspects"] = [
        {
            "aspect_term": name,
            "sentiment": sentiment,
            "evidence": evidence,
            "complaint_present": complaint,
        }
        for name, sentiment, evidence, complaint in aspects
    ]

# Save JSONL.
with DATA.open("w", encoding="utf-8", newline="\n") as f:
    for record in rows:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Reload and verify saved data.
with DATA.open(encoding="utf-8") as f:
    verified = [json.loads(line) for line in f if line.strip()]

assert len(verified) == 20

verified_by_id = {r["review_id"]: r for r in verified}
assert len(verified_by_id) == 20

for review_id, expected_aspects in annotations.items():
    record = verified_by_id[review_id]

    assert record["status"] == "complete"
    assert record["annotator"] is None
    assert len(record["aspects"]) == len(expected_aspects)

    for aspect in record["aspects"]:
        assert aspect["evidence"] in record["review_text"]

completed = sum(
    r["status"] == "complete" for r in verified
)

print("SUCCESS: Batch 6 saved and verified.")

for review_id in annotations:
    record = verified_by_id[review_id]
    print(
        review_id,
        "| annotator:", repr(record["annotator"]),
        "| aspects:", len(record["aspects"]),
        "| status:", record["status"],
    )

print("Total development records:", len(verified))
print("Completed development records:", completed)
print("Remaining development records:", 20 - completed)
print("Backup:", BACKUP)

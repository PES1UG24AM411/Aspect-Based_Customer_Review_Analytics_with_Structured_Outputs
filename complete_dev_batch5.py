
import json
import shutil
from pathlib import Path

DATA = Path("data/evaluation/dev_annotations.jsonl")
BACKUP = Path("data/evaluation/dev_annotations_before_batch5.jsonl")

annotations = {
    "AMZ-000418": [
        ("Device unlocking", "negative",
         "AT&T was also unable to unlock the device", True),
        ("Color accuracy", "negative",
         "the wrong color phone was sent", True),
        ("Listing accuracy", "negative",
         "with a listing that is completely false and misleading.", True),
        ("Customer service", "negative",
         "they directed me to Samsung, who in turn told me they couldn't assist and referred me to AT&T.", True),
    ],
    "AMZ-000447": [
        ("Battery life", "negative",
         "The battery drains extremely fast", True),
        ("Charging speed", "negative",
         "it takes forever to charge.", True),
        ("Overheating", "negative",
         "the screen gets very hot even when I’m just scrolling through social media or Chrome", True),
        ("Return experience", "negative",
         "I now have to return it and be without a phone for who knows how long while waiting for the refund process.", True),
    ],
    "AMZ-000500": [
        ("Packaging protection", "negative",
         "there was nothing packed in the box to keep it from being shook around.", True),
        ("Phone box condition", "negative",
         "the phone box was also in a not great state.", True),
        ("Internal phone rattle", "neutral",
         "The phone has a small rattle in it", False),
        ("Phone functionality", "positive",
         "but seems to function well.", False),
    ],
}

with DATA.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

assert len(rows) == 20, f"Expected 20 records, got {len(rows)}"
assert not BACKUP.exists(), f"Backup already exists: {BACKUP}"

by_id = {r["review_id"]: r for r in rows}
assert len(by_id) == 20, "Duplicate review IDs detected"

for review_id, aspects in annotations.items():
    assert review_id in by_id, f"Missing review: {review_id}"
    record = by_id[review_id]
    assert record["status"] == "todo", f"{review_id} is not todo"
    assert record.get("annotator") is None, (
        f"{review_id} already has an annotator"
    )

    for name, sentiment, evidence, complaint in aspects:
        assert evidence in record["review_text"], (
            f"Evidence not found: {review_id}: {name}"
        )
        assert sentiment in ("positive", "negative", "neutral")

# Preserve the original before editing.
shutil.copy2(DATA, BACKUP)

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

with DATA.open("w", encoding="utf-8", newline="\n") as f:
    for record in rows:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Reload and verify.
with DATA.open(encoding="utf-8") as f:
    verified = [json.loads(line) for line in f if line.strip()]

assert len(verified) == 20
verified_by_id = {r["review_id"]: r for r in verified}

for review_id, aspects in annotations.items():
    record = verified_by_id[review_id]
    assert record["status"] == "complete"
    assert record["annotator"] is None
    assert len(record["aspects"]) == len(aspects)

    for aspect in record["aspects"]:
        assert aspect["evidence"] in record["review_text"]

print("SUCCESS: Batch 5 saved and verified.")

for review_id in annotations:
    record = verified_by_id[review_id]
    print(
        review_id,
        "| annotator:", repr(record["annotator"]),
        "| aspects:", len(record["aspects"]),
        "| status:", record["status"],
    )

completed = sum(r["status"] == "complete" for r in verified)

print("Total development records:", len(verified))
print("Completed development records:", completed)
print("Remaining development records:", 20 - completed)
print("Backup:", BACKUP)

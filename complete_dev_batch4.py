
import json
import shutil
from pathlib import Path

DATA = Path("data/evaluation/dev_annotations.jsonl")
BACKUP = Path("data/evaluation/dev_annotations_before_batch4.jsonl")

assignments = {
    "AMZ-000192": "Tasya",
    "AMZ-000209": "Shweta",
    "AMZ-000254": "Sinchana",
}

annotations = {
    "AMZ-000192": [
        ("Data transfer", "positive",
         "I transferred my data over to the iPhone 16 Pro Max and it worked well and fairly easy too.", False),
        ("Battery health", "positive",
         "the battery life was at 98 %", False),
        ("Warranty", "positive",
         "it still has an Apple warranty until March of 2026.", False),
        ("Appearance", "positive", "The phone is beautiful", False),
        ("Phone functionality", "positive", "works great thus far", False),
        ("Software support", "positive",
         "my iPhone 16 Pro Max still gets another 6 years of updates on the iOS software.", False),
        ("Price/value", "positive",
         "this is a well made and solid purchase", False),
        ("Seller reliability", "positive",
         "the seller of this iPhone certainly does ship out beautiful and quality iPhones to the buyers.", False),
        ("Cosmetic condition", "positive", "No scratches or dings", False),
        ("Packaging", "positive",
         "The item arrived with great packaging", False),
        ("Battery life", "positive", "a great battery life too", False),
    ],
    "AMZ-000209": [
        ("Phone functionality", "negative",
         "a basically useless phone", True),
        ("Seller quality control", "negative",
         "Although they should have caught this", True),
    ],
    "AMZ-000254": [
        ("Noise cancellation", "positive",
         "The noise cancelling is absolutely next level", False),
        ("Sound quality", "positive",
         "The sound quality is super clear, deep bass, and crisp highs", False),
        ("Comfort", "positive",
         "They feel very light and comfortable, even after hours of use.", False),
        ("Build quality", "positive",
         "The build quality looks premium", False),
        ("Battery life", "positive",
         "battery life easily lasts me a few days of heavy use.", False),
    ],
}

with DATA.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

assert len(rows) == 20, f"Expected 20 records, found {len(rows)}"
assert not BACKUP.exists(), f"Backup already exists: {BACKUP}"

by_id = {r["review_id"]: r for r in rows}
assert len(by_id) == 20, "Duplicate review IDs detected"

for review_id in annotations:
    assert review_id in by_id, f"Missing review: {review_id}"
    assert by_id[review_id]["status"] != "complete", (
        f"{review_id} is already complete; refusing to overwrite"
    )
    assert by_id[review_id].get("annotator") is None, (
        f"{review_id} already has an annotator; refusing to overwrite"
    )

# Verify every evidence span against the original review text.
for review_id, aspects in annotations.items():
    review_text = by_id[review_id]["review_text"]
    for aspect, sentiment, evidence, complaint in aspects:
        assert evidence in review_text, (
            f"Evidence not found: {review_id} / {aspect}: {evidence}"
        )
        assert sentiment in ("positive", "negative", "neutral")

# Make a backup before editing.
shutil.copy2(DATA, BACKUP)

for review_id, aspects in annotations.items():
    record = by_id[review_id]
    record["annotator"] = "Greeshma"
    record["allocated_team_member"] = assignments[review_id]
    record["status"] = "complete"
    record["aspects"] = [
        {
            "aspect_term": aspect,
            "sentiment": sentiment,
            "evidence": evidence,
            "complaint_present": complaint,
        }
        for aspect, sentiment, evidence, complaint in aspects
    ]

# Write updated JSONL.
with DATA.open("w", encoding="utf-8", newline="\n") as f:
    for record in rows:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Reload and verify.
with DATA.open(encoding="utf-8") as f:
    verified = [json.loads(line) for line in f if line.strip()]

assert len(verified) == 20
verified_by_id = {r["review_id"]: r for r in verified}

for review_id, expected_allocation in assignments.items():
    record = verified_by_id[review_id]
    assert record["status"] == "complete"
    assert record["annotator"] == "Greeshma"
    assert record["allocated_team_member"] == expected_allocation
    assert len(record["aspects"]) == len(annotations[review_id])
    for aspect in record["aspects"]:
        assert aspect["evidence"] in record["review_text"]

print("SUCCESS: Batch 4 saved and verified.")
for review_id in annotations:
    record = verified_by_id[review_id]
    print(
        review_id,
        "| annotator:", record["annotator"],
        "| allocated:", record["allocated_team_member"],
        "| aspects:", len(record["aspects"]),
        "| status:", record["status"],
    )

print("Total development records:", len(verified))
print("Completed development records:",
      sum(r["status"] == "complete" for r in verified))
print("Backup:", BACKUP)

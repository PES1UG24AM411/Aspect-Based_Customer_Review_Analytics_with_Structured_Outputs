
import json
import shutil
from pathlib import Path

folder = Path("data/evaluation")
active = folder / "dev_annotations.jsonl"
backup = folder / "dev_annotations_before_batch2.jsonl"

if not active.exists():
    raise FileNotFoundError(active)

if backup.exists():
    raise FileExistsError(
        f"Backup already exists: {backup}. "
        "Do not overwrite it."
    )

# Preserve the current active file.
shutil.copy2(active, backup)

with active.open("r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

by_id = {r["review_id"]: r for r in rows}

expected_ids = {"AMZ-000007", "AMZ-000008", "AMZ-000009"}
if not expected_ids.issubset(by_id):
    raise ValueError("One or more target review IDs are missing.")

def aspect(term, sentiment, evidence, complaint=False):
    return {
        "aspect_term": term,
        "sentiment": sentiment,
        "evidence": evidence,
        "complaint_present": complaint,
    }

annotations = {
    "AMZ-000007": [
        aspect("Screen quality", "positive",
               "Screen quality was flawless."),
        aspect("Overheating", "positive",
               "Zero overheating."),
        aspect("Durability", "positive",
               "Durable enough device"),
        aspect("Price/value", "positive",
               "Worth the money"),
        aspect("Appearance", "positive",
               "Appearance is good"),
        aspect("Touch sensitivity", "positive",
               "Touch sensitivity was same/similar to a new device."),
    ],

    "AMZ-000008": [
        aspect("Warranty", "positive",
               "One-year warranty/return period with Amazon."),
        aspect("Refurbishment transparency", "positive",
               "It came with a checklist and a list of all the refurbished/replacements done to the device in the Service History"),
        aspect("Battery life", "positive",
               "I got 100% battery life since the battery and logic board were replaced."),
        aspect("Software compatibility", "positive",
               "It can run iOS 26"),
        aspect("MagSafe", "positive",
               "MagSafe wireless charging"),
        aspect("StandBy", "positive",
               "StandBy mode to turn it into a desk calendar/clock screen"),
        aspect("Mail features", "positive",
               "Category grouping in Mail"),
        aspect("Face ID", "positive",
               "Face ID unlock is quick and easy to set up and is blazing fast unlocking the iPhone."),
        aspect("Screen condition", "neutral",
               "The 6.1-inch screen and power button did have hairline scratches"),
        aspect("Display quality", "positive",
               "The display still looks great"),
        aspect("Weight and thickness", "neutral",
               "iPhone 13 is heavier and thicker than the iPhone 12."),
        aspect("Camera performance", "negative",
               "It takes a long time to focus, especially for close-up shots, and boke is not reliable.",
               True),
    ],

    "AMZ-000009": [
        aspect("Physical condition", "positive",
               "it looked great out of the box"),
    ],
}

# Update only the three selected records.
for review_id, aspects in annotations.items():
    record = by_id[review_id]

    if record["status"] == "complete":
        raise ValueError(f"{review_id} is already complete. Stopping.")

    for item in aspects:
        if item["evidence"] not in record["review_text"]:
            raise ValueError(
                f"Evidence is not an exact substring: "
                f"{review_id} / {item['aspect_term']}"
            )

    record["aspects"] = aspects
    record["annotator"] = "Greeshma"
    record["status"] = "complete"

# Write updated JSONL.
temp = folder / "dev_annotations_batch2.tmp"
with temp.open("w", encoding="utf-8", newline="\n") as f:
    for record in rows:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

temp.replace(active)

# Reload and verify.
with active.open("r", encoding="utf-8") as f:
    saved = [json.loads(line) for line in f if line.strip()]

assert len(saved) == 20
assert len({r["review_id"] for r in saved}) == 20

saved_by_id = {r["review_id"]: r for r in saved}

for review_id in expected_ids:
    record = saved_by_id[review_id]
    assert record["status"] == "complete"
    assert record["annotator"] == "Greeshma"
    for item in record["aspects"]:
        assert item["evidence"] in record["review_text"]

print("SUCCESS: Batch 2 saved and verified.")
for review_id in sorted(expected_ids):
    record = saved_by_id[review_id]
    print(
        review_id,
        "| annotator:", record["annotator"],
        "| aspects:", len(record["aspects"]),
        "| status:", record["status"],
    )

print("Total development records:", len(saved))
print("Backup:", backup)

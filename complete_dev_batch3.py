
import json
import shutil
from pathlib import Path

folder = Path("data/evaluation")
active = folder / "dev_annotations.jsonl"
backup = folder / "dev_annotations_before_batch3.jsonl"

if not active.exists():
    raise FileNotFoundError(active)

if backup.exists():
    raise FileExistsError(f"Backup already exists: {backup}")

shutil.copy2(active, backup)

with active.open("r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

by_id = {r["review_id"]: r for r in rows}

def aspect(term, sentiment, evidence, complaint=False):
    return {
        "aspect_term": term,
        "sentiment": sentiment,
        "evidence": evidence,
        "complaint_present": complaint,
    }

annotations = {
    "AMZ-000010": [
        aspect("Delivery speed", "positive",
               "it arrived in just two days as expected"),
        aspect("Refurbished condition", "positive",
               "It’s in excellent condition for being refurbished."),
        aspect("Phone functionality", "positive",
               "It works like new."),
        aspect("Battery life", "positive",
               "The battery life is great"),
        aspect("Price/value", "positive",
               "it’s saving me a ton of money."),
    ],

    "AMZ-000011": [
        aspect("Refurbishment quality", "neutral",
               "They appear to be rebuilt or refurbished using a combination of used and newer replacement parts."),
        aspect("Cosmetic condition", "negative",
               "I noticed visible blemishes on the back glass as well as several chips along the casing.",
               True),
        aspect("Phone functionality", "positive",
               "The phone itself seems to work well so far"),
        aspect("Price/value", "positive",
               "considering the price, I’m still satisfied overall."),
        aspect("Condition description accuracy", "negative",
               "describing the cosmetic condition as “excellent” feels misleading.",
               True),
    ],

    "AMZ-000059": [
        aspect("Cosmetic condition", "positive",
               "It felt brand new"),
        aspect("Minor scratches", "neutral",
               "the tiniest scratching around the charger plug part, which would have happened with me using it anyway, so I don't care."),
        aspect("Included accessories", "positive",
               "It came with a charger, which I was very pleased about."),
        aspect("Setup experience", "positive",
               "I had no trouble setting up the phone"),
        aspect("Software cleanliness", "positive",
               "there was nothing extra on the phone"),
        aspect("Design", "positive",
               "I'm so happy it has the non glass back and flat screen again!"),
    ],
}

expected_ids = set(annotations)

if not expected_ids.issubset(by_id):
    raise ValueError("A target review is missing.")

for review_id, aspects in annotations.items():
    record = by_id[review_id]

    if record["status"] == "complete":
        raise ValueError(f"{review_id} is already complete.")

    if record.get("annotator") is not None:
        raise ValueError(
            f"{review_id} has an existing annotator; stopping."
        )

    for item in aspects:
        if item["evidence"] not in record["review_text"]:
            raise ValueError(
                f"Invalid evidence: {review_id} / {item['aspect_term']}"
            )

    record["aspects"] = aspects
    record["annotator"] = "Greeshma"
    record["status"] = "complete"

temp = folder / "dev_annotations_batch3.tmp"

with temp.open("w", encoding="utf-8", newline="\n") as f:
    for record in rows:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

temp.replace(active)

with active.open("r", encoding="utf-8") as f:
    saved = [json.loads(line) for line in f if line.strip()]

assert len(saved) == 20
assert len({r["review_id"] for r in saved}) == 20

saved_by_id = {r["review_id"]: r for r in saved}

for review_id, expected_aspects in annotations.items():
    record = saved_by_id[review_id]
    assert record["status"] == "complete"
    assert record["annotator"] == "Greeshma"
    assert len(record["aspects"]) == len(expected_aspects)

    for item in record["aspects"]:
        assert item["evidence"] in record["review_text"]

print("SUCCESS: Batch 3 saved and verified.")

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

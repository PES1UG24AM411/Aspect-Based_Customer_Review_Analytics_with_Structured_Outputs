import json
from pathlib import Path

path = Path("data/evaluation/dev_annotations.jsonl")

with path.open("r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

todo = [r for r in records if r["status"] != "complete"]

print(f"Unfinished development reviews: {len(todo)}")

for record in todo[:3]:
    print("\n" + "=" * 75)
    print("REVIEW ID:", record["review_id"])
    print("TEXT:", record["review_text"])
    print("CURRENT STATUS:", record["status"])
    print("CURRENT ASPECTS:")
    for aspect in record.get("aspects", []):
        print(json.dumps(aspect, ensure_ascii=False))
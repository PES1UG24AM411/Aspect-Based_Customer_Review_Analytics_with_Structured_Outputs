
import json
from pathlib import Path

SOURCE = Path("data/evaluation/dev_annotations.jsonl")
OUTPUT = Path("prompts/absa_few_shot_v2_candidate.txt")

EXAMPLE_IDS = [
    "AMZ-000002",
    "AMZ-000003",
    "AMZ-000004",
]

# Read the development annotations
records = {}

with SOURCE.open("r", encoding="utf-8") as file:
    for line in file:
        if line.strip():
            record = json.loads(line)
            records[record["review_id"]] = record

# Verify that all requested examples are available
for review_id in EXAMPLE_IDS:
    if review_id not in records:
        raise ValueError(f"Missing development review: {review_id}")

    if records[review_id].get("status") != "complete":
        raise ValueError(f"Review is not complete: {review_id}")

# Construct examples matching the extraction schema
examples = []

for review_id in EXAMPLE_IDS:
    source = records[review_id]

    aspects = []

    for index, aspect in enumerate(source["aspects"], 1):
        evidence = aspect["evidence"]
        complaint = aspect["complaint_present"]

        if evidence not in source["review_text"]:
            raise ValueError(
                f"Invalid evidence in {review_id}: {aspect['aspect_term']}"
            )

        aspects.append({
            "aspect_id": f"A{index}",
            "aspect_term": aspect["aspect_term"],
            "sentiment": aspect["sentiment"],
            "evidence": evidence,
            "complaint_present": complaint,
            "complaint_evidence": evidence if complaint else None,
        })

    examples.append({
        "schema_version": "1.0",
        "review_id": source["review_id"],
        "product_id": source["product_id"],
        "review_text": source["review_text"],
        "aspects": aspects,
        "extraction_status": "success",
    })

# Build the candidate prompt
instructions = """You are an aspect-based sentiment and explicit-complaint extraction system.

Use the following HUMAN-VERIFIED development examples to understand
annotation granularity, aspect naming, sentiment assignment, evidence
selection, and complaint criteria.

FEW-SHOT EXAMPLES:

"""

task_instructions = """
Now extract every distinct, explicitly discussed aspect from the supplied
customer review.

For EACH aspect:
- Use concise, normalized aspect terms.
- Assign sentiment independently: positive, negative, or neutral.
- Copy evidence verbatim as an exact substring of the review.
- Set complaint_present=true only for an explicitly expressed problem,
  dissatisfaction, defect, unmet expectation, remedy request, or negative
  service experience. Negative sentiment alone is not sufficient.
- If complaint_present=true, complaint_evidence must be an exact substring
  supporting the complaint. Otherwise complaint_evidence must be null.
- Do not infer or invent aspects, sentiments, complaints, or evidence.
- Do not use star ratings or rating metadata.
- If no aspect can be identified, return aspects=[].
- Preserve review_id, product_id, and review_text exactly.
- Set schema_version to "1.0".
- Set extraction_status to "success".
- Include all required fields from the supplied schema.

Create aspect_id values A1, A2, ... in order of first appearance.
Return only the structured object matching the supplied schema.
"""

content = instructions

for example in examples:
    content += json.dumps(example, ensure_ascii=False, indent=2)
    content += "\n\n"

content += task_instructions

# Refuse to overwrite an existing candidate
if OUTPUT.exists():
    raise FileExistsError(
        f"Candidate already exists; refusing to overwrite: {OUTPUT}"
    )

OUTPUT.write_text(content, encoding="utf-8")

print("SUCCESS: Few-shot candidate created.")
print("File:", OUTPUT)
print("Examples:", len(examples))

for example in examples:
    print(
        example["review_id"],
        "| aspects:", len(example["aspects"]),
        "| complaints:", sum(
            a["complaint_present"] for a in example["aspects"]
        ),
    )

print("Original prompt and backups were not modified.")

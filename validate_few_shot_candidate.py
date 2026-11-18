
import json
from pathlib import Path
from src.extraction.schemas import ExtractionRecord

path = Path("prompts/absa_few_shot_v2_candidate.txt")
content = path.read_text(encoding="utf-8")

start_marker = "FEW-SHOT EXAMPLES:"
end_marker = "\nNow extract every distinct"

if start_marker not in content or end_marker not in content:
    raise ValueError("Could not locate the example section")

section = content.split(start_marker, 1)[1].split(end_marker, 1)[0]

decoder = json.JSONDecoder()
examples = []
position = 0

while position < len(section):
    while position < len(section) and section[position].isspace():
        position += 1

    if position >= len(section):
        break

    example, position = decoder.raw_decode(section, position)
    validated = ExtractionRecord.model_validate(example)
    examples.append(validated)

expected_ids = [
    "AMZ-000002",
    "AMZ-000003",
    "AMZ-000004",
]

actual_ids = [example.review_id for example in examples]

if actual_ids != expected_ids:
    raise ValueError(f"Unexpected example IDs: {actual_ids}")

for example in examples:
    for aspect in example.aspects:
        if aspect.evidence not in example.review_text:
            raise ValueError(
                f"Invalid evidence: {example.review_id}, {aspect.aspect_id}"
            )

        if aspect.complaint_present:
            if aspect.complaint_evidence not in example.review_text:
                raise ValueError(
                    f"Invalid complaint evidence: "
                    f"{example.review_id}, {aspect.aspect_id}"
                )

print("Few-shot candidate schema validation")
print("-------------------------------------")
print("Examples validated:", len(examples))

for example in examples:
    print(
        example.review_id,
        "| aspects:", len(example.aspects),
        "| schema: valid"
    )

print("All evidence spans verified.")
print("VALIDATION PASSED")

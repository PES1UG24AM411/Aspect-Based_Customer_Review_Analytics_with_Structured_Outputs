import json
from pathlib import Path

def read_jsonl(path):
    path=Path(path)
    if not path.exists(): return []
    rows=[]
    for line_number,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def extract_json_object(raw_text):
    """Extract and parse one JSON object from model output.

    Handles Markdown code fences and surrounding explanatory text.
    Does not guess or repair invalid JSON syntax.
    """
    import re

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Model output is empty.")

    text = raw_text.strip()

    # Prefer the contents of a Markdown JSON code fence, if present.
    fenced = re.search(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        text = fenced.group(1).strip()

    # Find the first complete JSON object, respecting quoted braces.
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:index + 1]
                try:
                    result = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSON object is malformed: {exc}"
                    ) from exc

                if not isinstance(result, dict):
                    raise ValueError("Expected a JSON object.")
                return result

    raise ValueError("JSON object is incomplete or has unbalanced braces.")

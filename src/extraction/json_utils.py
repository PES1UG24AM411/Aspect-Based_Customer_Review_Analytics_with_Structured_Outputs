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

"""Versioned prompt strategies for offline inspection and controlled evaluation."""
from pathlib import Path

from .config import PROJECT_ROOT

PROMPT_FILES = {
    "zero_shot": "absa_zero_shot_v1.txt",
    "few_shot": "absa_few_shot_v1.txt",
    "few_shot_v2": "absa_few_shot_v2_candidate.txt",
    "cot": "absa_cot_v1.txt",
}

def load_prompt(strategy="zero_shot"):
    if strategy not in PROMPT_FILES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose from: {', '.join(PROMPT_FILES)}"
        )

    path = PROJECT_ROOT / "prompts" / PROMPT_FILES[strategy]
    if not path.is_file():
        raise FileNotFoundError(path)

    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt is empty: {path}")

    if strategy == "few_shot" and "[TODO:" in prompt:
        raise ValueError(
            "Few-shot prompt is not ready: add human-verified development examples first."
        )

    return prompt

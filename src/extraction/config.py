import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")
PROMPT_VERSION = "absa_extraction_v1"
SCHEMA_VERSION = "1.0"
API_KEY_PRESENT = bool(os.getenv("OPENAI_API_KEY"))

# Current published API pricing for gpt-6-luna; revise if pricing changes.
MODEL_INPUT_USD_PER_MILLION = 0.10
MODEL_OUTPUT_USD_PER_MILLION = 0.50

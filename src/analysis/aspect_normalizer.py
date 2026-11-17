"""Map free-text aspect terms from the LLM to canonical aspect names.

The extraction prompt asks for concise terms but does not constrain a vocabulary, so
"battery", "Battery life" and "battery drain" must be unified before aggregation or
evaluation. Unknown terms are kept (lower-cased, singularised) rather than dropped.
Extend CANONICAL_RULES when the pilot/full output reveals new frequent terms.
"""
import re

CANONICAL_RULES = [
    ("battery life", r"batter|charg|power (?:life|drain)|\bdrain"),
    ("camera", r"camera|photo|picture|video quality|lens"),
    ("screen", r"screen|display|brightness|resolution"),
    ("price/value", r"price|value|cost|expens|cheap|worth|afford|money"),
    ("shipping/delivery", r"ship|deliver|arriv"),
    ("performance", r"perform|speed|\bfast|\bslow|lag|processor|\bram\b|responsiv"),
    ("packaging", r"packag|box"),
    ("customer service", r"customer (?:service|support)|\bsupport\b|seller|warranty|\bservice\b"),
    ("returns/refunds", r"return|refund|replac"),
    ("sound quality", r"sound|audio|speaker|volume|bass|microphone|\bmic\b"),
    ("build quality", r"build|durab|material|construct|sturd|flimsy|\bquality\b"),
    ("design", r"design|look|appearance|style|color|colour|size|weight|\bfit\b|comfort"),
    ("ease of use", r"ease|easy|use(?:r)?[- ]friendly|usab|setup|set up|install|interface|intuitive"),
    ("compatibility", r"compatib|connect|bluetooth|wi-?fi|pair"),
    ("software", r"software|\bapp\b|update|\bios\b|firmware|\bui\b"),
    ("storage", r"storage|memory|capacity"),
]


def normalize_aspect(term: str) -> str:
    t = re.sub(r"[^a-z0-9/&' -]+", " ", str(term).lower()).strip()
    t = re.sub(r"\s+", " ", t)
    for canonical, pattern in CANONICAL_RULES:
        if re.search(pattern, t):
            return canonical
    t = re.sub(r"(?<=[a-z])ies$", "y", t) if len(t) > 4 else t
    t = re.sub(r"(?<![s])s$", "", t) if len(t) > 3 else t
    return t or "unknown"

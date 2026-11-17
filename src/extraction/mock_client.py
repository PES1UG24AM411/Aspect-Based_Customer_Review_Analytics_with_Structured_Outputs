"""Keyword-rule stand-in for the LLM client (TESTING / UI DEMO ONLY).

It mimics ``client.responses.parse`` so the real pipeline, validators, retry and
resume logic can be exercised offline. Its output is NOT model output and must
never be reported as experimental results.
"""
import re
from types import SimpleNamespace

from .schemas import Aspect, ExtractionRecord

MOCK_MODEL_LABEL = "mock_keyword_extractor"

ASPECT_PATTERNS = {
    "Battery life": r"\bbatter(?:y|ies)\b|\bcharg(?:e|es|ing|er)\b",
    "Camera": r"\bcamera|\bphotos?\b|\bpictures?\b",
    "Screen": r"\bscreen\b|\bdisplay\b",
    "Performance": r"\bperformance\b|\bfast\b|\bslow\b|\blag",
    "Price/value": r"\bprice\b|\bvalue\b|\bexpensive\b|\bcheap\b|\bworth\b",
    "Shipping/delivery": r"\bshipping\b|\bdelivery\b|\barrived\b|\bdelivered\b",
    "Packaging": r"\bpackag",
    "Customer service": r"\bcustomer (?:service|support)\b|\bseller\b|\brefund\b|\breturn",
    "Sound quality": r"\bsound\b|\baudio\b|\bspeaker",
    "Build quality": r"\bbuild\b|\bdurab|\bquality\b|\bbroke\b|\bflimsy\b",
}
NEG = r"\bbad\b|\bworst\b|\bpoor\b|\bterrible\b|\bdisappoint|\bbroke|\bdrain|\bslow\b|\bnever\b|\bissue|\bproblem|\blate\b|\bdefect|\bwaste|\bflimsy|\bcan't\b|\bdoesn't\b|\bdidn't\b|\bnot\b|\bstopped\b"
POS = r"\bgreat\b|\bexcellent\b|\blove|\bgood\b|\bamazing\b|\bperfect\b|\bfast\b|\bbest\b|\bhappy\b|\bwonderful\b|\bimpressed\b|\bsmooth\b"


def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+|,\s*(?=(?:but|though|however|and)\b)|;\s*", text)
    return [x.strip() for x in parts if x and x.strip() and x.strip() in text]


def mock_extract(review_id, product_id, review_text):
    aspects = []
    for sentence in _sentences(review_text):
        for term, pat in ASPECT_PATTERNS.items():
            if any(a.aspect_term == term for a in aspects) or not re.search(pat, sentence, re.I):
                continue
            neg, pos = len(re.findall(NEG, sentence, re.I)), len(re.findall(POS, sentence, re.I))
            sentiment = "negative" if neg > pos else "positive" if pos > neg else "neutral"
            complaint = sentiment == "negative"
            aspects.append(Aspect(aspect_id=f"A{len(aspects)+1}", aspect_term=term, sentiment=sentiment,
                                  evidence=sentence, complaint_present=complaint,
                                  complaint_evidence=sentence if complaint else None))
    return ExtractionRecord(schema_version="1.0", review_id=review_id, product_id=product_id,
                            review_text=review_text, aspects=aspects, extraction_status="success")


class _Responses:
    def __init__(self):
        self.calls = 0

    def parse(self, *, model=None, input="", text_format=None, **_):
        self.calls += 1
        rid = re.search(r"^review_id: (.*)$", input, re.M).group(1)
        pid = re.search(r"^product_id: (.*)$", input, re.M).group(1)
        text = input.split("review_text: ", 1)[1].split("\n\nCorrect these validation issues:")[0]
        return SimpleNamespace(output_parsed=mock_extract(rid, pid, text),
                               usage=SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0))


class MockClient:
    def __init__(self):
        self.responses = _Responses()

"""Semantic checks beyond structural Pydantic validation."""
from .schemas import ExtractionRecord

def validate_record(record: ExtractionRecord, *, expected_review_id: str,
                    expected_product_id: str, expected_review_text: str) -> ExtractionRecord:
    if record.review_id != expected_review_id:
        raise ValueError("review_id changed from source")
    if record.product_id != expected_product_id:
        raise ValueError("product_id changed from source")
    if record.review_text != expected_review_text:
        raise ValueError("review_text changed from source")
    for aspect in record.aspects:
        if aspect.evidence not in expected_review_text:
            raise ValueError(f"Evidence is not an exact substring of review: {aspect.aspect_id}")
        if aspect.complaint_present:
            if aspect.complaint_evidence not in expected_review_text:
                raise ValueError(f"Complaint evidence is not an exact substring of review: {aspect.aspect_id}")
    return record

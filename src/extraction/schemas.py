"""Validated output models for Member 2 ABSA extraction."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Sentiment = Literal["positive", "negative", "neutral"]

class Aspect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    aspect_id: str = Field(min_length=1)
    aspect_term: str = Field(min_length=1)
    sentiment: Sentiment
    evidence: str = Field(min_length=1)
    complaint_present: bool
    complaint_evidence: str | None

    @model_validator(mode="after")
    def complaint_evidence_consistent(self):
        if self.complaint_present and not self.complaint_evidence:
            raise ValueError("complaint_evidence is required when complaint_present is true")
        if not self.complaint_present and self.complaint_evidence is not None:
            raise ValueError("complaint_evidence must be null when complaint_present is false")
        return self

class ExtractionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"]
    review_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    review_text: str = Field(min_length=1)
    aspects: list[Aspect]
    extraction_status: Literal["success"]

    @model_validator(mode="after")
    def unique_aspect_ids(self):
        ids = [a.aspect_id for a in self.aspects]
        if len(ids) != len(set(ids)):
            raise ValueError("aspect_id values must be unique within a review")
        return self

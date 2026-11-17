"""Gold-standard (human annotation) schema."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Sentiment = Literal["positive", "negative", "neutral"]


class GoldAspect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    aspect_term: str = Field(min_length=1)
    sentiment: Sentiment
    evidence: str = Field(min_length=1)
    complaint_present: bool


class GoldRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    review_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    review_text: str = Field(min_length=1)
    annotator: str | None = None
    status: Literal["todo", "complete"] = "todo"
    aspects: list[GoldAspect] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def complete_needs_annotator_and_evidence(self):
        if self.status == "complete":
            if not self.annotator:
                raise ValueError("completed annotation requires annotator")
            for a in self.aspects:
                if a.evidence not in self.review_text:
                    raise ValueError(f"gold evidence not an exact substring: {a.aspect_term!r}")
        return self

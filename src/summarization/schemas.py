from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class ProductSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: str
    overview: str
    strengths: list[Claim]
    weaknesses: list[Claim]


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    verdict: Literal["supported", "partially_supported", "unsupported"]
    reason: str


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: list[ClaimVerdict]

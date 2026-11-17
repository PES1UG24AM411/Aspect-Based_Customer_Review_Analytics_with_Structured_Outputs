"""Support-checking of summaries: deterministic structural check + optional LLM judge.

The LLM judge flags *unsupported claims relative to the cited evidence*. It does not measure
extraction accuracy and must not be reported as ground truth.
"""
from src.extraction.config import MODEL, PROJECT_ROOT
from .schemas import JudgeResult, ProductSummary

JUDGE_PROMPT = PROJECT_ROOT / "prompts" / "summary_judge_v1.txt"


def structural_check(summary: ProductSummary, evidence):
    """Deterministic: every claim cites existing evidence ids whose sentiment polarity fits the claim list."""
    by_id = {e["id"]: e for e in evidence}
    issues = []
    for kind, claims, want in (("strength", summary.strengths, "positive"), ("weakness", summary.weaknesses, "negative")):
        for c in claims:
            missing = [i for i in c.evidence_ids if i not in by_id]
            if missing:
                issues.append({"claim": c.claim, "kind": kind, "issue": f"cites unknown evidence ids {missing}"})
                continue
            if not any(by_id[i]["sentiment"] == want for i in c.evidence_ids):
                issues.append({"claim": c.claim, "kind": kind, "issue": f"no cited evidence has {want} sentiment"})
    return {"n_claims": len(summary.strengths) + len(summary.weaknesses), "issues": issues, "passed": not issues}


def llm_judge(summary: ProductSummary, evidence, *, client=None, model=None):
    from openai import OpenAI
    client = client or OpenAI(timeout=60.0, max_retries=0)
    by_id = {e["id"]: e for e in evidence}
    parts = []
    for kind, claims in (("STRENGTH", summary.strengths), ("WEAKNESS", summary.weaknesses)):
        for c in claims:
            cited = "; ".join(f'{i}: "{by_id[i]["quote"]}"' for i in c.evidence_ids if i in by_id)
            parts.append(f"{kind}: {c.claim}\n  cited evidence -> {cited or '(none valid)'}")
    if not parts:
        return JudgeResult(verdicts=[])
    resp = client.responses.parse(model=model or MODEL, text_format=JudgeResult,
                                  input=f"{JUDGE_PROMPT.read_text(encoding='utf-8')}\n\nCLAIMS:\n" + "\n".join(parts))
    if resp.output_parsed is None:
        raise ValueError("No structured judge result returned")
    return resp.output_parsed

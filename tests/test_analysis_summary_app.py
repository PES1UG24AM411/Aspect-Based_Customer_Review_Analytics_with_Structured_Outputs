import json
import pandas as pd
from src.analysis.aggregate import aggregate, load_records, aspect_frame
from src.extraction.mock_client import mock_extract
from src.summarization.summarizer import extractive_summary, build_evidence
from src.summarization.judge import structural_check
from src.summarization.schemas import Claim, ProductSummary

REV = [("R1", "P1", "The battery is great. Shipping was late and slow."),
       ("R2", "P1", "Battery is terrible, it drains fast. Screen is great."),
       ("R3", "P1", "Battery is bad, always dies. Love the screen."),
       ("R4", "P2", "The camera is excellent."),
       ("R5", "P2", "Nothing to say here.")]


def recs():
    return [mock_extract(*r) for r in REV]


def test_aggregation_counts_and_no_double_counting(tmp_path):
    rs = recs()
    # same record present twice (pilot + full) must be counted once
    p = tmp_path / "a.jsonl"; q = tmp_path / "b.jsonl"
    for f in (p, q):
        f.write_text("\n".join(r.model_dump_json() for r in rs) + "\n")
    loaded, skipped = load_records([p, q])
    assert len(loaded) == 5 and skipped == 0
    t = aggregate(loaded)
    assert t["product_overview"].set_index("product_id").loc["P1", "n_reviews"] == 3
    sd = t["sentiment_distribution"]
    b = sd[(sd.product_id == "P1") & (sd.aspect == "battery life")].iloc[0]
    assert b.total == 3 and b.negative == 2 and b.positive == 1
    assert not t["recurring_negative"].empty
    assert (t["recurring_negative"].aspect == "battery life").any()
    assert (t["aspect_comparison"].shape[1]) == 2


def test_duplicate_aspect_in_one_review_counts_once():
    r = mock_extract("R9", "P9", "Battery is good. The charger is bad.")
    af = aspect_frame([r])
    t = aggregate([r])
    assert t["aspect_frequency"].n_reviews.max() == 1


def test_invalid_lines_skipped(tmp_path):
    p = tmp_path / "x.jsonl"; p.write_text('{"review_id": "R1"}\n' + recs()[0].model_dump_json() + "\n")
    loaded, skipped = load_records([p])
    assert len(loaded) == 1 and skipped == 1


def test_empty_aggregate():
    t = aggregate([])
    assert t["aspect_frequency"].empty


def test_extractive_summary_cites_real_evidence_and_passes_check():
    af = aspect_frame(recs())
    s = extractive_summary(af, "P1")
    ev = build_evidence(af, "P1")
    assert s.weaknesses and s.strengths
    chk = structural_check(s, ev)
    assert chk["passed"], chk


def test_structural_check_flags_bad_citations():
    af = aspect_frame(recs()); ev = build_evidence(af, "P1")
    bad = ProductSummary(product_id="P1", overview="x", strengths=[Claim(claim="great", evidence_ids=["E999"])],
                         weaknesses=[Claim(claim="bad", evidence_ids=[ev[0]["id"]] if ev[0]["sentiment"] != "negative" else ["E999"])])
    assert not structural_check(bad, ev)["passed"]


def test_judge_with_fake_client():
    from types import SimpleNamespace
    from src.summarization.judge import llm_judge
    from src.summarization.schemas import JudgeResult, ClaimVerdict
    af = aspect_frame(recs()); ev = build_evidence(af, "P1"); s = extractive_summary(af, "P1")
    out = JudgeResult(verdicts=[ClaimVerdict(claim=c.claim, verdict="supported", reason="ok") for c in s.strengths + s.weaknesses])
    fake = SimpleNamespace(responses=SimpleNamespace(parse=lambda **k: SimpleNamespace(output_parsed=out)))
    assert len(llm_judge(s, ev, client=fake).verdicts) == len(s.strengths + s.weaknesses)


def test_summary_cli_requires_key_for_llm(monkeypatch):
    import pytest
    from src.summarization import run_summaries
    monkeypatch.setattr(run_summaries, "API_KEY_PRESENT", False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        run_summaries.main(["--llm"])

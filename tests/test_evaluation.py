import json
import pytest
from src.analysis.aspect_normalizer import normalize_aspect
from src.evaluation import metrics as M
from src.evaluation.run_eval import load_gold, write_report
from src.evaluation.sampling import make_splits, write_splits
import pandas as pd

TEXT = "The camera is excellent, but the battery drains too quickly. Delivery was late."


def asp(term, sent, ev, cmp=False):
    return {"aspect_term": term, "sentiment": sent, "evidence": ev, "complaint_present": cmp}


GOLD = {"R1": {"review_text": TEXT, "aspects": [
    asp("camera", "positive", "The camera is excellent"),
    asp("battery life", "negative", "the battery drains too quickly", True),
    asp("delivery", "negative", "Delivery was late", True)]}}


def test_normalizer_unifies_variants():
    assert normalize_aspect("Battery") == normalize_aspect("battery drain") == "battery life"
    assert normalize_aspect("Shipping speed") == "shipping/delivery"
    assert normalize_aspect("Zipper") == "zipper"


def test_prf_known_values():
    r = M.prf(8, 2, 2)
    assert r["precision"] == 0.8 and r["recall"] == 0.8 and r["f1"] == 0.8
    assert M.prf(0, 0, 0)["f1"] == 0.0


def test_perfect_prediction():
    res, errs = M.evaluate(GOLD, {"R1": {"aspects": [dict(a) for a in GOLD["R1"]["aspects"]]}})
    assert res["aspect_extraction"]["f1"] == 1.0 and not errs
    assert res["sentiment_on_matched_aspects"]["accuracy"] == 1.0
    assert res["complaint_on_matched_aspects"]["f1"] == 1.0


def test_errors_are_counted_by_hand():
    pred = {"R1": {"aspects": [
        asp("Camera quality", "negative", "The camera is excellent"),            # sentiment error
        asp("Battery", "negative", "the battery drains too quickly", False),     # complaint error
        asp("Screen", "positive", "The camera is excellent, but")]}}              # FP (delivery missed -> FN)
    res, errs = M.evaluate(GOLD, pred)
    a = res["aspect_extraction"]
    assert (a["tp"], a["fp"], a["fn"]) == (2, 1, 1)
    assert res["error_counts"] == {"sentiment_error": 1, "complaint_error": 1, "false_positive_aspect": 1, "missed_aspect": 1}
    s = res["sentiment_on_matched_aspects"]
    assert s["accuracy"] == 0.5 and s["confusion_matrix"]["positive"]["negative"] == 1
    assert res["complaint_on_matched_aspects"]["fn"] == 1


def test_matching_is_one_to_one():
    pred = {"R1": {"aspects": [asp("camera", "positive", "The camera is excellent"),
                               asp("camera", "positive", "The camera is excellent")]}}
    res, _ = M.evaluate(GOLD, pred)
    assert res["aspect_extraction"]["tp"] == 1 and res["aspect_extraction"]["fp"] == 1


def test_only_overlapping_reviews_scored_and_coverage_reported():
    res, _ = M.evaluate({**GOLD, "R2": {"review_text": "x", "aspects": []}}, {"R1": {"aspects": []}})
    assert res["reviews_scored"] == 1 and res["coverage"]["missing_predictions"] == ["R2"]


def test_kappa_and_weighted_macro():
    t = ["positive", "positive", "negative", "neutral"]; p = ["positive", "negative", "negative", "neutral"]
    r = M.classification_report(t, p, M.LABELS)
    assert r["accuracy"] == 0.75 and r["per_class"]["positive"]["support"] == 2
    assert 0 < r["cohen_kappa"] < 1
    assert M.cohen_kappa(t, t, M.LABELS) == 1.0


def test_load_gold_refuses_missing_and_incomplete(tmp_path):
    with pytest.raises(SystemExit, match="No valid reference"):
        load_gold(tmp_path / "none.jsonl")
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"review_id": "R1", "product_id": "P", "review_text": TEXT, "status": "todo", "aspects": []}) + "\n")
    with pytest.raises(SystemExit, match="no completed"):
        load_gold(p)


def test_load_gold_detects_dev_leakage(tmp_path):
    g = tmp_path / "g.jsonl"
    g.write_text(json.dumps({"review_id": "R1", "product_id": "P", "review_text": TEXT, "status": "complete",
                             "annotator": "a", "aspects": []}) + "\n")
    s = tmp_path / "splits.json"; s.write_text(json.dumps({"dev_ids": ["R1"], "eval_ids": []}))
    with pytest.raises(SystemExit, match="leakage"):
        load_gold(g, s)


def test_gold_evidence_must_be_substring():
    from src.evaluation.schemas import GoldRecord
    with pytest.raises(ValueError, match="substring"):
        GoldRecord(review_id="R", product_id="P", review_text=TEXT, status="complete", annotator="a",
                   aspects=[asp("x", "positive", "not in text")])


def test_splits_deterministic_disjoint_and_exclude_pilot():
    df = pd.DataFrame({"review_id": [f"R{i:04d}" for i in range(400)], "product_id": "P",
                       "review_text": "some review text", "rating": [1 + i % 5 for i in range(400)]})
    pilot = [f"R{i:04d}" for i in range(10)]
    a = make_splits(df, pilot_ids=pilot); b = make_splits(df, pilot_ids=pilot)
    assert a == b
    dev, ev = a
    assert not set(dev) & set(ev) and set(pilot) <= set(dev) and len(ev) == 100


def test_report_files_written(tmp_path):
    res, errs = M.evaluate(GOLD, {"R1": {"aspects": [asp("camera", "negative", "The camera is excellent")]}})
    write_report(res, errs, tmp_path)
    for f in ("evaluation_metrics.json", "evaluation_report.md", "sentiment_confusion.png", "complaint_confusion.png", "error_analysis.csv"):
        assert (tmp_path / f).is_file()

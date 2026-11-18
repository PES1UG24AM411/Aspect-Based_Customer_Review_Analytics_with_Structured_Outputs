import json
import pytest
from src.analysis.aspect_normalizer import normalize_aspect
from src.evaluation import metrics as M
from src.evaluation.run_eval import (
    compare_prompt_strategies,
    load_gold,
    load_strategy_predictions,
    write_report,
)
from src.evaluation.sampling import (
    make_splits,
    select_human_evaluation_ids,
    write_splits,
)
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


def test_human_sample_uses_existing_shared_assignment_deterministically():
    splits = {
        "dev_ids": ["D1"],
        "eval_ids": [f"R{i}" for i in range(20)],
    }
    assignment = {
        "shared_review_ids": [f"R{i}" for i in range(18)],
    }

    selected = select_human_evaluation_ids(splits, assignment)

    assert selected == [f"R{i}" for i in range(15)]
    assert selected == select_human_evaluation_ids(splits, assignment)
    assert not set(selected).intersection(splits["dev_ids"])


def test_strategy_comparison_scores_same_fifteen_human_gold_reviews(tmp_path):
    review_ids = [f"R{i:02}" for i in range(15)]
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"dev_ids": ["D1"], "eval_ids": review_ids}))
    assignment = tmp_path / "assignment.json"
    assignment.write_text(json.dumps({"shared_review_ids": review_ids}))
    gold_path = tmp_path / "gold.jsonl"
    gold_rows = [
        {
            "review_id": review_id,
            "product_id": "P1",
            "review_text": "The camera is excellent.",
            "annotator": "human",
            "status": "complete",
            "aspects": [asp("camera", "positive", "The camera is excellent")],
        }
        for review_id in review_ids
    ]
    gold_path.write_text("\n".join(json.dumps(row) for row in gold_rows) + "\n")
    dataset_path = tmp_path / "dataset.csv"
    pd.DataFrame([
        {
            "review_id": review_id,
            "product_id": "P1",
            "review_text": "The camera is excellent.",
            "language": "English",
        }
        for review_id in review_ids
    ]).to_csv(dataset_path, index=False)

    prediction_paths = {}
    for strategy in ("zero_shot", "few_shot", "cot"):
        path = tmp_path / f"{strategy}.jsonl"
        rows = [
            {
                "schema_version": "1.0",
                "review_id": review_id,
                "product_id": "P1",
                "review_text": "The camera is excellent.",
                "aspects": [{
                    "aspect_id": "A1",
                    "aspect_term": "camera",
                    "sentiment": "positive",
                    "evidence": "The camera is excellent",
                    "complaint_present": False,
                    "complaint_evidence": None,
                }],
                "extraction_status": "success",
            }
            for review_id in review_ids
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        prediction_paths[strategy] = path

    report = compare_prompt_strategies(
        gold_path=gold_path,
        splits_path=splits,
        assignment_path=assignment,
        dataset_path=dataset_path,
        prediction_paths=prediction_paths,
    )

    assert report["evaluation_sample"]["target_reviews"] == 15
    assert report["evaluation_sample"]["review_ids"] == review_ids
    for strategy in ("zero_shot", "few_shot", "cot"):
        entry = report["strategies"][strategy]
        result = entry["metrics"]
        assert result["reviews_scored"] == 15
        assert entry["macro_aspect_metrics"]["precision"] == 1.0
        assert entry["macro_aspect_metrics"]["recall"] == 1.0
        assert entry["macro_aspect_metrics"]["f1"] == 1.0


def test_strategy_predictions_report_invalid_and_missing_as_reviews(tmp_path):
    gold = {
        "R1": {"review_id": "R1", "product_id": "P1", "review_text": TEXT, "aspects": []},
        "R2": {"review_id": "R2", "product_id": "P2", "review_text": TEXT, "aspects": []},
    }
    path = tmp_path / "predictions.jsonl"
    invalid = {
        "schema_version": "1.0",
        "review_id": "R1",
        "product_id": "WRONG",
        "review_text": TEXT,
        "aspects": [],
        "extraction_status": "success",
    }
    outside = {
        "schema_version": "1.0",
        "review_id": "OTHER",
        "product_id": "P1",
        "review_text": TEXT,
        "aspects": [],
        "extraction_status": "success",
    }
    path.write_text(json.dumps(invalid) + "\n" + json.dumps(outside) + "\n")

    predictions, diagnostics = load_strategy_predictions(path, gold)

    assert set(predictions) == {"R1", "R2"}
    assert predictions["R1"]["aspects"] == []
    assert predictions["R2"]["aspects"] == []
    assert diagnostics["missing_or_invalid_review_ids"] == ["R1", "R2"]
    assert diagnostics["invalid_record_count"] == 1
    assert diagnostics["ignored_out_of_sample_records"] == 1


def test_strategy_comparison_rejects_incomplete_human_gold(tmp_path):
    review_ids = [f"R{i:02}" for i in range(15)]
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"dev_ids": [], "eval_ids": review_ids}))
    assignment = tmp_path / "assignment.json"
    assignment.write_text(json.dumps({"shared_review_ids": review_ids}))
    gold_path = tmp_path / "gold.jsonl"
    row = {
        "review_id": review_ids[0],
        "product_id": "P1",
        "review_text": TEXT,
        "status": "complete",
        "annotator": "human",
        "aspects": [],
    }
    gold_path.write_text(json.dumps(row) + "\n")
    dataset_path = tmp_path / "dataset.csv"
    pd.DataFrame([
        {
            "review_id": review_id,
            "product_id": "P1",
            "review_text": TEXT,
            "language": "English",
        }
        for review_id in review_ids
    ]).to_csv(dataset_path, index=False)

    with pytest.raises(SystemExit, match="Human gold is incomplete"):
        compare_prompt_strategies(
            gold_path=gold_path,
            splits_path=splits,
            assignment_path=assignment,
            dataset_path=dataset_path,
            prediction_paths={"zero_shot": "x", "few_shot": "y", "cot": "z"},
        )


def test_report_files_written(tmp_path):
    res, errs = M.evaluate(GOLD, {"R1": {"aspects": [asp("camera", "negative", "The camera is excellent")]}})
    write_report(res, errs, tmp_path)
    for f in ("evaluation_metrics.json", "evaluation_report.md", "sentiment_confusion.png", "complaint_confusion.png", "error_analysis.csv"):
        assert (tmp_path / f).is_file()

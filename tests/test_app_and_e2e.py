import json
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from src.extraction import pipeline
from src.extraction.approval import approve_pilot
from src.extraction.mock_client import MockClient
from src.analysis.aggregate import main as agg_main
from src.summarization import run_summaries
from src.evaluation import run_eval

from pathlib import Path
APP = str(Path(__file__).resolve().parents[1] / "app/streamlit_app.py")
PAGES = ["Overview", "Review explorer", "Aspects & sentiment", "Complaints", "Products", "Summaries", "Evaluation", "Pipeline status"]


@pytest.mark.parametrize("mock", [False, True])
def test_every_page_renders_without_exception(mock):
    at = AppTest.from_file(APP, default_timeout=90).run()
    if mock:
        at.sidebar.checkbox[0].check().run()
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, at.exception[0].value if at.exception else "")


def test_real_predictions_take_priority_over_mock(tmp_path, monkeypatch):
    from app import data_access

    real_records = [{"review_id": "REAL-1"}]

    monkeypatch.setattr(
        data_access,
        "load_records",
        lambda paths: (real_records, None)
    )

    reviews = pd.DataFrame({
        "language": ["English"],
        "review_id": ["MOCK-1"],
        "product_id": ["P1"],
        "review_text": ["Sample review"]
    })

    records, source = data_access.load_extraction(
        root=tmp_path,
        use_mock=True,
        reviews=reviews
    )

    assert records == real_records
    assert source == "model predictions"


def test_evaluation_ui_shows_fifteen_review_target_and_independent_progress():
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.radio[0].set_value("Evaluation").run()
    at.radio[0].set_value("Evaluation (15-review target)").run()

    assert not at.exception
    assert any(
        metric.label == "Human evaluation target" and metric.value == "15 unique reviews"
        for metric in at.metric
    )
    assert any("0 / 15 selected reviews completed" in caption.value for caption in at.caption)
    assert any("Review 1 of 15" in item.value for item in at.markdown)

    at.selectbox[0].set_value("Annotator 2").run()
    assert not at.exception
    assert any("annotator_2.jsonl" in caption.value for caption in at.caption)


def test_end_to_end_workflow_offline(tmp_path):
    """preprocess output -> pilot -> approve -> full -> aggregate -> summaries -> evaluation (mock client, tmp dirs)."""
    from src.extraction.config import PROJECT_ROOT
    df = pd.read_csv(PROJECT_ROOT / "data/processed/amazon_reviews_cleaned.csv").head(40)
    ds = tmp_path / "clean.csv"; df.to_csv(ds, index=False)
    env = dict(dataset=ds, out_dir=tmp_path / "ext", reports_dir=tmp_path / "rep", client=MockClient(), require_key=False)
    pipeline.run(**env)
    approve_pilot(reviewer="t", reports_dir=env["reports_dir"], out_dir=env["out_dir"], dataset=ds)
    m = pipeline.run(full=True, **env)
    eng = pipeline.english_eligible(ds)
    assert m["complete"] and m["completed_total"] == len(eng)
    preds = [str(tmp_path / "ext/predictions.jsonl"), str(tmp_path / "ext/pilot_predictions.jsonl")]
    agg_main(["--predictions", *preds, "--reviews", str(ds), "--out", str(tmp_path / "an")])
    assert (tmp_path / "an/product_overview.csv").is_file()
    run_summaries.main(["--predictions", *preds, "--out", str(tmp_path / "an/s.jsonl"), "--min-reviews", "1"])
    rows = [json.loads(l) for l in (tmp_path / "an/s.jsonl").read_text().splitlines()]
    assert rows and all(r["structural_check"]["passed"] for r in rows)
    # evaluation: gold built independently of predictions (hand-written fixture for the test only)
    first = eng.iloc[0]
    gold = {"review_id": str(first.review_id), "product_id": str(first.product_id), "review_text": str(first.review_text),
            "status": "complete", "annotator": "test", "aspects": []}
    g = tmp_path / "gold.jsonl"; g.write_text(json.dumps(gold) + "\n")
    run_eval.main(["--gold", str(g), "--splits", str(tmp_path / "nosplits.json"), "--predictions", *preds, "--out", str(tmp_path / "ev")])
    assert (tmp_path / "ev/evaluation_report.md").is_file()

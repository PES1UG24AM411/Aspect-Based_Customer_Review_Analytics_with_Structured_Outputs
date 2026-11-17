import json
import pandas as pd
import pytest
from src.extraction import pipeline
from src.extraction.approval import approve_pilot, require_pilot_approval
from src.extraction.mock_client import MockClient
from src.extraction.config import PROJECT_ROOT

TEXTS = ["The battery is great. Delivery was late and the box was damaged.",
         "Screen is bad, camera is excellent.", "Nothing relevant here at all folks.",
         "Customer service was terrible and refund never came."]


@pytest.fixture
def env(tmp_path):
    rows = []
    for i in range(14):
        rows.append({"review_id": f"R{i:03d}", "product_id": f"P{i%3}", "review_text": TEXTS[i % 4] + f" Unit {i}.",
                     "language": "English", "rating": 1 + i % 5})
    ds = tmp_path / "clean.csv"; pd.DataFrame(rows).to_csv(ds, index=False)
    return dict(dataset=ds, out_dir=tmp_path / "ext", reports_dir=tmp_path / "rep", client=MockClient(), require_key=False)


def lines(p): return [json.loads(l) for l in p.read_text().splitlines()]


def test_pilot_processes_exactly_ten_and_resumes(env):
    m = pipeline.run(**env)
    assert m["processed_this_run"] == 10 and m["successes_this_run"] == 10 and m["complete"]
    calls = env["client"].responses.calls
    m2 = pipeline.run(**env)
    assert m2["processed_this_run"] == 0 and m2["skipped_already_completed"] == 10
    assert env["client"].responses.calls == calls


def test_full_blocked_without_approval(env):
    pipeline.run(**env)
    with pytest.raises(SystemExit, match="no pilot approval"):
        pipeline.run(full=True, **env)


def test_cannot_approve_incomplete_pilot(env):
    with pytest.raises(SystemExit, match="Cannot approve"):
        approve_pilot(reviewer="x", reports_dir=env["reports_dir"], out_dir=env["out_dir"], dataset=env["dataset"])


def test_full_after_approval_skips_pilot_and_no_duplicates(env):
    pipeline.run(**env)
    approve_pilot(reviewer="tester", reports_dir=env["reports_dir"], out_dir=env["out_dir"], dataset=env["dataset"])
    calls = env["client"].responses.calls
    m = pipeline.run(full=True, workers=3, **env)
    assert m["eligible_reviews"] == 14 and m["reused_from_pilot"] == 10
    assert m["processed_this_run"] == 4 and env["client"].responses.calls == calls + 4
    assert m["complete"] and m["completed_total"] == 14
    ids = [r["review_id"] for r in lines(env["out_dir"] / "predictions.jsonl")]
    assert len(ids) == len(set(ids)) == 4
    m2 = pipeline.run(full=True, **env)
    assert m2["processed_this_run"] == 0 and m2["skipped_already_completed"] == 14


def test_stale_approval_rejected(env):
    pipeline.run(**env)
    approve_pilot(reviewer="t", reports_dir=env["reports_dir"], out_dir=env["out_dir"], dataset=env["dataset"])
    p = env["reports_dir"] / "pilot_approval.json"
    d = json.loads(p.read_text()); d["prompt_version"] = "old"; p.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="stale"):
        require_pilot_approval(reports_dir=env["reports_dir"], dataset=env["dataset"])


def test_missing_api_key_stops_before_any_call(env, monkeypatch):
    monkeypatch.setattr(pipeline, "API_KEY_PRESENT", False)
    env = {**env, "require_key": True, "client": None}
    with pytest.raises(SystemExit, match="OPENAI_API_KEY missing"):
        pipeline.run(**env)


class FlakyClient:
    """Fails with a 500 twice then succeeds; another review always fails validation."""
    def __init__(self):
        self.inner = MockClient(); self.n = 0; self.responses = self
    def parse(self, **kw):
        self.n += 1
        if self.n <= 2:
            e = RuntimeError("server"); e.status_code = 500; raise e
        return self.inner.responses.parse(**kw)


def test_transient_errors_retried(env):
    m = pipeline.run(**{**env, "client": FlakyClient(), "sleep": lambda s: None})
    assert m["successes_this_run"] == 10 and m["failures_this_run"] == 0


class AuthFail:
    def __init__(self): self.responses = self
    def parse(self, **kw):
        e = RuntimeError("bad key"); e.status_code = 401; raise e


def test_auth_failure_aborts_run(env):
    with pytest.raises(SystemExit, match="Authentication"):
        pipeline.run(**{**env, "client": AuthFail()})


class BadEvidence:
    def __init__(self): self.responses = self
    def parse(self, **kw):
        from src.extraction.schemas import ExtractionRecord
        import re
        rid = re.search(r"^review_id: (.*)$", kw["input"], re.M).group(1)
        pid = re.search(r"^product_id: (.*)$", kw["input"], re.M).group(1)
        text = kw["input"].split("review_text: ", 1)[1]
        from types import SimpleNamespace
        rec = ExtractionRecord.model_validate({"schema_version": "1.0", "review_id": rid, "product_id": pid,
              "review_text": text, "extraction_status": "success", "aspects": [{"aspect_id": "A1", "aspect_term": "x",
              "sentiment": "positive", "evidence": "INVENTED", "complaint_present": False, "complaint_evidence": None}]})
        return SimpleNamespace(output_parsed=rec, usage=None)


def test_hallucinated_evidence_logged_as_failure(env):
    m = pipeline.run(**{**env, "client": BadEvidence()})
    assert m["failures_this_run"] == 10 and m["successes_this_run"] == 0 and not m["complete"]
    f = lines(env["out_dir"] / "pilot_failures.jsonl")
    assert f[0]["category"] == "validation_or_output" and f[0]["attempts"] == 3
    assert not (env["out_dir"] / "pilot_predictions.jsonl").exists()

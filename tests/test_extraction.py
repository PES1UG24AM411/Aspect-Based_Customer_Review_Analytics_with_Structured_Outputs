import json
import pytest
from pydantic import ValidationError
from src.extraction.schemas import ExtractionRecord
from src.extraction.validators import validate_record
from src.extraction.retry_handler import ExtractionFailure

REVIEW="The camera is excellent, but the battery drains too quickly. Delivery was late."

def valid_payload():
    return {"schema_version":"1.0","review_id":"R1","product_id":"P1","review_text":REVIEW,
      "aspects":[
       {"aspect_id":"A1","aspect_term":"Camera","sentiment":"positive","evidence":"The camera is excellent",
        "complaint_present":False,"complaint_evidence":None},
       {"aspect_id":"A2","aspect_term":"Battery life","sentiment":"negative","evidence":"the battery drains too quickly",
        "complaint_present":True,"complaint_evidence":"the battery drains too quickly"},
       {"aspect_id":"A3","aspect_term":"Delivery","sentiment":"negative","evidence":"Delivery was late",
        "complaint_present":True,"complaint_evidence":"Delivery was late"}],
      "extraction_status":"success"}

def test_valid_json_and_multiple_aspects():
    obj=ExtractionRecord.model_validate_json(json.dumps(valid_payload()))
    assert len(obj.aspects)==3

def test_invalid_json():
    with pytest.raises(ValidationError):
        ExtractionRecord.model_validate_json("{not-json")

def test_invalid_sentiment():
    p=valid_payload(); p["aspects"][0]["sentiment"]="mixed"
    with pytest.raises(ValidationError): ExtractionRecord.model_validate(p)

def test_missing_ids_rejected():
    p=valid_payload(); p.pop("review_id")
    with pytest.raises(ValidationError): ExtractionRecord.model_validate(p)

def test_evidence_not_found():
    obj=ExtractionRecord.model_validate(valid_payload())
    changed=valid_payload()
    changed["aspects"][0]["evidence"]="camera quality is wonderful"
    bad=ExtractionRecord.model_validate(changed)
    with pytest.raises(ValueError, match="Evidence"):
        validate_record(bad,expected_review_id="R1",expected_product_id="P1",
                        expected_review_text=REVIEW)

def test_complaint_inconsistency():
    p=valid_payload(); p["aspects"][0]["complaint_evidence"]="camera is excellent"
    with pytest.raises(ValidationError): ExtractionRecord.model_validate(p)

def test_complaint_true_requires_evidence():
    p=valid_payload(); p["aspects"][1]["complaint_evidence"]=None
    with pytest.raises(ValidationError): ExtractionRecord.model_validate(p)

def test_empty_aspect_list_valid():
    p=valid_payload(); p["aspects"]=[]
    assert ExtractionRecord.model_validate(p).aspects==[]

def test_duplicate_aspect_ids_rejected():
    p=valid_payload(); p["aspects"][1]["aspect_id"]="A1"
    with pytest.raises(ValidationError): ExtractionRecord.model_validate(p)

def test_ids_and_text_must_be_preserved():
    obj=ExtractionRecord.model_validate(valid_payload())
    with pytest.raises(ValueError, match="review_id"):
        validate_record(obj,expected_review_id="OTHER",expected_product_id="P1",expected_review_text=REVIEW)

def test_failure_type_records_attempts():
    err=ExtractionFailure("validation_or_output","bad evidence",3)
    assert err.category=="validation_or_output" and err.attempts==3


def test_semantic_retry_then_success():
    from types import SimpleNamespace
    from src.extraction.llm_client import extract_one
    from src.extraction.retry_handler import ExtractionFailure
    good=ExtractionRecord.model_validate(valid_payload())
    bad_payload=valid_payload(); bad_payload["aspects"][0]["evidence"]="invented evidence"
    bad=ExtractionRecord.model_validate(bad_payload)
    class FakeResponses:
        def __init__(self): self.calls=0
        def parse(self, **kwargs):
            self.calls+=1
            return SimpleNamespace(output_parsed=bad if self.calls==1 else good, usage=None)
    fake=SimpleNamespace(responses=FakeResponses())
    parsed,_=extract_one({"review_id":"R1","product_id":"P1","review_text":REVIEW},client=fake)
    assert parsed.review_id=="R1" and fake.responses.calls==2

def test_semantic_retry_exhaustion():
    from types import SimpleNamespace
    from src.extraction.llm_client import extract_one
    bad_payload=valid_payload(); bad_payload["aspects"][0]["evidence"]="not in source"
    bad=ExtractionRecord.model_validate(bad_payload)
    class FakeResponses:
        def __init__(self): self.calls=0
        def parse(self, **kwargs):
            self.calls+=1
            return SimpleNamespace(output_parsed=bad, usage=None)
    fake=SimpleNamespace(responses=FakeResponses())
    with pytest.raises(ExtractionFailure) as err:
        extract_one({"review_id":"R1","product_id":"P1","review_text":REVIEW},client=fake)
    assert err.value.attempts==3 and fake.responses.calls==3


def test_offline_preflight_does_not_call_model(monkeypatch):
    import src.extraction.pipeline as pipeline

    def forbidden_api_call(*args, **kwargs):
        raise AssertionError("offline preflight must not call the model")

    monkeypatch.setattr(pipeline, "extract_one", forbidden_api_call)
    result = pipeline.preflight()
    assert result["status"] == "offline_preflight_passed"
    assert result["api_call_made"] is False
    assert result["pilot_size"] == 10


def test_jsonl_writer_uses_real_newline(tmp_path):
    from src.extraction.pipeline import append_jsonl

    target = tmp_path / "records.jsonl"
    append_jsonl(target, {"review_id": "R1"})
    append_jsonl(target, {"review_id": "R2"})
    lines = target.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["review_id"] for line in lines] == ["R1", "R2"]


def test_json_recovery_from_markdown_fence():
    from src.extraction.json_utils import extract_json_object

    result = extract_json_object('```json\n{"review_id":"R1"}\n```')
    assert result == {"review_id": "R1"}


def test_json_recovery_with_surrounding_text():
    from src.extraction.json_utils import extract_json_object

    result = extract_json_object(
        'Here is the result:\n{"review_id":"R1","text":"value"}\nDone.'
    )
    assert result == {"review_id": "R1", "text": "value"}


def test_json_recovery_handles_braces_inside_strings():
    from src.extraction.json_utils import extract_json_object

    result = extract_json_object('{"review_text":"battery {issue}"}')
    assert result["review_text"] == "battery {issue}"


def test_json_recovery_rejects_malformed_json():
    from src.extraction.json_utils import extract_json_object

    with pytest.raises(ValueError, match="malformed"):
        extract_json_object('{"review_id":}')

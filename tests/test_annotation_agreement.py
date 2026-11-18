import pytest

from src.evaluation.annotation_agreement import calculate_agreement


def make_record(aspects, status="complete"):
    return {
        "review_id": "R1",
        "product_id": "P1",
        "review_text": "The camera is great but the battery is poor.",
        "annotator": "test",
        "status": status,
        "aspects": aspects,
        "notes": "",
    }


def make_aspect(term, sentiment, evidence, complaint=False):
    return {
        "aspect_term": term,
        "sentiment": sentiment,
        "evidence": evidence,
        "complaint_present": complaint,
    }


def test_perfect_agreement():
    aspects = [
        make_aspect("camera", "positive", "camera is great"),
        make_aspect("battery", "negative", "battery is poor", True),
    ]

    report = calculate_agreement(
        {"R1": make_record(aspects)},
        {"R1": make_record(aspects)},
        ["R1"],
    )

    assert report["aspect_summary"]["matched_aspects"] == 2
    assert report["sentiment_agreement"]["accuracy"] == 1.0
    assert report["sentiment_agreement"]["cohen_kappa"] == 1.0
    assert report["complaint_agreement"]["accuracy"] == 1.0


def test_sentiment_disagreement_is_reported():
    a = [make_aspect("camera", "positive", "camera is great")]
    b = [make_aspect("camera", "negative", "camera is great", True)]

    report = calculate_agreement(
        {"R1": make_record(a)},
        {"R1": make_record(b)},
        ["R1"],
    )

    assert report["aspect_summary"]["matched_aspects"] == 1
    assert report["sentiment_agreement"]["accuracy"] == 0.0
    assert len(report["disagreements"]) == 1


def test_unmatched_aspects_are_counted():
    a = [make_aspect("camera", "positive", "camera is great")]
    b = [make_aspect("battery", "negative", "battery is poor")]

    report = calculate_agreement(
        {"R1": make_record(a)},
        {"R1": make_record(b)},
        ["R1"],
    )

    assert report["aspect_summary"]["matched_aspects"] == 0
    assert report["aspect_summary"]["unmatched_annotator_1"] == 1
    assert report["aspect_summary"]["unmatched_annotator_2"] == 1
    assert report["sentiment_agreement"]["cohen_kappa"] is None


def test_incomplete_annotation_is_rejected():
    a = make_record([], status="todo")
    b = make_record([])

    with pytest.raises(ValueError, match="not completed"):
        calculate_agreement({"R1": a}, {"R1": b}, ["R1"])


def test_different_review_text_is_rejected():
    a = make_record([])
    b = make_record([])
    b["review_text"] = "Different review."

    with pytest.raises(ValueError, match="Review text differs"):
        calculate_agreement({"R1": a}, {"R1": b}, ["R1"])


def test_duplicate_shared_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        calculate_agreement({}, {}, ["R1", "R1"])


import sys
from pathlib import Path

import pandas as pd
import pytest

# Make src importable when pytest runs from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import preprocessing as prep


# -------------------------------------------------
# TEST 1: HTML ENTITY CLEANING
# -------------------------------------------------

def test_html_entity_cleaning():

    text = "Great &amp; useful product"

    result = prep.clean_review_text(text)

    assert result == "Great & useful product"


# -------------------------------------------------
# TEST 2: HTML TAG REMOVAL
# -------------------------------------------------

def test_html_tag_removal():

    text = "<p>Excellent product</p>"

    result = prep.clean_review_text(text)

    assert result == "Excellent product"


# -------------------------------------------------
# TEST 3: SCRAPING ARTIFACT REMOVAL
# -------------------------------------------------

def test_scraping_artifact_removal():

    text = (
        "Excellent product. "
        "[Read more Read less] "
        "15 people found this helpful"
    )

    result = prep.clean_review_text(text)

    assert "Read more" not in result
    assert "found this helpful" not in result
    assert "Excellent product." in result


# -------------------------------------------------
# TEST 4: NEGATION PRESERVATION
# -------------------------------------------------

def test_negation_preservation():

    text = "This product is NOT good."

    result = prep.clean_review_text(text)

    assert "NOT" in result


# -------------------------------------------------
# TEST 5: EMOJI PRESERVATION
# -------------------------------------------------

def test_emoji_preservation():

    text = "Amazing product! ❤️"

    result = prep.clean_review_text(text)

    assert "❤️" in result


# -------------------------------------------------
# TEST 6: WHITESPACE NORMALIZATION
# -------------------------------------------------

def test_whitespace_normalization():

    text = "Very     good     product"

    result = prep.clean_review_text(text)

    assert result == "Very good product"


# -------------------------------------------------
# TEST 7: EMPTY TEXT
# -------------------------------------------------

def test_empty_text():

    assert prep.clean_review_text(None) == ""


# -------------------------------------------------
# TEST 8: ASIN EXTRACTION
# -------------------------------------------------

def test_asin_extraction():

    url = (
        "https://www.amazon.com/dp/"
        "B0ABC12345"
    )

    result = prep.extract_asin(url)

    assert result == "B0ABC12345"


# -------------------------------------------------
# TEST 9: GP PRODUCT ASIN EXTRACTION
# -------------------------------------------------

def test_gp_product_asin_extraction():

    url = (
        "https://www.amazon.com/gp/product/"
        "B0ABC12345"
    )

    result = prep.extract_asin(url)

    assert result == "B0ABC12345"


# -------------------------------------------------
# TEST 10: INVALID ASIN
# -------------------------------------------------

def test_invalid_asin():

    result = prep.extract_asin(
        "https://www.amazon.com/example"
    )

    assert result == ""


# -------------------------------------------------
# TEST 11: SHORT REVIEW IS UNCERTAIN
# -------------------------------------------------

def test_short_review_is_uncertain():

    result = prep.detect_review_language(
        "Good product"
    )

    assert result["language"] == "uncertain"
    assert result["detection_status"] == "too_short"


# -------------------------------------------------
# TEST 12: EMPTY REVIEW LANGUAGE
# -------------------------------------------------

def test_empty_review_language():

    result = prep.detect_review_language("")

    assert result["language"] == "uncertain"
    assert result["detection_status"] == "empty_text"


# -------------------------------------------------
# TEST 13: DUPLICATE HANDLING
# -------------------------------------------------

def test_exact_duplicate_within_same_asin():

    df = pd.DataFrame({
        "review_id": [
            "AMZ-000001",
            "AMZ-000002",
            "AMZ-000003"
        ],
        "source_row": [2, 3, 4],
        "product_id": [
            "B0ABC12345",
            "B0ABC12345",
            "B0XYZ12345"
        ],
        "review_text": [
            "Excellent product!",
            "Excellent product!",
            "Excellent product!"
        ]
    })

    retained, removed = prep.identify_duplicates(df)

    # Same ASIN duplicate should be removed.
    assert len(retained) == 2

    # Identical text across different products stays.
    assert len(removed) == 1

    assert (
        removed.iloc[0]["removal_reason"]
        == "exact_duplicate_within_same_asin"
    )


# -------------------------------------------------
# TEST 14: CASE-INSENSITIVE DUPLICATES
# -------------------------------------------------

def test_case_insensitive_duplicate():

    df = pd.DataFrame({
        "review_id": [
            "AMZ-000001",
            "AMZ-000002"
        ],
        "source_row": [2, 3],
        "product_id": [
            "B0ABC12345",
            "B0ABC12345"
        ],
        "review_text": [
            "Excellent Product",
            "excellent product"
        ]
    })

    retained, removed = prep.identify_duplicates(df)

    assert len(retained) == 1
    assert len(removed) == 1


# -------------------------------------------------
# TEST 15: MISSING ASIN RECORDS RETAINED
# -------------------------------------------------

def test_missing_asin_records_not_deduplicated():

    df = pd.DataFrame({
        "review_id": [
            "AMZ-000001",
            "AMZ-000002"
        ],
        "source_row": [2, 3],
        "product_id": ["", ""],
        "review_text": [
            "Same review text",
            "Same review text"
        ]
    })

    retained, removed = prep.identify_duplicates(df)

    assert len(retained) == 2
    assert len(removed) == 0


# -------------------------------------------------
# TEST 16: WORD COUNT
# -------------------------------------------------

def test_word_count():

    assert prep.word_count(
        "This is a good product"
    ) == 5


# -------------------------------------------------
# TEST 17: DUPLICATE NORMALIZATION
# -------------------------------------------------

def test_duplicate_normalization():

    result = prep.normalize_duplicate_text(
        "  GREAT   Product  "
    )

    assert result == "great product"


# -------------------------------------------------
# TEST 18: LANGUAGE RESULT STRUCTURE
# -------------------------------------------------

def test_language_result_structure():

    result = prep.detect_review_language(
        "Good product"
    )

    assert "language" in result
    assert "confidence" in result
    assert "confidence_margin" in result
    assert "detection_status" in result

# -------------------------------------------------
# TEST 19: FULL LANGUAGE DETECTOR CONFIGURATION
# -------------------------------------------------

def test_detector_uses_all_languages():

    assert prep.DETECTOR is not None


# -------------------------------------------------
# TEST 20: UNSUPPORTED LANGUAGE CLASSIFICATION
# -------------------------------------------------

def test_unsupported_language_is_uncertain(monkeypatch):

    class MockLanguage:
        GERMAN = "GERMAN"

    class MockConfidence:
        def __init__(self, language, value):
            self.language = language
            self.value = value

    class MockDetector:

        def compute_language_confidence_values(self, text):

            return [
                MockConfidence(
                    MockLanguage.GERMAN,
                    0.95
                ),
                MockConfidence(
                    prep.Language.ENGLISH,
                    0.03
                ),
                MockConfidence(
                    prep.Language.FRENCH,
                    0.02
                )
            ]

    monkeypatch.setattr(
        prep,
        "DETECTOR",
        MockDetector()
    )

    result = prep.detect_review_language(
        "Dies ist ein sehr guter und zuverlässiger Artikel"
    )

    assert result["language"] == "uncertain"

    assert (
        result["detection_status"]
        == "unsupported_language"
    )

    assert result["confidence"] == 0.95


# -------------------------------------------------
# TEST 21: CONFIDENCE VALUES PRESERVED
# -------------------------------------------------

def test_confidence_values_preserved(monkeypatch):

    class MockConfidence:
        def __init__(self, language, value):
            self.language = language
            self.value = value

    class MockDetector:

        def compute_language_confidence_values(self, text):

            return [
                MockConfidence(
                    prep.Language.ENGLISH,
                    0.90
                ),
                MockConfidence(
                    prep.Language.SPANISH,
                    0.05
                ),
                MockConfidence(
                    prep.Language.FRENCH,
                    0.05
                )
            ]

    monkeypatch.setattr(
        prep,
        "DETECTOR",
        MockDetector()
    )

    result = prep.detect_review_language(
        "This product works very well and provides excellent quality"
    )

    assert result["language"] == "English"
    assert result["confidence"] == 0.90
    assert result["confidence_margin"] == 0.85
    assert result["detection_status"] == "accepted"


# -------------------------------------------------
# TEST 22: LOW CONFIDENCE STILL UNCERTAIN
# -------------------------------------------------

def test_low_confidence_remains_uncertain(monkeypatch):

    class MockConfidence:
        def __init__(self, language, value):
            self.language = language
            self.value = value

    class MockDetector:

        def compute_language_confidence_values(self, text):

            return [
                MockConfidence(
                    prep.Language.ENGLISH,
                    0.70
                ),
                MockConfidence(
                    prep.Language.SPANISH,
                    0.20
                ),
                MockConfidence(
                    prep.Language.FRENCH,
                    0.10
                )
            ]

    monkeypatch.setattr(
        prep,
        "DETECTOR",
        MockDetector()
    )

    result = prep.detect_review_language(
        "This product works very well and provides excellent quality"
    )

    assert result["language"] == "uncertain"
    assert result["detection_status"] == "low_confidence"
    assert result["confidence"] == 0.70


# -------------------------------------------------
# CONSERVATIVE AMAZON INTERFACE CLEANING
# -------------------------------------------------

def test_helpful_report_interface_removed_but_opinion_preserved():
    assert prep.clean_review_text("Helpful [Report] Great quality!") == "Great quality!"
    assert prep.clean_review_text("This helpful product summary is useful.") == "This helpful product summary is useful."


def test_exact_scraping_error_removed():
    result = prep.clean_review_text("Good fit. There was a problem filtering reviews. Very comfy.")
    assert result == "Good fit. Very comfy."


def test_amazon_translation_metadata_removed_review_preserved():
    result = prep.clean_review_text("Me encanta este producto. Translated from Spanish by Amazon")
    assert result == "Me encanta este producto."


def test_javascript_link_markup_removed_and_link_text_preserved():
    result = prep.clean_review_text('Works well <a href="javascript:void(0)">open details</a> after setup.')
    assert result == "Works well open details after setup."


def test_product_summary_removed_only_with_known_section_boundary():
    result = prep.clean_review_text(
        "My review is genuine.\nProduct Summary\nGenerated overview here.\nCustomer Reviews\nI love it!"
    )
    assert "Generated overview here" not in result
    assert "My review is genuine." in result
    assert "Customer Reviews" in result
    assert "I love it!" in result


def test_product_summary_without_boundary_does_not_delete_review_body():
    result = prep.clean_review_text("Product Summary\nI personally found it durable and comfortable.")
    assert "I personally found it durable and comfortable." in result


def test_cleaning_preserves_negation_punctuation_and_emoji():
    result = prep.clean_review_text("Not bad—actually GREAT!!! ❤️ Helpful [Report]")
    assert result == "Not bad—actually GREAT!!! ❤️"



def test_inline_hash_product_summary_metadata_removed():
    result = prep.clean_review_text(
        "Battery is excellent and the phone looks new. # Product Summary: Apple iPhone 15, 128GB"
    )
    assert result == "Battery is excellent and the phone looks new."


def test_plain_product_summary_opinion_is_preserved():
    text = "The product summary in the box was helpful, but the product itself is excellent."
    assert prep.clean_review_text(text) == text


def test_amazon_javascript_and_feedback_fragments_removed():
    result = prep.clean_review_text(
        "Works perfectly. (javascript:void(0)) [Report](javascript:void(0)) "
        "Sending feedback...Thanks, we'll investigate in the next few days."
    )
    assert result == "Works perfectly."


def test_amazon_translation_and_expandable_ui_removed():
    result = prep.clean_review_text(
        "Muy bueno. Sorry, we couldn't translate the review. "
        "Brief content visible, double tap to read full content."
    )
    assert result == "Muy bueno."


def test_product_summary_marker_removal_preserves_review_punctuation_emoji():
    result = prep.clean_review_text(
        "Not bad—actually GREAT!!! ❤️ # Product Summary: device details"
    )
    assert result == "Not bad—actually GREAT!!! ❤️"


def test_double_hash_product_summary_metadata_removed():
    result = prep.clean_review_text(
        "My watch arrived scratched. ## Product Summary: Apple Watch SE 3 * product specs"
    )
    assert result == "My watch arrived scratched."


def test_exact_amazon_recommendation_footer_removed_but_generic_summary_preserved():
    footer = (
        "Product summary presents key information. Close to see all product details. "
        "Your recently viewed items and featured recommendations After viewing product detail pages, "
        "look here to find an easy way to navigate back to pages you are interested in."
    )
    assert prep.clean_review_text("The product summary was useful.") == "The product summary was useful."
    assert prep.clean_review_text("My review. " + footer) == "My review."


# -------------------------------------------------
# FINAL CLEANING FIX REGRESSION TESTS
# -------------------------------------------------

def test_standalone_feedback_suffix_removed_without_sending_feedback_prefix():
    result = prep.clean_review_text(
        "The product works as expected. Thanks, we'll investigate in the next few days."
    )
    assert result == "The product works as expected."


def test_standalone_feedback_suffix_removed_after_thank_you_feedback_ui():
    result = prep.clean_review_text(
        "Good quality. Thank you for your feedback. Thanks, we'll investigate in the next few days."
    )
    assert result == "Good quality. Thank you for your feedback."


def test_feedback_suffix_inside_customer_authored_sentence_is_preserved():
    text = (
        "The seller wrote, Thanks, we'll investigate in the next few days. "
        "I am still waiting for a response."
    )
    assert prep.clean_review_text(text) == text


def test_quoted_feedback_sentence_is_preserved():
    text = "The seller replied: \"Thanks, we\'ll investigate in the next few days.\""
    assert prep.clean_review_text(text) == text


def test_matching_duplicate_heading_markup_removed_but_title_and_body_preserved():
    title = "##### Good phone, but 84% battery health and too expensive"
    body = (
        "##### Good phone, but 84% battery health and too expensive "
        "The phone arrived with a small scratch."
    )
    clean_title, clean_body = prep.remove_duplicated_leading_heading(title, body)
    assert clean_title == "Good phone, but 84% battery health and too expensive"
    assert clean_body == (
        "Good phone, but 84% battery health and too expensive "
        "The phone arrived with a small scratch."
    )


def test_duplicate_heading_requires_same_marker_and_matching_title_prefix():
    title = "##### Good phone"
    mismatched_marker = "## Good phone The phone arrived."
    mismatched_title = "##### Good phones are great."
    assert prep.remove_duplicated_leading_heading(title, mismatched_marker) == (
        title, mismatched_marker
    )
    assert prep.remove_duplicated_leading_heading(title, mismatched_title) == (
        title, mismatched_title
    )


def test_unpaired_heading_marker_is_preserved():
    title = "Good phone"
    body = "##### Good phone The phone arrived."
    assert prep.remove_duplicated_leading_heading(title, body) == (title, body)


def test_matching_heading_removal_allows_source_title_truncation():
    title = "##### I have part time tinnitus and for sleep, these are a blessing to have especially as a side sle"
    body = (
        "##### I have part time tinnitus and for sleep, these are a blessing to have especially as a side sleeper. "
        "WHY???"
    )
    clean_title, clean_body = prep.remove_duplicated_leading_heading(title, body)
    assert clean_title == title.removeprefix("##### ")
    assert clean_body == body.removeprefix("##### ")

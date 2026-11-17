
#!/usr/bin/env python3

"""
Member 1 - Stage 1: Dataset Preprocessing

Project:
Aspect-Based Customer Review Analytics with Structured Outputs

Features:
1. Preserves the original CSV.
2. Cleans review text without paraphrasing.
3. Detects English, Spanish and French using Lingua.
4. Separates uncertain-language reviews.
5. Removes empty reviews.
6. Removes exact duplicates only within the same ASIN.
7. Maintains an exclusion audit.
8. Generates standardized CSVs, statistics and a report.

No aspect annotation, sentiment labelling or model training.
"""

from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import argparse
import html
import json
import re
import sys

import pandas as pd

from lingua import Language, LanguageDetectorBuilder


# ---------------------------------------------------------
# 1. PROJECT CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ARE_CustomerReviews_Cleaned_v2.csv"
)

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "preprocessing_report.md"

SUPPORTED_LANGUAGES = {
    Language.ENGLISH: "English",
    Language.SPANISH: "Spanish",
    Language.FRENCH: "French",
}

MIN_WORDS = 8
MIN_CONFIDENCE = 0.80
MIN_CONFIDENCE_MARGIN = 0.15

DETECTOR = (
    LanguageDetectorBuilder
    .from_all_languages()
    .with_preloaded_language_models()
    .build()
)


# ---------------------------------------------------------
# 2. TEXT CLEANING
# ---------------------------------------------------------

ARTIFACT_PATTERNS = [
    r"\[\s*read\s+more\s+read\s+less\s*\]",
    r"\[\s*read\s+more\s*\]",
    r"\[\s*read\s+less\s*\]",
    r"\b\d[\d,]*\s+people?\s+found\s+this helpful\b",
    r"\bone person found this helpful\b",
    r"\bsending feedback\.\.\.\s*thank you for your feedback\.?",
    r"\bsorry,\s*we failed to record your vote\.?\s*please try again\.?",
    r"\bsorry,\s*we failed to report this review\.?\s*please try again\.?",
    # Amazon review action strip. Keep ordinary uses of “helpful” intact.
    r"\bHelpful\s*\[\s*Report\s*\]",
    r"\bHelpful\s*\|\s*Report\b",
    # Translation notice is metadata, not the translated customer review.
    r"\bTranslated\s+from\s+(?:Spanish|French|English)\s+by\s+Amazon\b",
    r"\bThere was a problem filtering reviews\.?",
    # JavaScript-backed Amazon action links and their UI labels.
    r"\[[^\]]*\]\(\s*javascript:void\(0\)\s*\)",
    r"\(?javascript:void\(0\)\)?",
    # Amazon feedback / translation / expandable-review interface text.
    r"sending feedback\.\.\.\s*thanks,?\s*we(?:'|’)ll investigate in the next few days\.?",
    r"sorry,?\s*we couldn['’]t translate the review\.?",
    r"thank you for your feedback\.\s*you selected\s*[\"“](?:yes, it is useful|no, it is not useful)[\"”]\.?",
    r"brief content visible,?\s*double tap to read full content\.?",
    r"full content visible,?\s*double tap to read brief content\.?",
    r"please reload the page\.?",
    # Exact Amazon navigation/recommendation footer, not ordinary review text.
    r"product summary presents key information\.\s*close to see all product details\.\s*"
    r"your recently viewed items and featured recommendations\s*"
    r"after viewing product detail pages, look here to find an easy way to navigate back to pages you are interested in\.?",
]

COMPILED_ARTIFACTS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in ARTIFACT_PATTERNS
]

# Product Summary is an Amazon-generated section. Remove its contents only
# when a recognizable following section heading gives us a safe boundary.
PRODUCT_SUMMARY_BOUNDARY = re.compile(
    r"(?im)^\s*(?:Customer Reviews?|Top Reviews?|Reviews?\s+with\s+images|"
    r"Questions?\s*&\s*Answers?|About this item|Product details|"
    r"Frequently bought together)\s*:??\s*$"
)


def _remove_product_summary(text):
    """Remove Amazon's explicitly labelled appended Product Summary metadata.

    The source dataset commonly appends this as ``# Product Summary: ...``
    on the same line as the customer review. The exact hash-prefixed label is
    an unambiguous metadata boundary, so everything after it is discarded.
    Plain mentions such as "product summary" are deliberately untouched.
    """
    appended = re.search(r"(?i)\s+#{1,6}\s*Product\s+Summary\s*:", text)
    if appended:
        return text[:appended.start()].rstrip()

    # Retain the conservative handling for a standalone heading format.
    heading = re.compile(r"(?im)^\s*Product Summary\s*:?\s*$")
    match = heading.search(text)
    if not match:
        return text

    boundary = PRODUCT_SUMMARY_BOUNDARY.search(text, match.end())
    if boundary:
        return text[:match.start()] + " " + text[boundary.start():]

    # No clear boundary: preserve the body rather than guessing.
    return text


def remove_terminal_feedback_suffix(text):
    """Remove the exact Amazon feedback sentence only as a terminal suffix."""
    return re.sub(
        r"(?:^|\s)Thanks, we['’]ll investigate in the next few days\.\s*$",
        " ",
        text,
        flags=re.IGNORECASE,
    ).strip()


def has_terminal_feedback_suffix(value):
    """Return whether a review ends with the exact standalone feedback suffix."""
    if pd.isna(value):
        return False
    return bool(re.search(
        r"(?:^|\s)Thanks, we['’]ll investigate in the next few days\.\s*$",
        str(value),
        flags=re.IGNORECASE,
    ))


def remove_duplicated_leading_heading(review_title, review_text):
    """Remove matching leading Markdown heading markup from standardized fields.

    This is deliberately narrow: both title and body must start with the same
    1-6 hash heading marker, and the body text after that marker must begin
    with the complete title text after its marker. The title wording and body
    wording are preserved; only the leading markup is removed.
    """
    title = "" if pd.isna(review_title) else str(review_title)
    body = "" if pd.isna(review_text) else str(review_text)

    title_match = re.match(r"^\s*(#{1,6})[ \t]+(.+?)\s*$", title)
    body_match = re.match(r"^\s*(#{1,6})[ \t]+(.+)$", body, flags=re.DOTALL)

    if not title_match or not body_match:
        return title, body

    title_marker, title_text = title_match.groups()
    body_marker, body_after_marker = body_match.groups()

    if title_marker != body_marker:
        return title, body

    title_text = title_text.strip()
    body_after_marker = body_after_marker.lstrip()

    if not body_after_marker.startswith(title_text):
        return title, body

    boundary_index = len(title_text)
    if (
        len(body_after_marker) > boundary_index
        and body_after_marker[boundary_index].isalnum()
        and len(title.strip()) != 100
    ):
        # Avoid treating a partial word as a title match. The source title
        # field is capped at 100 characters, so a 100-character field may be
        # truncated mid-word and is accepted only with matching heading markup.
        return title, body

    return title_text, body_after_marker


def clean_review_text(value, *, remove_feedback_suffix=True):
    """
    Clean identifiable interface artifacts and formatting.

    Does not paraphrase, summarize or rewrite reviews. Preserves ordinary
    customer wording, punctuation, negations and emojis.
    """
    if pd.isna(value):
        return ""

    text = str(value)
    text = html.unescape(text)

    # Strip HTML tags; link text remains, so surrounding review wording survives.
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\xa0", " ")

    # Remove bounded Amazon-generated summary block before individual notices.
    text = _remove_product_summary(text)

    for pattern in COMPILED_ARTIFACTS:
        text = pattern.sub(" ", text)

    # Remove the terminal feedback sentence independently of any preceding
    # "Sending feedback..." UI phrase. The pipeline can disable this one step
    # temporarily to count how many suffixes were actually removed.
    if remove_feedback_suffix:
        text = remove_terminal_feedback_suffix(text)

    # Remove a trailing action strip only when its identifiable interface cue
    # is present; do not broadly remove every occurrence of “Helpful”.
    text = re.sub(
        r"\bHelpful\s+Sending feedback.*$",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------
# 3. LANGUAGE DETECTION
# ---------------------------------------------------------

def detect_review_language(text):
    """
    Detect language using Lingua's full supported language set.

    Only English, Spanish and French are accepted.

    Other detected languages are classified as uncertain
    with status unsupported_language.

    Short, ambiguous and low-confidence reviews are
    classified as uncertain.
    """

    if not text or not text.strip():
        return {
            "language": "uncertain",
            "confidence": 0.0,
            "confidence_margin": 0.0,
            "detection_status": "empty_text"
        }

    words = re.findall(
        r"\b[\w'-]+\b",
        text,
        flags=re.UNICODE
    )

    if len(words) < MIN_WORDS:
        return {
            "language": "uncertain",
            "confidence": 0.0,
            "confidence_margin": 0.0,
            "detection_status": "too_short"
        }

    try:
        confidence_values = (
            DETECTOR.compute_language_confidence_values(text)
        )

        if not confidence_values:
            return {
                "language": "uncertain",
                "confidence": 0.0,
                "confidence_margin": 0.0,
                "detection_status": "no_detection"
            }

        ranked = sorted(
            confidence_values,
            key=lambda item: item.value,
            reverse=True
        )

        best = ranked[0]

        second_confidence = (
            ranked[1].value if len(ranked) > 1 else 0.0
        )

        confidence = float(best.value)

        margin = confidence - second_confidence

        # Reject languages outside the three target languages.
        language_name = SUPPORTED_LANGUAGES.get(
            best.language
        )

        if language_name is None:
            return {
                "language": "uncertain",
                "confidence": confidence,
                "confidence_margin": margin,
                "detection_status": "unsupported_language"
            }

        # Reject low-confidence predictions.
        if confidence < MIN_CONFIDENCE:
            return {
                "language": "uncertain",
                "confidence": confidence,
                "confidence_margin": margin,
                "detection_status": "low_confidence"
            }

        # Reject ambiguous predictions.
        if margin < MIN_CONFIDENCE_MARGIN:
            return {
                "language": "uncertain",
                "confidence": confidence,
                "confidence_margin": margin,
                "detection_status": "ambiguous"
            }

        return {
            "language": language_name,
            "confidence": confidence,
            "confidence_margin": margin,
            "detection_status": "accepted"
        }

    except Exception as exc:
        return {
            "language": "uncertain",
            "confidence": 0.0,
            "confidence_margin": 0.0,
            "detection_status": (
                f"detection_error: {type(exc).__name__}"
            )
        }

# ---------------------------------------------------------
# 4. PRODUCT ID EXTRACTION
# ---------------------------------------------------------

def extract_asin(product_url):
    """
    Extract Amazon ASIN from Product URL.

    Returns empty string when extraction fails.
    Does not invent product identifiers.
    """

    if pd.isna(product_url):
        return ""

    match = re.search(
        r"/(?:dp|gp/product)/([A-Z0-9]{10})(?:[/?#]|$)",
        str(product_url),
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return ""


# ---------------------------------------------------------
# 5. REVIEW LENGTH
# ---------------------------------------------------------

def word_count(text):
    if pd.isna(text):
        return 0

    return len(
        re.findall(
            r"\b[\w'-]+\b",
            str(text),
            flags=re.UNICODE
        )
    )


def length_statistics(series):

    lengths = series.fillna("").map(word_count)

    if lengths.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None
        }

    return {
        "count": int(lengths.count()),
        "mean": round(float(lengths.mean()), 2),
        "median": round(float(lengths.median()), 2),
        "minimum": int(lengths.min()),
        "maximum": int(lengths.max())
    }


# ---------------------------------------------------------
# 6. DUPLICATE HANDLING
# ---------------------------------------------------------

def normalize_duplicate_text(text):
    """
    Normalization is used only for duplicate comparison.

    Original cleaned review text is not modified.
    """

    text = str(text).casefold()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def identify_duplicates(df):

    """
    Exact duplicates are removed only within the same ASIN.

    Reviews without an ASIN are not automatically
    deduplicated because product identity is unknown.

    First occurrence is retained.
    """

    seen = set()

    retained_indices = []

    removed_records = []

    for index, row in df.iterrows():

        product_id = row["product_id"]

        review_text = normalize_duplicate_text(
            row["review_text"]
        )

        if not product_id:
            retained_indices.append(index)
            continue

        duplicate_key = (
            product_id,
            review_text
        )

        if duplicate_key in seen:

            removed_records.append({
                "review_id": row["review_id"],
                "source_row": row["source_row"],
                "product_id": product_id,
                "review_text": row["review_text"],
                "removal_reason": (
                    "exact_duplicate_within_same_asin"
                )
            })

        else:

            seen.add(duplicate_key)

            retained_indices.append(index)

    retained = df.loc[retained_indices].copy()

    removed = pd.DataFrame(
        removed_records,
        columns=[
            "review_id",
            "source_row",
            "product_id",
            "review_text",
            "removal_reason"
        ]
    )

    return retained, removed


# ---------------------------------------------------------
# 7. REPORT GENERATION
# ---------------------------------------------------------

def generate_report(stats, output_path):

    report = f"""# Stage 1 – Dataset Preprocessing Report

## 1. Overview

Project: Aspect-Based Customer Review Analytics
with Structured Outputs.

Preprocessing only. No aspect annotation,
sentiment labelling or model training was performed.

## 2. Dataset statistics

| Metric | Count |
|---|---:|
| Original records | {stats['original_records']} |
| English reviews | {stats['english_reviews']} |
| Spanish reviews | {stats['spanish_reviews']} |
| French reviews | {stats['french_reviews']} |
| Uncertain reviews | {stats['uncertain_reviews']} |
| Empty reviews removed | {stats['empty_reviews_removed']} |
| Exact duplicates removed | {stats['duplicates_removed']} |
| Total retained reviews | {stats['retained_reviews']} |
| Unique products | {stats['unique_products']} |
| Missing ASINs | {stats['missing_asins']} |

## 3. Language detection

Library: Lingua language detector.

Supported languages:
- English
- Spanish
- French

Minimum review length: {MIN_WORDS} words.

Minimum confidence: {MIN_CONFIDENCE}.

Minimum confidence margin: {MIN_CONFIDENCE_MARGIN}.

Short, ambiguous and low-confidence reviews
are placed in the uncertain-language dataset.

Lingua evaluates its full available language set. Only English,
Spanish and French are accepted into language-specific partitions;
other detected languages are assigned to the uncertain partition.

Manual verification is recommended before annotation.

## 4. Duplicate handling

Exact duplicates are compared using cleaned review text
and extracted ASIN.

Only duplicate reviews belonging to the same ASIN
are removed.

The first occurrence is retained.

Reviews with missing ASINs are not automatically
deduplicated.

All removed rows are recorded in the removal audit.

## 5. Text cleaning

Cleaning includes:

- HTML entity decoding.
- HTML tag removal.
- Removal of identifiable Amazon interface and translation artifacts.
- Removal of explicitly labelled appended ``# Product Summary:`` metadata.
- Removal of the exact standalone Amazon feedback suffix
  ``Thanks, we'll investigate in the next few days.`` when appended at the end
  of a review body. Occurrences inside customer-authored sentences or quotations
  are preserved.
- Conservative removal of matching leading Markdown heading markup only when
  both standardized title and body carry the same marker and the body begins
  with the complete title after the marker is stripped. The title/review wording
  itself is retained.
- Whitespace normalization.
- Removal of empty review bodies.

Final targeted-fix counts (from the source run):
- Terminal Amazon feedback suffixes removed: {stats.get('final_cleaning_fixes', {}).get('terminal_feedback_suffix_removed', 0)}
- Matching duplicated heading markers removed: {stats.get('final_cleaning_fixes', {}).get('duplicated_leading_heading_markup_removed', 0)}

Meaningful punctuation, negations, emojis and
customer-authored wording are preserved. Original source columns are copied
unchanged alongside standardized fields.

## 6. Review-length statistics

### Before cleaning

{json.dumps(stats['length_before'], indent=2)}

### After cleaning

{json.dumps(stats['length_after'], indent=2)}

## 7. Rating distributions

### Before preprocessing

{json.dumps(stats['rating_before'], indent=2)}

### After preprocessing

{json.dumps(stats['rating_after'], indent=2)}

## 8. Limitations

- Language detection is not perfect.
- Short reviews are assigned uncertain.
- Code-switching may reduce detection confidence.
- Only English, Spanish and French are explicitly supported.
- Product IDs depend on successful ASIN extraction.
- Exact duplicates are identified only within known products.
- Near-duplicates are not automatically removed.
- Review dates preserve the source representation.

## 9. Output files

- amazon_reviews_cleaned.csv
- amazon_reviews_spanish.csv
- amazon_reviews_french.csv
- amazon_reviews_uncertain_language.csv
- amazon_reviews_removed.csv
- preprocessing_stats.json

## 10. Scope restriction

This stage performs dataset preprocessing only.

No ABSA annotation, aspect extraction,
sentiment labelling or model training.
"""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        report,
        encoding="utf-8"
    )


# ---------------------------------------------------------
# 8. MAIN PREPROCESSING PIPELINE
# ---------------------------------------------------------

def run_preprocessing(input_path, output_dir, report_path):

    input_path = Path(input_path).resolve()

    output_dir = Path(output_dir).resolve()

    report_path = Path(report_path).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    # Prevent writing outputs over the original input.
    if output_dir == input_path.parent:
        raise ValueError(
            "Output directory cannot be the input directory."
        )

    # Load original dataset without modifying it.
    original_df = pd.read_csv(input_path)

    original_count = len(original_df)

    required_columns = [
        "Product URL",
        "Product Title",
        "Rating",
        "Date",
        "Verified",
        "Review Title",
        "Review Content"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in original_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Preserve original source fields.
    df = original_df.copy(deep=True)

    # Project-local source row and review ID.
    df["source_row"] = range(
        2,
        original_count + 2
    )

    df["review_id"] = df["source_row"].map(
        lambda x: f"AMZ-{x - 1:06d}"
    )

    # Extract ASIN.
    df["product_id"] = df["Product URL"].map(
        extract_asin
    )

    # Standardized project fields.
    df["product_name"] = (
        df["Product Title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["rating"] = pd.to_numeric(
        df["Rating"],
        errors="coerce"
    )

    df["review_title"] = df["Review Title"].map(clean_review_text)
    review_text_before_feedback_fix = df["Review Content"].map(
        lambda value: clean_review_text(value, remove_feedback_suffix=False)
    )
    feedback_suffix_removed = int(
        review_text_before_feedback_fix.map(has_terminal_feedback_suffix).sum()
    )
    df["review_text"] = review_text_before_feedback_fix.map(
        remove_terminal_feedback_suffix
    )

    # Remove only matching duplicated heading markup verified across the title
    # and body; do not strip arbitrary leading hashes from customer text.
    original_standardized_fields = list(
        zip(df["review_title"], df["review_text"])
    )
    cleaned_fields = [
        remove_duplicated_leading_heading(title, body)
        for title, body in original_standardized_fields
    ]
    heading_fix_ids = {
        review_id
        for review_id, before, after in zip(
            df["review_id"], original_standardized_fields, cleaned_fields
        )
        if before != after
    }
    df["review_title"] = [title for title, _ in cleaned_fields]
    df["review_text"] = [body for _, body in cleaned_fields]

    df["verified_purchase"] = (
        df["Verified"]
        .astype("string")
        .str.strip()
        .str.lower()
        .map({
            "yes": True,
            "true": True,
            "verified": True,
            "no": False,
            "false": False,
            "unverified": False
        })
        .astype("boolean")
    )

    df["review_date"] = (
        df["Date"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["source_url"] = (
        df["Product URL"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove empty review bodies.
    empty_mask = (
        df["review_text"].str.strip() == ""
    )

    empty_reviews = df[empty_mask].copy()

    empty_audit = pd.DataFrame({
        "review_id": empty_reviews["review_id"],
        "source_row": empty_reviews["source_row"],
        "product_id": empty_reviews["product_id"],
        "review_text": empty_reviews["review_text"],
        "removal_reason": "empty_review_body"
    })

    nonempty = df[~empty_mask].copy()

    # Identify exact duplicates.
    retained, duplicate_audit = identify_duplicates(
        nonempty
    )

    removed = pd.concat(
        [empty_audit, duplicate_audit],
        ignore_index=True
    )

    duplicated_heading_markup_removed = int(
        retained["review_id"].isin(heading_fix_ids).sum()
    )

    # Language detection.
    language_results = retained["review_text"].map(
        detect_review_language
    )

    retained["language"] = language_results.map(
        lambda x: x["language"]
    )

    retained["language_confidence"] = language_results.map(
        lambda x: x["confidence"]
    )

    retained["language_confidence_margin"] = (
        language_results.map(
            lambda x: x["confidence_margin"]
        )
    )

    retained["language_detection_status"] = (
        language_results.map(
            lambda x: x["detection_status"]
        )
    )

    # Preserve source fields alongside standardized fields.
    standardized_columns = [
        "review_id",
        "product_id",
        "product_name",
        "rating",
        "review_title",
        "review_text",
        "language",
        "language_confidence",
        "language_confidence_margin",
        "language_detection_status",
        "verified_purchase",
        "review_date",
        "source_url"
    ]

    source_columns = list(original_df.columns)

    retained_output = retained[
        standardized_columns + source_columns
    ].copy()

    # Split by language.
    english = retained_output[
        retained_output["language"] == "English"
    ]

    spanish = retained_output[
        retained_output["language"] == "Spanish"
    ]

    french = retained_output[
        retained_output["language"] == "French"
    ]

    uncertain = retained_output[
        retained_output["language"] == "uncertain"
    ]

    # Write outputs.
    english.to_csv(
        output_dir / "amazon_reviews_cleaned.csv",
        index=False
    )

    spanish.to_csv(
        output_dir / "amazon_reviews_spanish.csv",
        index=False
    )

    french.to_csv(
        output_dir / "amazon_reviews_french.csv",
        index=False
    )

    uncertain.to_csv(
        output_dir / "amazon_reviews_uncertain_language.csv",
        index=False
    )

    removed.to_csv(
        output_dir / "amazon_reviews_removed.csv",
        index=False
    )

    # Statistics.
    def rating_distribution(data):
        return {
            str(key): int(value)
            for key, value in (
                data.value_counts(dropna=False)
                .sort_index()
                .items()
            )
        }

    stats = {
        "original_records": original_count,

        "english_reviews": len(english),

        "spanish_reviews": len(spanish),

        "french_reviews": len(french),

        "uncertain_reviews": len(uncertain),

        "empty_reviews_removed": len(empty_reviews),

        "duplicates_removed": len(duplicate_audit),

        "retained_reviews": len(retained),

        "unique_products": int(
            retained.loc[
                retained["product_id"] != "",
                "product_id"
            ].nunique()
        ),

        "missing_asins": int(
            (retained["product_id"] == "").sum()
        ),

        "rating_before": rating_distribution(
            df["rating"]
        ),

        "rating_after": rating_distribution(
            retained["rating"]
        ),

        "length_before": length_statistics(
            df["Review Content"]
        ),

        "length_after": length_statistics(
            retained["review_text"]
        ),

        "final_cleaning_fixes": {
            "terminal_feedback_suffix_removed": feedback_suffix_removed,
            "duplicated_leading_heading_markup_removed": int(
                duplicated_heading_markup_removed
            )
        },

        "language_detection": {
            "library": "Lingua",
            "supported_languages": [
                "English",
                "Spanish",
                "French"
            ],
            "minimum_words": MIN_WORDS,
            "minimum_confidence": MIN_CONFIDENCE,
            "minimum_confidence_margin": (
                MIN_CONFIDENCE_MARGIN
            )
        }
    }

    stats_path = (
        output_dir / "preprocessing_stats.json"
    )

    stats_path.write_text(
        json.dumps(
            stats,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # Generate report.
    generate_report(
        stats,
        report_path
    )

    # Verify row accounting.
    total_accounted = (
        len(english)
        + len(spanish)
        + len(french)
        + len(uncertain)
        + len(removed)
    )

    if total_accounted != original_count:
        raise RuntimeError(
            "Row accounting failed: "
            f"{total_accounted} != {original_count}"
        )

    print("\nSTAGE 1 PREPROCESSING COMPLETED")

    print("\nDATASET STATISTICS")

    print(
        json.dumps(
            stats,
            indent=4,
            ensure_ascii=False
        )
    )

    print("\nROW ACCOUNTING")

    print(
        f"{len(english)} English + "
        f"{len(spanish)} Spanish + "
        f"{len(french)} French + "
        f"{len(uncertain)} uncertain + "
        f"{len(removed)} removed = "
        f"{original_count}"
    )

    print("\nOUTPUT DIRECTORY:")

    print(output_dir)

    print("\nREPORT:")

    print(report_path)

    return stats


# ---------------------------------------------------------
# 9. COMMAND-LINE ENTRY POINT
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Amazon ABSA Stage 1 preprocessing"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT
    )

    args = parser.parse_args()

    run_preprocessing(
        args.input,
        args.output_dir,
        args.report
    )


if __name__ == "__main__":
    main()

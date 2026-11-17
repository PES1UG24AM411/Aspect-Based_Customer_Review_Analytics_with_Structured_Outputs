# Stage 1 – Dataset Preprocessing Report

## 1. Overview

Project: Aspect-Based Customer Review Analytics
with Structured Outputs.

Preprocessing only. No aspect annotation,
sentiment labelling or model training was performed.

## 2. Dataset statistics

| Metric | Count |
|---|---:|
| Original records | 1728 |
| English reviews | 1082 |
| Spanish reviews | 328 |
| French reviews | 21 |
| Uncertain reviews | 222 |
| Empty reviews removed | 61 |
| Exact duplicates removed | 14 |
| Total retained reviews | 1653 |
| Unique products | 419 |
| Missing ASINs | 0 |

## 3. Language detection

Library: Lingua language detector.

Supported languages:
- English
- Spanish
- French

Minimum review length: 8 words.

Minimum confidence: 0.8.

Minimum confidence margin: 0.15.

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
- Terminal Amazon feedback suffixes removed: 20
- Matching duplicated heading markers removed: 196

Meaningful punctuation, negations, emojis and
customer-authored wording are preserved. Original source columns are copied
unchanged alongside standardized fields.

## 6. Review-length statistics

### Before cleaning

{
  "count": 1728,
  "mean": 157.46,
  "median": 105.5,
  "minimum": 5,
  "maximum": 2319
}

### After cleaning

{
  "count": 1653,
  "mean": 128.64,
  "median": 74.0,
  "minimum": 0,
  "maximum": 2312
}

## 7. Rating distributions

### Before preprocessing

{
  "1.0": 282,
  "2.0": 37,
  "3.0": 99,
  "4.0": 207,
  "5.0": 1103
}

### After preprocessing

{
  "1.0": 282,
  "2.0": 36,
  "3.0": 99,
  "4.0": 206,
  "5.0": 1030
}

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

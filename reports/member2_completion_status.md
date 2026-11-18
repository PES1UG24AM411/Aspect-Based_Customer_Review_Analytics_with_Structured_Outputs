# Member 2 — handover and annotation status (18 November 2026)

## Offline work completed

- Inspected the supplied repository ZIP and processed CSV.
- Added Pydantic v2 output models and semantic source/evidence validators.
- Added the versioned ABSA prompt and OpenAI Responses API structured parsing adapter.
- Added bounded semantic retries, separately bounded transient retries, failure JSONL,
  fixed-cohort resume behavior, output manifests and a CLI that is locked to a 10-row pilot.
- Added an offline preflight mode that does not create an API client or make a network call.
- Added offline extraction tests and completed the README setup/run/handover instructions.
- Corrected JSONL newline writing and ensured semantic retry exhaustion is logged with
  its actual category and attempt count.
- The approved 10-review run used `absa_extraction_v1`; its approval does not authorize
  a claim that full-dataset extraction is complete.

## Dataset and preservation checks

- Processed CSV: 1,082 rows, 23 columns; 1,082 English-labelled rows.
- Required columns: `review_id`, `product_id`, `review_text`, `language`.
- 1,082 unique non-null review IDs; no missing product IDs or review text.
- 381 unique products; 9 duplicate review-text strings; no duplicate
  `(product_id, review_text)` pairs.
- SHA-256: `1f669b4d3085cc69d525e70d53cc159cffb1c62f3fba009ec74afadaf75e380d`.
- Binary comparison against the original ZIP confirms `src/preprocessing.py`,
  `tests/test_preprocessing.py`, and `data/processed/amazon_reviews_cleaned.csv`
  are byte-for-byte unchanged.

## Offline verification results

- The approved 10-review pilot manifest records 10 successes, zero failures, and
  zero outstanding pilot IDs; the pilot was manually reviewed and approved as
  recorded in `reports/pilot_approval.json`.
- The development annotation file contains 20 records: 5 complete and 15 todo.
  `validate_dev_annotations.py` currently reports those 15 incomplete records, so
  its audit is **not passed**.
- The human evaluation file contains 300 records: 1 complete and 299 todo. No
  evaluation metrics can be claimed from this incomplete set.
- `data/evaluation/ai_annotations.jsonl` contains 63 AI-generated silver records.
  The 300-record run manifest reports 26 successes and 274 failures; the separate
  recovery manifest reports 10 recovered records. These are not human annotations
  or gold labels.
- `git diff --check` passed. The full suite passed in the repository virtual
  environment: `.venv/bin/python -m pytest -q` — **106 passed**. Bare
  `python -m pytest -q` resolved to a separate Python 3.11 installation without
  Streamlit and Lingua and failed during collection; activate `.venv` before
  using the README test command.

## Extraction and evaluation still pending

1. Complete and validate the remaining development and human evaluation
  annotations; keep evaluation IDs disjoint from development and pilot IDs.
2. Full-dataset extraction has not been run. If it is required, review the existing
  pilot's documented issues, confirm account access to `gpt-6-luna`, and follow
  the approval-gated pipeline workflow.
3. Do not calculate or report evaluation metrics until the human gold annotations
  are complete. Keep AI silver records separate from ground truth.
4. Real-model summaries/judge remain unverified in the current handover artifacts.

The pilot does have model predictions, token usage, estimated cost, and latency
recorded in its manifest; these figures apply only to that pilot. No full-dataset
extraction or human-gold evaluation metrics are claimed. Human annotation/evaluation
remains Member 3's work; frontend integration remains Member 4's work.

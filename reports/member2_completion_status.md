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
- The guide-approved target is 15 unique reviews, selected as the first 15 IDs
  from the existing 30-ID shared assignment in the seed-42 evaluation split.
  Current selected completion is 0/15 in each independent annotator workspace
  and 1/15 in the retained consolidated human-gold file. The original 300-row
  files are preserved; no evaluation metrics can be claimed yet.
- `data/evaluation/ai_annotations.jsonl` contains 63 AI-generated silver records.
  The 300-record run manifest reports 26 successes and 274 failures; the separate
  recovery manifest reports 10 recovered records. These are not human annotations
  or gold labels.
- `git diff --check` passed. The focused approval/evaluation/agreement/Streamlit/strategy
  tests passed (41 tests), and the complete suite passed in the activated
  repository environment (`python -m pytest -q`) — **112 passed**. No API calls
  were made.

## Revised human evaluation requirement

- The comparison target is 15 unique human-gold reviews shared across zero-shot,
  few-shot v2, and CoT. The Streamlit evaluation page filters each existing
  independent annotator workspace to that cohort and displays actual completion
  out of 15. Non-selected rows and the historical 300-review assignment remain
  untouched.
- Strategy comparison validates the selected gold records and source identity,
  scores all three strategies over the same cohort, reports common-label macro
  precision/recall/F1, and includes invalid/missing prediction diagnostics.
- No comparison metrics are claimed. Both independent annotation workspaces and
  the adjudicated 15-record gold set still need completion. Each prompting
  strategy also needs its own manually reviewed and approved pilot before full
  extraction.

## Extraction and evaluation still pending

1. Finish and adjudicate the same 15 IDs; preserve all other annotation records.
2. Run separate reviewed/approved strategy pilots and full extractions if API use
   is authorized. This handover did not make API calls.
3. Run the three-strategy comparison only after the 15 human-gold records and all
   predictions validate. Keep the 63 AI silver records separate from gold.
4. Real-model summaries/Judge remain unverified in the current handover artifacts.

The pilot does have model predictions, token usage, estimated cost, and latency
recorded in its manifest; these figures apply only to that pilot. No full-dataset
extraction or human-gold evaluation metrics are claimed. Human annotation/evaluation
remains Member 3's work; frontend integration remains Member 4's work.

## Preserved project deliverables

- Structured extraction, JSONL output, CSV export, aggregation, per-product
  summary code, and bounded extraction validation retries remain in place.
- `extract_json_object` can extract fenced/surrounded JSON but does not repair
  malformed JSON syntax; exhausted parse/validation failures are logged. A
  dedicated syntax-repair step remains incomplete.
- Two saved summaries are extractive; the faithfulness report has zero LLM-Judge
  claims. Real LLM summary/Judge evaluation remains pending.

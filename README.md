
# Aspect-Based Customer Review Analytics with Structured Outputs

## Project Overview

This project analyzes e-commerce customer reviews using
Aspect-Based Sentiment Analysis (ABSA).

The objective is to identify product aspects and their
associated sentiments and generate structured outputs.

## Member 1 Responsibilities

Member 1 is responsible for:

1. Dataset collection.
2. Dataset inspection.
3. Dataset preprocessing.
4. Language filtering.
5. Duplicate identification.
6. Ground-truth annotation preparation.

## Stage 0 – Dataset Inspection

Completed.

The dataset was previously inspected.

Recorded baseline:

- Original records: 1,728
- Columns: 10
- Unique ASINs: 452
- No exact duplicate full rows.

See:

`docs/amazon_dataset_inspection.md`

## Stage 1 – Dataset Preprocessing

The preprocessing pipeline uses Python, Pandas and Lingua.

### Features

- Review text cleaning.
- HTML entity decoding.
- Scraping artifact removal.
- English, Spanish and French detection.
- Uncertain-language separation.
- Empty review removal.
- Exact duplicate removal within the same ASIN.
- Project-local review IDs.
- ASIN extraction.
- Statistics generation.
- Markdown report generation.
- Duplicate-removal audit.

### Language Detection

Lingua is used for multilingual language identification.

Configured languages:

- English
- Spanish
- French

Short, ambiguous and low-confidence reviews are
assigned to the uncertain category.

The uncertain partition must be manually reviewed
before it is used for the main experiment.

### Output Files

Generated under `data/processed/`:

- `amazon_reviews_cleaned.csv`
- `amazon_reviews_spanish.csv`
- `amazon_reviews_french.csv`
- `amazon_reviews_uncertain_language.csv`
- `amazon_reviews_removed.csv`
- `preprocessing_stats.json`

Report:

`docs/preprocessing_report.md`

## Installation

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Preprocessing

```bash
python src/preprocessing.py
```

Or specify paths:

```bash
python src/preprocessing.py \
  --input data/raw/ARE_CustomerReviews_Cleaned_v2.csv \
  --output-dir data/processed \
  --report docs/preprocessing_report.md
```

## Run Tests

```bash
pytest -v
```

## Important Data Policies

1. Original CSV is never modified.
2. Original source fields are retained in processed CSVs.
3. Review wording is not paraphrased.
4. Exact duplicates are removed only within the same ASIN.
5. Identical reviews across different products are retained.
6. Reviews without ASIN are not automatically deduplicated.
7. Uncertain-language reviews are preserved separately.
8. Removed rows are documented.

## Project Scope

Stage 1 (preprocessing) never generates annotations or labels. Later stages are described below.

## Reproducibility

The preprocessing pipeline is deterministic for a fixed
input file, dependency version and configured thresholds.

Review IDs are generated from source row positions.

The statistics and report are regenerated each time
the script runs.


---

# Stages 2-7: extraction, evaluation, analysis, summaries, UI

## Current handover status (18 November 2026)

- The processed dataset contains 1,082 English reviews. The fixed 10-review pilot
  completed with 10 schema-valid records and no logged failures; it was manually
  reviewed and approved for `absa_extraction_v1`. This approval is tied to that
  prompt, model, schema, and dataset hash.
- Full-dataset extraction has **not** been run. The pilot approval does not mean
  the remaining reviews have been extracted.
- The guide-approved human evaluation target is 15 unique reviews. The stable
  cohort is the first 15 IDs in the existing shared-review assignment, which is
  part of the seed-42 evaluation split. Each annotator has an independent
  workspace for the same 15 reviews. Current completion: 0/15 in each annotator
  workspace and 1/15 in the retained consolidated gold file. The old 300-review
  split and annotation records remain intact; no final evaluation metrics exist.
- `data/evaluation/ai_annotations.jsonl` contains 63 AI-generated silver
  annotations, not human gold labels. Do not use silver records for scoring.
- Zero-shot, few-shot v2, and CoT prompts are present. Their comparison has not
  been run; see the evaluation instructions and status reports below.
- See [reports/handover_status.md](reports/handover_status.md) and
  [reports/member2_completion_status.md](reports/member2_completion_status.md)
  for current validation details and remaining work.

## Pipeline at a glance

```
raw CSV -> preprocessing -> data/processed/amazon_reviews_cleaned.csv
        -> pilot (10 reviews) -> human review -> approve -> full extraction  -> data/extracted/*.jsonl
        -> aggregate (data/analysis/*.csv) -> summaries (data/analysis/summaries.jsonl)
        -> evaluation vs human gold (reports/evaluation/*) -> Streamlit app
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put OPENAI_API_KEY in .env locally; never commit it
```
Confirm your API project can use the model in `OPENAI_MODEL` (default in `.env.example`); override it in `.env` if not.

## Stage A - pilot (10 reviews, requires API key)

```bash
python -m src.extraction.pipeline --preflight          # offline checks, no API call
python -m src.extraction.pipeline                      # fixed 10-review pilot
python -m src.extraction.pilot_report                  # writes reports/pilot_review.md
```
Outputs: `data/extracted/pilot_predictions.jsonl`, `pilot_failures.jsonl`, `reports/pilot_run_manifest.json`.
Read `reports/pilot_review.md`, tick the manual checklist (hallucinations, missed aspects, wrong sentiment/complaint),
tune the prompt on **dev** reviews only if needed (re-run the pilot), then approve:

```bash
python -m src.extraction.pipeline --approve-pilot --reviewer "Your Name" --notes "..."
```
Approval is recorded in `reports/pilot_approval.json` and is bound to model, prompt version, schema version and dataset hash;
changing any of these invalidates it.

## Stage B - full extraction (blocked until approval)

```bash
python -m src.extraction.pipeline --full --workers 4
```
Resumable: reviews with a valid record (in `predictions.jsonl` or the pilot file) are skipped. Failures go to
`data/extracted/failures.jsonl`; `reports/full_run_manifest.json` records eligible / skipped / succeeded / failed /
outstanding review IDs. Re-run the same command to retry failures. Only treat the dataset as processed when
the manifest shows `"complete": true`.

## Stage C - evaluation (requires human gold labels)

The guide-approved target is 15 unique human-gold reviews. The app selects the
first 15 IDs from the existing 30-review shared assignment, itself drawn from
the seed-42 evaluation split. Both annotators continue working independently on
that same subset; the existing 300-row workspaces and consolidated gold file are
preserved. The app shows completion progress for the selected 15 only. Resolve
disagreements and adjudicate one human gold record per selected review into
`data/evaluation/gold_annotations.jsonl` before scoring. The evaluator schema-
validates gold and checks selected IDs, product IDs, and review text against the
processed dataset.

Zero-shot, few-shot v2, and CoT runs must each use the same fixed human-gold IDs.
Each strategy has a separate pilot approval gate bound to its own prompt version.
For each value `zero_shot`, `few_shot_v2`, and `cot`, run the pilot, review it,
approve that strategy's pilot, and then run full extraction. Example for one
strategy (the pilot and full commands make API requests; the approval command is
local. Run API commands only when explicitly authorized and budgeted):

```bash
python -m src.extraction.pipeline --strategy zero_shot --out-dir data/extracted --reports-dir reports
python -m src.extraction.pipeline --strategy zero_shot --approve-pilot --reviewer "Your Name" --out-dir data/extracted --reports-dir reports
python -m src.extraction.pipeline --strategy zero_shot --full --out-dir data/extracted --reports-dir reports
```

Repeat with the other strategy names. The full runs require a separate completed
pilot approval for each strategy. No model runs are performed by the comparison
command. Once all three prediction files and all 15 adjudicated gold records exist,
run:

```bash
python -m src.evaluation.run_eval --compare-strategies --mode lenient
```

Default prediction inputs are `data/extracted/zero_shot/predictions.jsonl`,
`data/extracted/few_shot_v2/predictions.jsonl`, and
`data/extracted/cot/predictions.jsonl`; use `--zero-shot-predictions`,
`--few-shot-predictions`, or `--cot-predictions` to override paths. Results are
written under `reports/evaluation/` and use the same 15 records for every
strategy. The report includes macro precision, recall, and F1 over a common set
of normalized aspect classes, prediction coverage, invalid records, and missing
IDs. Missing or invalid per-review predictions count as an empty extraction,
not as dropped reviews. AI silver annotations are never loaded as gold.

The 15-review sample is small and supports only an exploratory comparison; it
limits precision and generalizability. Do not claim metrics until all three
experiments have actually run. The older single-prediction evaluator remains
available as `python -m src.evaluation.run_eval --predictions ...`.

## Stage D - product analysis
`python -m src.analysis.aggregate` -> `data/analysis/` (aspect frequency, sentiment distribution, complaints, recurring negative
product-aspect pairs, cross-product aspect comparison, product overview). Keyed by `product_id`; reviews de-duplicated by `review_id`;
aspect terms mapped to canonical names via `src/analysis/aspect_normalizer.py` (rule-based; extend after inspecting real output).

## Stage E - summaries and judge
`python -m src.summarization.run_summaries` writes deterministic extractive summaries (no API) with evidence citations and a
structural support check. `--llm` uses the LLM summariser plus an LLM judge that flags claims unsupported by the cited evidence.
The judge is a consistency check, **not** ground truth and not a substitute for the human evaluation in Stage C.

## Stage F - Streamlit app
`streamlit run app/streamlit_app.py`. Reads the outputs above. Without predictions it shows status messages; the sidebar
"MOCK" option previews the UI using a keyword rule set (clearly bannered, not model output).

## Other deliverables and current evidence

- Structured extraction schemas, source/evidence validation, JSONL predictions,
  and CSV export are implemented. The pipeline has bounded validation retries
  and logs exhausted failures.
- JSON output handling extracts a JSON object from fenced or surrounding text,
  but it does not repair syntactically malformed JSON. Invalid parse/validation
  responses are retried and then recorded as failures; a separate, explicit
  malformed-syntax repair feature remains incomplete.
- Product aggregation and per-product extractive/LLM summary code are present.
  The current saved summaries contain two extractive records. The faithfulness
  report says `no_llm_judge_results` (zero judged claims), so real LLM-as-Judge
  faithfulness evaluation remains pending.
- Extraction, aggregation, and summarization are available as separate stages;
  summaries build from validated extraction evidence. The LLM summarizer and
  Judge require API-backed runs and have not been verified here.

## Tests
`pytest -q` (offline; uses fake/mock clients, no API calls). Covers preprocessing, extraction, approval gate, resume, retries,
auth failure, missing key, evaluation metrics, aggregation, summaries, and every Streamlit page.

## Limitations
- Only the approved 10-review pilot has been run through the extraction pipeline; full-dataset extraction and human-gold evaluation remain incomplete.
- The 15-review human sample is not yet fully annotated/adjudicated; the three prompting strategies have not yet been compared.
- The 63 existing AI-generated silver annotations are separate from human gold and do not support accuracy/F1 claims.
- Explicit syntax repair for malformed JSON remains incomplete; bounded parse/validation retries and failure logging are not the same as repairing malformed syntax.
- LLM summary/Judge faithfulness evaluation is not complete unless saved judge verdicts exist; offline structural checks are not an LLM-as-Judge result.
- The mock extractor (`src/extraction/mock_client.py`) exists for tests/UI previews only.
- Aspect normalisation is heuristic; lenient matching is a convention, so report the mode used.
- Cost estimates in manifests use the prices in `src/extraction/config.py`; verify them.
- Non-English and uncertain-language reviews are not extracted.


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

1. `data/evaluation/splits.json` holds the deterministic dev/eval split (seed 42; eval is disjoint from dev and pilot).
2. Annotate `data/evaluation/gold_annotations_template.jsonl` per `docs/annotation_guidelines.md`, save as
   `data/evaluation/gold_annotations.jsonl`.
3. `python -m src.evaluation.run_eval [--mode strict|lenient]`

Outputs in `reports/evaluation/`: `evaluation_metrics.json`, `evaluation_report.md`, `error_analysis.csv`, confusion-matrix PNGs.
Metrics: aspect P/R/F1 (+bootstrap CI), joint aspect+sentiment, sentiment per-class/macro/weighted/accuracy/kappa and confusion
matrix, complaint P/R/F1 (aspect and review level), per-aspect scores. The command refuses to run without completed gold labels
or if gold reviews overlap the dev set.

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

## Tests
`pytest -q` (offline; uses fake/mock clients, no API calls). Covers preprocessing, extraction, approval gate, resume, retries,
auth failure, missing key, evaluation metrics, aggregation, summaries, and every Streamlit page.

## Limitations
- No LLM predictions or evaluation metrics are included in this repository until you run Stages A-C with a valid key and gold labels.
- The mock extractor (`src/extraction/mock_client.py`) exists for tests/UI previews only.
- Aspect normalisation is heuristic; lenient matching is a convention, so report the mode used.
- Cost estimates in manifests use the prices in `src/extraction/config.py`; verify them.
- Non-English and uncertain-language reviews are not extracted.

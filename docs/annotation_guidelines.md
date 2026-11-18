# Annotation guidelines (15-review human-gold evaluation set)

Purpose: build the human reference for comparing zero-shot, few-shot v2, and CoT
extraction. **Model predictions and AI-generated silver annotations are never human
ground truth.** The guide-approved target is 15 unique reviews. The app selects the
first 15 IDs in the existing `shared_review_ids` assignment; those IDs belong to the
reproducible seed-42 evaluation split and are disjoint from development/pilot IDs.
This reuses the established overlap allocation rather than inventing a new one.
The legacy 300-review split, template, workspace records, and backups are retained;
the UI focuses each independent workspace on the selected 15 IDs.

## Procedure
1. In Streamlit, select **Evaluation (15-review target)** and choose your assigned
  independent workspace (`Annotator 1` or `Annotator 2`). The two existing
  workspaces independently contain the shared review set; each must reach 15/15.
2. For each selected review, fill `aspects`, set `annotator` to your name and
  `status` to `"complete"`. Save progress as needed. Non-selected records in the
  retained 300-row files are not part of this target and must remain preserved.
3. Do not look at star ratings, AI suggestions, or model predictions while
  annotating. No review is automatically labelled by the evaluation workflow.
4. After both annotators finish, run `python -m src.evaluation.annotation_agreement`
  to compute agreement on the same 15 shared reviews. Resolve disagreements and
  adjudicate one human record per selected review into
  `data/evaluation/gold_annotations.jsonl`; do not copy model/silver labels.
5. Never tune prompts on `eval_ids`. Use only the development split for prompt
  development. Evaluate zero-shot, few-shot v2, and CoT on these same 15 IDs.
6. The comparison command validates completed gold records against `GoldRecord`
  and checks review/product identity against the processed dataset. Do not report
  metrics until all 15 gold records and all three prediction files are ready.

## What is an aspect?
An explicitly discussed product attribute (battery life, camera, screen, performance, durability, price/value,
design, ease of use, compatibility, ...) or an experience aspect (shipping/delivery, packaging, seller/customer
service, returns/refunds). Annotate only what the reviewer actually says; do not infer. Different attributes are
different aspects. One entry per distinct aspect per review (merge repeated mentions; pick the clearest evidence).

## Fields (see `GoldAspect`)
- `aspect_term`: short normalised name ("battery life").
- `sentiment`: `positive` | `negative` | `neutral` — the reviewer's stance toward *that aspect*, not the review overall.
  Mixed statements about one aspect: choose the dominant stance; if truly balanced or purely factual use `neutral`.
- `evidence`: an **exact substring** of the review (validated automatically), the shortest span supporting the label.
- `complaint_present`: `true` only if the reviewer explicitly states a problem, defect, unmet expectation, remedy
  request or bad service experience about that aspect. Negative sentiment without an explicit problem is `false`
  (e.g. "price is high but fair").

## Edge cases
- Sarcasm: label the intended stance.
- Comparative ("better than my old phone"): positive for the aspect discussed.
- Condition for another product / not about the reviewed item: skip.
- No aspects: leave `aspects` empty (valid).

## Matching used in scoring
`strict`: canonical aspect names equal (see `src/analysis/aspect_normalizer.py`). `lenient` (default): equal, or
similar term (>=0.8), or overlapping evidence span (IoU>=0.5) with related term. Both are reportable; state which.

For strategy comparison, macro precision, recall, and F1 are unweighted averages
over the shared union of normalized aspect classes in human gold and all three
prediction sets. Missing/invalid per-review predictions are counted as empty
aspect lists and reported; invalid records are not silently accepted. The fixed
15-review sample is small, so results are exploratory and should not be generalized
without further data.

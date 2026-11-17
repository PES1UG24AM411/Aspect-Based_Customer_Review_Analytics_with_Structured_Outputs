# Annotation guidelines (gold evaluation set)

Purpose: build the human reference against which extraction is scored. **Model predictions are never
ground truth.** Annotate `data/evaluation/gold_annotations_template.jsonl` (100 reviews, drawn by
`src/evaluation/sampling.py`, seed 42, stratified by star bucket, disjoint from the dev/pilot set in
`data/evaluation/splits.json`). The template contains no model output, to avoid anchoring.

## Procedure
1. Copy the template to `data/evaluation/gold_annotations.jsonl`.
2. For each review fill `aspects`, set `annotator` to your name and `status` to `"complete"`.
3. Do not look at star ratings or model output while annotating.
4. Two annotators should independently label at least 30 reviews; compute Cohen's kappa for sentiment
   (`src.evaluation.metrics.cohen_kappa`) and resolve disagreements before scoring.
5. Never tune the prompt on reviews listed in `eval_ids`. Tune only on `dev_ids`.

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

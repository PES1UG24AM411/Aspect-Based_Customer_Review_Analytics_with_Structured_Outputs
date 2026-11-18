# Project handover status — 18 November 2026

## Verified project state

- The processed dataset has 1,082 English reviews and SHA-256
	`1f669b4d3085cc69d525e70d53cc159cffb1c62f3fba009ec74afadaf75e380d`.
- The fixed 10-review extraction pilot completed: 10 successes, zero failures,
	zero outstanding pilot IDs. Its manifest is `reports/pilot_run_manifest.json`.
- The pilot outputs were manually reviewed and approved by Tasya for
	`absa_extraction_v1`; approval is recorded in `reports/pilot_approval.json`.
	The recorded review notes identify sentiment and aspect-granularity issues.
- Full-dataset extraction has not been run. The only extraction output in
	`data/extracted/` is the 10-record pilot; no full-run manifest or full-run
	predictions are present.
- The guide-approved evaluation target is 15 unique reviews: the first 15 IDs in
	the pre-existing 30-ID shared assignment, contained in the seed-42 evaluation
	split. Both annotators independently work on this same selected subset. Current
	selected completion is 0/15 for Annotator 1, 0/15 for Annotator 2, and 1/15 in
	the retained consolidated gold file. The legacy 300-row files remain intact.
- The local AI annotation JSONL has 63 records. They are AI-generated silver
	annotations, not human gold. The 300-record AI run manifest reports 26
	successes and 274 failures; a separate recovery manifest reports 10 recovered.
	Neither result is used by the human-gold evaluator.
- Zero-shot, few-shot v2, and CoT prompts and strategy comparison support are
	present. The comparison validates the same 15 completed human-gold records for
	each strategy, uses macro precision/recall/F1 over a common aspect-label set,
	and reports invalid/missing prediction coverage. No strategy comparison has
	been run and no metrics are available.
- No human-gold evaluation metrics are available. LLM summaries/judge have not
	been verified against a real-model run in the current handover artifacts.

## Other project deliverables

- Structured extraction, semantic source/evidence validation, JSONL output, CSV
	export, aggregation, per-product summary generation, and bounded parse/semantic
	retries are implemented in the existing pipeline.
- JSON extraction handles code fences and surrounding prose, but does not repair
	syntactically malformed JSON. Failures are retried and eventually logged; an
	explicit malformed-syntax repair stage remains incomplete.
- Two saved product summaries are extractive. The saved faithfulness report is
	`no_llm_judge_results` with zero judged claims; real LLM-as-Judge evaluation is
	pending. The modular extraction/aggregation/summarization chain exists, but an
	LLM summary/Judge run has not been validated.

## Remaining work

1. Finish both independent 15-review workspaces, compute agreement on those 15,
	adjudicate disagreements, and complete the same 15 records in the consolidated
	human-gold file. Existing non-target annotation records must remain preserved.
2. Run and manually approve separate 10-review pilots for zero-shot, few-shot v2,
	and CoT before full strategy extraction; each approval is bound to its prompt.
3. Run the comparison only when all 15 gold records and all three prediction files
	exist; report its small-sample limitation and never use AI silver as gold.
4. Record any later real-model LLM summary/Judge run separately from offline
	structural checks and tests.

## Validation note

`git diff --check` passed. The focused approval/evaluation/agreement/Streamlit/
strategy tests passed (41 tests), and the complete suite passed in the repository
virtual environment (`python -m pytest -q`: 112 passed). No API calls were made.
Annotation status was checked from the local JSONL files and manifests; the
selected human-gold cohort is not complete.

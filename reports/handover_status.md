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
- The fixed development set has 20 records: 5 complete and 15 todo. The current
	300-record human evaluation file has 1 complete record and 299 todo. The local
	development validator reports 15 incomplete-record errors; this is unfinished
	annotation, not a passing audit.
- The local AI annotation JSONL has 63 records. They are AI-generated silver
	annotations, not human gold. The 300-record AI run manifest reports 26
	successes and 274 failures; a separate recovery manifest reports 10 recovered.
	Neither result is a human evaluation set or a basis for accuracy/F1.
- No human-gold evaluation metrics are available. LLM summaries/judge have not
	been verified against a real-model run in the current handover artifacts.

## Remaining work

1. Complete and validate the development annotations and 300-review human gold
	 evaluation annotations; retain the dev/eval separation in
	 `data/evaluation/splits.json`.
2. If full extraction is required, review the approved pilot and its known issues,
	 confirm the API project and model access, then run and verify the full pipeline.
3. Run evaluation only after the human gold set is complete; report its scoring
	 mode and do not treat AI silver annotations as ground truth.
4. Record any later real-model summary/judge run separately from offline tests.

## Validation note

`git diff --check` passed, and the full suite passed with the repository
environment (`.venv/bin/python -m pytest -q`: 106 passed). The shell's bare
`python -m pytest -q` resolved to a separate Python 3.11 installation without
Streamlit and Lingua, so collection failed there; use the activated project
environment for the documented command. Annotation status was checked from the
local JSONL files and manifests; the development annotation validator does not
pass until the remaining 15 records are completed.

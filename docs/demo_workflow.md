# Final demonstration workflow

1. `pytest -q` - show all tests pass.
2. `python -m src.extraction.pipeline --preflight` - offline readiness.
3. `python -m src.extraction.pipeline` then `python -m src.extraction.pilot_report` - run and inspect the pilot.
4. Show `--full` refusing before approval; run `--approve-pilot`; run `--full --workers 4`; show `reports/full_run_manifest.json`.
5. `python -m src.analysis.aggregate` and `python -m src.summarization.run_summaries`.
6. Show the annotated gold file and `python -m src.evaluation.run_eval`; open `reports/evaluation/evaluation_report.md`.
7. `streamlit run app/streamlit_app.py`: walk through Overview -> Aspects & sentiment -> Complaints -> Products (compare two) -> Summaries -> Evaluation -> Pipeline status.
8. Demonstrate failure handling: remove the API key (clear message), point to `data/extracted/failures.jsonl`.

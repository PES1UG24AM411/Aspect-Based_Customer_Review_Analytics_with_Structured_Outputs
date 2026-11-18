"""Evaluate extraction predictions against completed human annotations (never against model output)."""
import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from src.extraction.config import PROJECT_ROOT, MODEL, PROMPT_VERSION
from src.extraction.json_utils import read_jsonl
from src.extraction.schemas import ExtractionRecord
from src.extraction.validators import validate_record
from .metrics import evaluate, LABELS, prf
from src.analysis.aspect_normalizer import normalize_aspect
from .sampling import select_human_evaluation_ids
from .schemas import GoldRecord

STRATEGY_PREDICTIONS = {
    "zero_shot": PROJECT_ROOT / "data/extracted/zero_shot/predictions.jsonl",
    "few_shot": PROJECT_ROOT / "data/extracted/few_shot_v2/predictions.jsonl",
    "cot": PROJECT_ROOT / "data/extracted/cot/predictions.jsonl",
}


def load_gold(path, splits_path=None):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"No gold annotation file at {path}. No valid reference set exists, so no accuracy can be reported. "
                         "Annotate data/evaluation/gold_annotations_template.jsonl following docs/annotation_guidelines.md.")
    recs = [GoldRecord.model_validate(r) for r in read_jsonl(path)]
    done = {r.review_id: r.model_dump() for r in recs if r.status == "complete"}
    if not done:
        raise SystemExit("Gold file has no completed annotations (status=complete). Nothing to evaluate against.")
    if splits_path and Path(splits_path).is_file():
        dev = set(json.loads(Path(splits_path).read_text())["dev_ids"])
        leak = dev & set(done)
        if leak:
            raise SystemExit(f"Evaluation leakage: {len(leak)} gold reviews are in the dev (prompt-tuning) set: {sorted(leak)[:5]}")
    return done, len(recs)


def load_predictions(*paths):
    out = {}
    for p in paths:
        for r in read_jsonl(p):
            try:
                out[r["review_id"]] = ExtractionRecord.model_validate(r).model_dump()
            except Exception:
                continue
    return out


def load_strategy_gold(path, expected_ids, splits_path=None, dataset_path=None):
    """Load exactly the selected, completed human-gold cohort."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"No human gold annotation file at {path}.")

    records = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            record = GoldRecord.model_validate(raw)
        except Exception as exc:
            raise SystemExit(
                f"Invalid human gold record at {path}:{line_number}: {exc}"
            ) from exc
        if record.review_id in records:
            raise SystemExit(
                f"Duplicate human gold review_id {record.review_id} at {path}:{line_number}."
            )
        records[record.review_id] = record.model_dump()

    expected_ids = list(expected_ids)
    if len(expected_ids) != len(set(expected_ids)):
        raise SystemExit("The selected human evaluation IDs contain duplicates.")
    if splits_path and Path(splits_path).is_file():
        split_data = json.loads(Path(splits_path).read_text(encoding="utf-8"))
        dev_ids = set(split_data.get("dev_ids", []))
        eval_ids = set(split_data.get("eval_ids", []))
        if set(expected_ids) - eval_ids:
            raise SystemExit("Selected human gold IDs are not all in the evaluation split.")
        leak = dev_ids.intersection(expected_ids)
        if leak:
            raise SystemExit(f"Evaluation leakage into dev IDs: {sorted(leak)}")

    incomplete = [
        review_id for review_id in expected_ids
        if review_id not in records or records[review_id]["status"] != "complete"
    ]
    if incomplete:
        raise SystemExit(
            f"Human gold is incomplete for {len(incomplete)}/{len(expected_ids)} "
            f"selected reviews: {incomplete}"
        )
    selected_gold = {review_id: records[review_id] for review_id in expected_ids}
    if dataset_path:
        dataset = pd.read_csv(
            dataset_path, dtype={"review_id": str, "product_id": str}
        )
        required = {"review_id", "product_id", "review_text", "language"}
        missing_columns = required - set(dataset.columns)
        if missing_columns:
            raise SystemExit(
                f"Dataset is missing required gold-source columns: {sorted(missing_columns)}"
            )
        dataset = dataset[dataset.review_id.isin(expected_ids)]
        if dataset.review_id.duplicated().any():
            raise SystemExit("Processed dataset has duplicate selected review IDs.")
        source = dataset.set_index("review_id").to_dict(orient="index")
        missing_source = sorted(set(expected_ids) - set(source))
        if missing_source:
            raise SystemExit(f"Selected review IDs missing from dataset: {missing_source}")
        for review_id, record in selected_gold.items():
            row = source[review_id]
            if str(row["language"]).casefold() != "english":
                raise SystemExit(f"Selected gold review {review_id} is not English.")
            if (
                record["product_id"] != str(row["product_id"])
                or record["review_text"] != str(row["review_text"])
            ):
                raise SystemExit(
                    f"Human gold source identity/text does not match dataset for {review_id}."
                )
    return selected_gold


def load_strategy_predictions(path, gold):
    """Load schema/source-valid predictions and diagnose missing/invalid rows."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"Prediction file not found: {path}")

    expected_ids = set(gold)
    candidates = {}
    invalid_records = []
    ignored_out_of_sample = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            invalid_records.append({"line": line_number, "review_id": None, "error": str(exc)})
            continue
        review_id = raw.get("review_id") if isinstance(raw, dict) else None
        if not isinstance(review_id, str):
            invalid_records.append({
                "line": line_number,
                "review_id": None,
                "error": "record must be an object with a string review_id",
            })
            continue
        if review_id not in expected_ids:
            ignored_out_of_sample += 1
            continue
        try:
            record = ExtractionRecord.model_validate(raw)
            validate_record(
                record,
                expected_review_id=review_id,
                expected_product_id=gold[review_id]["product_id"],
                expected_review_text=gold[review_id]["review_text"],
            )
        except Exception as exc:
            invalid_records.append({"line": line_number, "review_id": review_id, "error": str(exc)})
            continue
        candidates.setdefault(review_id, []).append(record.model_dump())

    duplicate_ids = sorted(review_id for review_id, rows in candidates.items() if len(rows) > 1)
    for review_id in duplicate_ids:
        invalid_records.append({
            "line": None,
            "review_id": review_id,
            "error": f"duplicate valid prediction records ({len(candidates[review_id])}); excluded as ambiguous",
        })

    predictions = {
        review_id: rows[0]
        for review_id, rows in candidates.items()
        if len(rows) == 1
    }
    missing_ids = sorted(expected_ids - set(predictions))
    # Treat a missing/invalid prediction as an empty extraction, not as a reason
    # to silently drop the gold review from the common 15-review denominator.
    for review_id in missing_ids:
        source = gold[review_id]
        predictions[review_id] = {
            "review_id": review_id,
            "product_id": source["product_id"],
            "review_text": source["review_text"],
            "aspects": [],
        }

    return predictions, {
        "expected_reviews": len(expected_ids),
        "valid_prediction_reviews": len(expected_ids) - len(missing_ids),
        "missing_or_invalid_review_ids": missing_ids,
        "invalid_record_count": len(invalid_records),
        "invalid_records": invalid_records,
        "ignored_out_of_sample_records": ignored_out_of_sample,
        "duplicate_review_ids": duplicate_ids,
    }


def compare_prompt_strategies(
    *, gold_path, splits_path, assignment_path, prediction_paths, dataset_path=None,
    mode="lenient"
):
    """Evaluate all configured prompting strategies on one selected gold cohort."""
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    assignment = json.loads(Path(assignment_path).read_text(encoding="utf-8"))
    expected_ids = select_human_evaluation_ids(splits, assignment)
    gold = load_strategy_gold(
        gold_path, expected_ids, splits_path, dataset_path=dataset_path
    )

    loaded = {}
    for strategy in ("zero_shot", "few_shot", "cot"):
        path = prediction_paths[strategy]
        predictions, diagnostics = load_strategy_predictions(path, gold)
        loaded[strategy] = (path, predictions, diagnostics)

    macro_labels = sorted({
        normalize_aspect(aspect["aspect_term"])
        for records in [gold, *(item[1] for item in loaded.values())]
        for record in records.values()
        for aspect in record["aspects"]
    })

    strategies = {}
    for strategy, (path, predictions, diagnostics) in loaded.items():
        result, errors = evaluate(gold, predictions, mode=mode)
        aspect_scores = result["per_aspect_extraction"]
        macro = {
            metric: [
                aspect_scores.get(label, prf(0, 0, 0))[metric]
                for label in macro_labels
            ]
            for metric in ("precision", "recall", "f1")
        }
        strategies[strategy] = {
            "prediction_file": str(path),
            "prediction_diagnostics": diagnostics,
            "metrics": result,
            "macro_aspect_metrics": {
                "label_count": len(macro_labels),
                "labels": macro_labels,
                "precision": round(sum(macro["precision"]) / len(macro_labels), 4)
                if macro_labels else 0.0,
                "recall": round(sum(macro["recall"]) / len(macro_labels), 4)
                if macro_labels else 0.0,
                "f1": round(sum(macro["f1"]) / len(macro_labels), 4)
                if macro_labels else 0.0,
            },
            "errors": errors,
        }

    return {
        "evaluation_sample": {
            "target_reviews": len(expected_ids),
            "completed_gold_reviews": len(gold),
            "review_ids": expected_ids,
            "selection": (
                "First 15 IDs in the existing shared-review assignment; the assignment "
                "is a subset of the existing seed-42 evaluation split."
            ),
        },
        "matching_mode": mode,
        "macro_metric_definition": (
            "Aspect-extraction macro precision, recall, and F1 are unweighted means "
            "over the common union of normalized aspect classes found in the 15 human "
            "gold records or any of the three prediction files. The label universe is "
            "shared across strategies."
        ),
        "strategies": strategies,
    }


def write_strategy_comparison(report, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "strategy_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    sample = report["evaluation_sample"]
    lines = [
        "# Prompt strategy comparison",
        "",
        f"Human-gold sample: {sample['completed_gold_reviews']}/{sample['target_reviews']} reviews.",
        f"Selection: {sample['selection']}",
        f"Matching mode: `{report['matching_mode']}`.",
        "",
        "| strategy | macro precision | macro recall | macro F1 | predictions available | missing/invalid | invalid records |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, entry in report["strategies"].items():
        metrics = entry["macro_aspect_metrics"]
        diagnostics = entry["prediction_diagnostics"]
        lines.append(
            f"| {strategy} | {metrics['precision']} "
            f"| {metrics['recall']} "
            f"| {metrics['f1']} "
            f"| {diagnostics['valid_prediction_reviews']}/{diagnostics['expected_reviews']} "
            f"| {len(diagnostics['missing_or_invalid_review_ids'])} "
            f"| {diagnostics['invalid_record_count']} |"
        )
    lines += [
        "",
        "Metrics are computed against human gold only. Missing or invalid per-review predictions are "
        "treated as empty aspect predictions and remain in the common 15-review denominator. "
        "AI-generated silver annotations are not used as gold. The 15-review sample is small; "
        "scores are exploratory and should not be generalized without uncertainty analysis.",
    ]
    (out_dir / "strategy_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_confusion(cm, labels, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = [[cm[t][p] for p in labels] for t in labels]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(labels)), [str(l) for l in labels]); ax.set_yticks(range(len(labels)), [str(l) for l in labels])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Gold"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, data[i][j], ha="center", va="center")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def write_report(result, errors, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if errors:
        keys = sorted({k for e in errors for k in e})
        with (out_dir / "error_analysis.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(errors)
    s = result["sentiment_on_matched_aspects"]
    plot_confusion(s["confusion_matrix"], LABELS, "Aspect sentiment (matched aspects)", out_dir / "sentiment_confusion.png")
    cm = result["complaint_on_matched_aspects"]["confusion_matrix_true_false"]
    plot_confusion({k == "True" and True or False: {(a == "True"): b for a, b in v.items()} for k, v in cm.items()},
                   [True, False], "Complaint (matched aspects)", out_dir / "complaint_confusion.png")
    a = result["aspect_extraction"]
    md = ["# Evaluation report", "",
          f"Model `{MODEL}`, prompt `{PROMPT_VERSION}`; matching mode `{result['mode']}`; reviews scored: {result['reviews_scored']}.",
          "Reference: human annotation (see docs/annotation_guidelines.md). Metrics are only as valid as that reference.", "",
          "## Aspect extraction", f"P={a['precision']} R={a['recall']} F1={a['f1']} (95% bootstrap CI over reviews: {a['f1_bootstrap_95ci_over_reviews']}); macro-F1 over gold aspect classes={a['macro_f1_over_gold_aspect_classes']}", "",
          "## Aspect sentiment (matched aspects)", f"accuracy={s['accuracy']} kappa={s['cohen_kappa']} macro-F1={s['macro_avg']['f1']} weighted-F1={s['weighted_avg']['f1']}", "",
          "| class | P | R | F1 | support |", "|---|---|---|---|---|"]
    md += [f"| {l} | {v['precision']} | {v['recall']} | {v['f1']} | {v['support']} |" for l, v in s["per_class"].items()]
    c = result["complaint_on_matched_aspects"]; r = result["complaint_review_level"]
    md += ["", "## Complaint detection", f"Aspect level (matched): P={c['precision']} R={c['recall']} F1={c['f1']} (n={c['n']})",
           f"Review level: P={r['precision']} R={r['recall']} F1={r['f1']}", "",
           f"## Errors\n{result['error_counts']} (details in error_analysis.csv)"]
    (out_dir / "evaluation_report.md").write_text("\n".join(md), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Compare zero-shot, few-shot v2, and CoT on the same 15 human-gold reviews.",
    )
    ap.add_argument("--gold", default=str(PROJECT_ROOT / "data/evaluation/gold_annotations.jsonl"))
    ap.add_argument("--splits", default=str(PROJECT_ROOT / "data/evaluation/splits.json"))
    ap.add_argument(
        "--assignment",
        default=str(PROJECT_ROOT / "data/evaluation/annotation_assignment.json"),
    )
    ap.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "data/processed/amazon_reviews_cleaned.csv"),
    )
    ap.add_argument("--predictions", nargs="*", default=[str(PROJECT_ROOT / "data/extracted/predictions.jsonl"),
                                                         str(PROJECT_ROOT / "data/extracted/pilot_predictions.jsonl")])
    ap.add_argument("--mode", choices=["strict", "lenient"], default="lenient")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "reports/evaluation"))
    ap.add_argument("--zero-shot-predictions", default=str(STRATEGY_PREDICTIONS["zero_shot"]))
    ap.add_argument("--few-shot-predictions", default=str(STRATEGY_PREDICTIONS["few_shot"]))
    ap.add_argument("--cot-predictions", default=str(STRATEGY_PREDICTIONS["cot"]))
    a = ap.parse_args(argv)
    if a.compare_strategies:
        report = compare_prompt_strategies(
            gold_path=a.gold,
            splits_path=a.splits,
            assignment_path=a.assignment,
            dataset_path=a.dataset,
            prediction_paths={
                "zero_shot": a.zero_shot_predictions,
                "few_shot": a.few_shot_predictions,
                "cot": a.cot_predictions,
            },
            mode=a.mode,
        )
        write_strategy_comparison(report, a.out)
        print(json.dumps({
            "target_reviews": report["evaluation_sample"]["target_reviews"],
            "strategies": {
                name: entry["macro_aspect_metrics"]["f1"]
                for name, entry in report["strategies"].items()
            },
        }, indent=2))
        return
    gold, _ = load_gold(a.gold, a.splits)
    preds = load_predictions(*a.predictions)
    result, errors = evaluate(gold, preds, mode=a.mode)
    if not result["reviews_scored"]:
        raise SystemExit("No overlap between gold reviews and predictions; nothing scored.")
    write_report(result, errors, a.out)
    print(json.dumps({k: result[k] for k in ("reviews_scored", "aspect_extraction", "error_counts")}, indent=2))


if __name__ == "__main__":
    main()

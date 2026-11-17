"""Evaluate extraction predictions against completed human annotations (never against model output)."""
import argparse
import csv
import json
from pathlib import Path

from src.extraction.config import PROJECT_ROOT, MODEL, PROMPT_VERSION
from src.extraction.json_utils import read_jsonl
from src.extraction.schemas import ExtractionRecord
from .metrics import evaluate, LABELS
from .schemas import GoldRecord


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
    ap.add_argument("--gold", default=str(PROJECT_ROOT / "data/evaluation/gold_annotations.jsonl"))
    ap.add_argument("--splits", default=str(PROJECT_ROOT / "data/evaluation/splits.json"))
    ap.add_argument("--predictions", nargs="*", default=[str(PROJECT_ROOT / "data/extracted/predictions.jsonl"),
                                                         str(PROJECT_ROOT / "data/extracted/pilot_predictions.jsonl")])
    ap.add_argument("--mode", choices=["strict", "lenient"], default="lenient")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "reports/evaluation"))
    a = ap.parse_args(argv)
    gold, _ = load_gold(a.gold, a.splits)
    preds = load_predictions(*a.predictions)
    result, errors = evaluate(gold, preds, mode=a.mode)
    if not result["reviews_scored"]:
        raise SystemExit("No overlap between gold reviews and predictions; nothing scored.")
    write_report(result, errors, a.out)
    print(json.dumps({k: result[k] for k in ("reviews_scored", "aspect_extraction", "error_counts")}, indent=2))


if __name__ == "__main__":
    main()

"""Offline faithfulness report from saved LLM judge verdicts."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.extraction.config import PROJECT_ROOT

VERDICTS = ("supported", "partially_supported", "unsupported")


def score_verdicts(verdicts):
    counts = Counter(verdicts)
    total = sum(counts.values())

    if total == 0:
        return {
            "claims_judged": 0,
            "supported": 0,
            "partially_supported": 0,
            "unsupported": 0,
            "faithfulness_percent": None,
            "strict_support_percent": None,
        }

    weighted = counts["supported"] + 0.5 * counts["partially_supported"]

    return {
        "claims_judged": total,
        "supported": counts["supported"],
        "partially_supported": counts["partially_supported"],
        "unsupported": counts["unsupported"],
        "faithfulness_percent": round(weighted / total * 100, 2),
        "strict_support_percent": round(counts["supported"] / total * 100, 2),
    }


def generate_report(input_path):
    input_path = Path(input_path)
    if not input_path.exists():
        raise SystemExit(
            f"Summary file not found: {input_path}\n"
            "Generate summaries first. Extractive summaries alone do not "
            "contain LLM judge results."
        )

    product_verdicts = {}
    products_without_judge = 0

    with input_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSON at {input_path}:{line_number}: {exc}"
                )

            product_id = record.get("product_id")
            judge = record.get("llm_judge")

            if not judge:
                products_without_judge += 1
                continue

            if not product_id or not isinstance(judge.get("verdicts"), list):
                raise SystemExit(
                    f"Invalid judge result at line {line_number}"
                )

            verdicts = []
            for item in judge["verdicts"]:
                verdict = item.get("verdict")
                if verdict not in VERDICTS:
                    raise SystemExit(
                        f"Unknown verdict {verdict!r} at line {line_number}"
                    )
                verdicts.append(verdict)

            product_verdicts.setdefault(product_id, []).extend(verdicts)

    rows = []
    all_verdicts = []

    for product_id, verdicts in sorted(product_verdicts.items()):
        metrics = score_verdicts(verdicts)
        rows.append({"product_id": product_id, **metrics})
        all_verdicts.extend(verdicts)

    overall = score_verdicts(all_verdicts)

    return {
        "status": (
            "completed" if all_verdicts else "no_llm_judge_results"
        ),
        "products_judged": len(rows),
        "products_without_judge": products_without_judge,
        "overall": overall,
        "per_product": rows,
        "scoring": (
            "Supported=1, partially_supported=0.5, unsupported=0. "
            "Faithfulness is a weighted percentage of LLM judge verdicts, "
            "not ground-truth accuracy."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build an offline faithfulness report from saved judge results."
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data/analysis/summaries.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "reports"),
    )
    args = parser.parse_args(argv)

    report = generate_report(args.input)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "faithfulness_report.json"
    csv_path = out_dir / "faithfulness_by_product.csv"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "product_id",
            "claims_judged",
            "supported",
            "partially_supported",
            "unsupported",
            "faithfulness_percent",
            "strict_support_percent",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["per_product"])

    print(f"Status: {report['status']}")
    print(f"Products with judge results: {report['products_judged']}")
    print(f"Products without judge results: {report['products_without_judge']}")

    overall = report["overall"]
    if overall["claims_judged"]:
        print(f"Claims judged: {overall['claims_judged']}")
        print(f"Faithfulness: {overall['faithfulness_percent']}%")
        print(f"Strict support: {overall['strict_support_percent']}%")
    else:
        print("No LLM judge verdicts available. No score calculated.")

    print(f"\nJSON report: {json_path}")
    print(f"CSV report:  {csv_path}")


if __name__ == "__main__":
    main()

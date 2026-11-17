"""Streamlit UI over existing pipeline outputs.  Run: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data_access import load_extraction, load_json, load_reviews, pipeline_status
from src.analysis.aggregate import aggregate
from src.evaluation.metrics import LABELS

st.set_page_config(page_title="ABSA Review Analytics", layout="wide")


@st.cache_data(show_spinner=False)
def _reviews():
    return load_reviews(ROOT)


@st.cache_data(show_spinner="Aggregating...")
def _tables(use_mock):
    reviews = _reviews()
    recs, source = load_extraction(ROOT, use_mock=use_mock, reviews=reviews)
    if not recs:
        return None, source
    return aggregate(recs, reviews), source


reviews = _reviews()
st.sidebar.title("ABSA Review Analytics")
page = st.sidebar.radio("Page", ["Overview", "Review explorer", "Aspects & sentiment", "Complaints",
                                 "Products", "Summaries", "Evaluation", "Pipeline status"])
use_mock = st.sidebar.checkbox("Use MOCK keyword extraction (UI demo only)", value=False,
                               help="Only used when no real predictions exist. Not model output.")

if reviews is None:
    st.error("Processed dataset not found: data/processed/amazon_reviews_cleaned.csv. Run `python src/preprocessing.py` first.")
    st.stop()

tables, source = _tables(use_mock)
if source == "MOCK":
    st.warning("SYNTHETIC DEMO DATA: produced by a keyword rule set, NOT by the LLM. Do not cite these numbers.")
elif tables is None:
    st.info("No extraction predictions yet. Pages that need them are limited. Run the pilot, review it, approve it, "
            "then run full extraction (see README). Tick the MOCK box in the sidebar only to preview the UI.")

eng = reviews[reviews.language.astype(str).str.casefold().eq("english")]
names = reviews.drop_duplicates("product_id").set_index("product_id").product_name.fillna("").to_dict()


def need(tbl):
    if tables is None or tables.get(tbl) is None or tables[tbl].empty:
        st.info("Not available yet: no extraction results.")
        return False
    return True


if page == "Overview":
    st.header("Project overview")
    st.write("Aspect-based sentiment analysis of Amazon reviews with schema-validated LLM extraction.")
    c = st.columns(4)
    c[0].metric("Retained reviews", len(reviews)); c[1].metric("English reviews", len(eng))
    c[2].metric("Products (English)", eng.product_id.nunique())
    c[3].metric("Extracted reviews", 0 if tables is None else len(tables["reviews"]))
    st.plotly_chart(px.bar(reviews.language.value_counts().reset_index(), x="language", y="count", title="Reviews by language"),
                    width="stretch")
    stats = load_json(ROOT / "data/processed/preprocessing_stats.json")
    if stats:
        with st.expander("Preprocessing statistics"):
            st.json(stats)

elif page == "Review explorer":
    st.header("Review-level analysis")
    if need("aspects") or tables is not None:
        rv = tables["reviews"]
        f1, f2 = st.columns(2)
        prod = f1.selectbox("Product", ["All"] + sorted(rv.product_id.unique().tolist()), format_func=lambda p: p if p == "All" else f"{p} — {names.get(p,'')[:60]}")
        sents = f2.multiselect("Sentiment", LABELS, default=LABELS)
        rid_pool = rv if prod == "All" else rv[rv.product_id == prod]
        rid = st.selectbox("Review", rid_pool.review_id.tolist()) if len(rid_pool) else None
        if rid:
            row = reviews[reviews.review_id == rid].iloc[0]
            st.caption(f"Product {row.product_id} | rating {row.rating}")
            st.write(row.review_text)
            a = tables["aspects"]; a = a[(a.review_id == rid) & a.sentiment.isin(sents)]
            st.dataframe(a[["aspect_term", "aspect", "sentiment", "complaint_present", "evidence", "complaint_evidence"]], width="stretch")
            if a.empty:
                st.caption("No aspects for the selected filters.")

elif page == "Aspects & sentiment":
    st.header("Aspect and sentiment visualisation")
    if need("sentiment_distribution"):
        sd = tables["sentiment_distribution"]
        prod = st.selectbox("Product filter", ["All"] + sorted(sd.product_id.unique().tolist()))
        g = (sd if prod == "All" else sd[sd.product_id == prod]).groupby("aspect")[["positive", "negative", "neutral"]].sum()
        g["total"] = g.sum(axis=1)
        top = st.slider("Top aspects", 3, 30, 12)
        g = g.sort_values("total", ascending=False).head(top).drop(columns="total").reset_index().melt("aspect", var_name="sentiment", value_name="n")
        st.plotly_chart(px.bar(g, x="aspect", y="n", color="sentiment", barmode="stack",
                               color_discrete_map={"positive": "#2e7d32", "negative": "#c62828", "neutral": "#9e9e9e"}), width="stretch")

elif page == "Complaints":
    st.header("Complaint analysis")
    if need("complaints"):
        c = tables["complaints"].groupby("aspect").n_complaint_reviews.sum().sort_values(ascending=False).head(15).reset_index()
        st.plotly_chart(px.bar(c, x="aspect", y="n_complaint_reviews", title="Most frequently reported complaints"), width="stretch")
        st.subheader("Products with recurring negative feedback")
        rn = tables["recurring_negative"]
        if rn.empty:
            st.caption("No product-aspect pair meets the recurrence threshold (>=3 mentions, >=50% negative).")
        else:
            rn = rn.assign(product_name=rn.product_id.map(names).str[:70])
            st.dataframe(rn, width="stretch")

elif page == "Products":
    st.header("Product search and comparison")
    if need("product_overview"):
        po = tables["product_overview"]
        q = st.text_input("Search product name or ID")
        if q:
            po = po[po.product_id.str.contains(q, case=False) | po.product_name.fillna("").str.contains(q, case=False)]
        st.dataframe(po, width="stretch")
        picks = st.multiselect("Compare products (aspect net sentiment, -1..+1)", po.product_id.tolist()[:50])
        if picks:
            comp = tables["aspect_comparison"].reindex(columns=picks)
            st.dataframe(comp.dropna(how="all"), width="stretch")
            st.plotly_chart(px.imshow(comp.dropna(how="all"), color_continuous_scale="RdYlGn", zmin=-1, zmax=1, aspect="auto"), width="stretch")
        if len(picks) == 1:
            st.write(f"Aspect frequency for {picks[0]}")
            af = tables["aspect_frequency"]; st.dataframe(af[af.product_id == picks[0]], width="stretch")

elif page == "Summaries":
    st.header("Product-level summaries")
    from app.data_access import load_json
    from src.extraction.json_utils import read_jsonl
    rows = read_jsonl(ROOT / "data/analysis/summaries.jsonl")
    if rows:
        by = {r["product_id"]: r for r in rows}
        pid = st.selectbox("Product", list(by))
        r = by[pid]; s = r["summary"]
        st.caption(f"Method: {r['method']} | structural check passed: {r['structural_check']['passed']}")
        st.write(s["overview"])
        ev = {e["id"]: e for e in r["evidence"]}
        for title, key in (("Strengths", "strengths"), ("Weaknesses", "weaknesses")):
            st.subheader(title)
            for c in s[key]:
                st.markdown(f"- {c['claim']}")
                for i in c["evidence_ids"]:
                    if i in ev: st.caption(f"  {i}: \"{ev[i]['quote']}\" (review {ev[i]['review_id']})")
        if r.get("llm_judge"):
            st.subheader("LLM judge (consistency check, not ground truth)")
            st.dataframe(pd.DataFrame(r["llm_judge"]["verdicts"]), width="stretch")
    elif tables is not None:
        from src.summarization.summarizer import extractive_summary
        pid = st.selectbox("Product", tables["product_overview"].product_id.tolist()[:50])
        s = extractive_summary(tables["aspects"], pid)
        st.caption("On-the-fly extractive summary (no stored summaries found; run `python -m src.summarization.run_summaries`).")
        st.write(s.overview)
        for t, cl in (("Strengths", s.strengths), ("Weaknesses", s.weaknesses)):
            st.subheader(t); [st.markdown(f"- {c.claim}") for c in cl] or st.caption("none")
    else:
        st.info("No extraction results yet.")



elif page == "Evaluation":
    import json
    from src.evaluation.schemas import GoldRecord

    st.header("Human annotation & evaluation")

    # Keep development and evaluation annotations in separate files.
    annotation_mode = st.radio(
        "Annotation dataset",
        ["Development (20 reviews)", "Evaluation (300 reviews)"],
        horizontal=True,
    )

    if annotation_mode == "Development (20 reviews)":
        template_path = ROOT / "data/evaluation/dev_annotations_template.jsonl"
        gold_path = ROOT / "data/evaluation/dev_annotations.jsonl"
        dataset_key = "dev"
    else:
        template_path = ROOT / "data/evaluation/gold_annotations_template.jsonl"
        gold_path = ROOT / "data/evaluation/gold_annotations.jsonl"
        dataset_key = "eval"

    if not template_path.exists():
        st.error(f"Annotation template not found: {template_path}")
        st.stop()

    source_path = gold_path if gold_path.exists() else template_path
    records = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not records:
        st.warning("No reviews available for annotation.")
        st.stop()

    completed = sum(r.get("status") == "complete" for r in records)
    st.progress(completed / len(records))
    st.caption(f"{completed} / {len(records)} reviews completed")

    index_key = f"{dataset_key}_annotation_index"
    if index_key not in st.session_state:
        st.session_state[index_key] = 0

    col_prev, col_info, col_next = st.columns([1, 2, 1])

    with col_prev:
        if st.button(
            "← Previous",
            disabled=st.session_state[index_key] == 0,
            use_container_width=True,
            key=f"{dataset_key}_previous",
        ):
            st.session_state[index_key] -= 1
            st.rerun()

    with col_info:
        st.markdown(
            f"<div style='text-align:center; padding-top:8px;'>"
            f"<b>Review {st.session_state[index_key] + 1} of {len(records)}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_next:
        if st.button(
            "Next →",
            disabled=st.session_state[index_key] >= len(records) - 1,
            use_container_width=True,
            key=f"{dataset_key}_next",
        ):
            st.session_state[index_key] += 1
            st.rerun()
    record = records[st.session_state[index_key]]

    st.subheader(f"Review {record['review_id']}")
    st.caption(f"Product ID: {record['product_id']}")
    st.info(record["review_text"])

    annotator = st.text_input(
        "Your annotator name",
        value=record.get("annotator") or "",
        key=f"{dataset_key}_annotator_{record['review_id']}",
    )

    st.markdown("### Aspects in this review")
    st.caption(
        "Add one entry per distinct aspect. Evidence must be copied exactly from the review."
    )

    aspects = record.get("aspects", [])
        # AI-assisted suggestions from existing predictions (development only).
    if dataset_key == "dev":
        prediction_path = ROOT / "data/extracted/pilot_predictions.jsonl"
        predictions = {}

        if prediction_path.exists():
            with prediction_path.open(encoding="utf-8") as f:
                predictions = {
                    row["review_id"]: row
                    for line in f if line.strip()
                    for row in [json.loads(line)]
                }

        prediction = predictions.get(record["review_id"])

        if prediction and prediction.get("extraction_status") == "success":
            st.info("AI suggestions are available from existing predictions.")

            if st.button(
                "✨ Use AI suggestions (replace current draft)",
                key=f"ai_suggest_{record['review_id']}",
                use_container_width=True,
            ):
                suggested = []

                for item in prediction.get("aspects", []):
                    evidence = item.get("evidence", "")

                    if (
                        evidence in record["review_text"]
                        and item.get("sentiment") in ["positive", "negative", "neutral"]
                        and item.get("aspect_term")
                    ):
                        suggested.append({
                            "aspect_term": item["aspect_term"],
                            "sentiment": item["sentiment"],
                            "evidence": evidence,
                            "complaint_present": bool(
                                item.get("complaint_present", False)
                            ),
                        })

                if suggested:
                    record["aspects"] = suggested
                    records[st.session_state[index_key]] = record

                    gold_path.write_text(
                        "\n".join(
                            json.dumps(r, ensure_ascii=False) for r in records
                        ) + "\n",
                        encoding="utf-8",
                    )

                    # Clear stale widget values so the new suggestions display.
                    for key in list(st.session_state.keys()):
                        if key.startswith((
                            f"{dataset_key}_term_{record['review_id']}_",
                            f"{dataset_key}_sentiment_{record['review_id']}_",
                            f"{dataset_key}_evidence_{record['review_id']}_",
                            f"{dataset_key}_complaint_{record['review_id']}_",
                        )):
                            del st.session_state[key]
                else:
                    st.warning("No valid AI suggestions found.")

    for i, aspect in enumerate(aspects):
        aspect_key = (
            f"{dataset_key}_{record['review_id']}_"
            f"{aspect.get('aspect_term', '')}_"
            f"{aspect.get('evidence', '')}"
        )
        with st.expander(
            f"Aspect {i + 1}: {aspect.get('aspect_term', '')}",
            expanded=True,
        ):
            aspect["aspect_term"] = st.text_input(
                "Aspect term",
                value=aspect.get("aspect_term", ""),
                key=f"{aspect_key}_term",
            )
            aspect["sentiment"] = st.selectbox(
                "Sentiment",
                ["positive", "negative", "neutral"],
                index=["positive", "negative", "neutral"].index(
                    aspect.get("sentiment", "neutral")
                ),
                key=f"{aspect_key}_sentiment",
            )
            aspect["evidence"] = st.text_area(
                "Exact evidence from review",
                value=aspect.get("evidence", ""),
                key=f"{aspect_key}_evidence",
            )
            aspect["complaint_present"] = st.checkbox(
                "Explicit complaint/problem present",
                value=aspect.get("complaint_present", False),
                key=f"{aspect_key}_complaint",
            )

            if st.button(
                "Remove aspect",
                key=f"{dataset_key}_remove_{record['review_id']}_{i}",
            ):
                aspects.pop(i)
                record["aspects"] = aspects
                records[st.session_state[index_key]] = record

                gold_path.write_text(
                    "\n".join(
                        json.dumps(r, ensure_ascii=False) for r in records
                    ) + "\n",
                    encoding="utf-8",
                )

                # Clear stale values after removing an aspect.
                prefixes = (
                    f"{dataset_key}_term_{record['review_id']}_",
                    f"{dataset_key}_sentiment_{record['review_id']}_",
                    f"{dataset_key}_evidence_{record['review_id']}_",
                    f"{dataset_key}_complaint_{record['review_id']}_",
                )

                for key in list(st.session_state.keys()):
                    if key.startswith(prefixes):
                        del st.session_state[key]

                st.rerun()

    record["aspects"] = aspects
    record["annotator"] = annotator

    record["notes"] = st.text_area(
        "Annotation notes (optional)",
        value=record.get("notes") or "",
        key=f"{dataset_key}_notes_{record['review_id']}",
    )

    def save_annotations():
        records[st.session_state[index_key]] = record
        gold_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Add aspect", key=f"{dataset_key}_add"):
            aspects.append({
                "aspect_term": "",
                "sentiment": "neutral",
                "evidence": "",
                "complaint_present": False,
            })
            record["aspects"] = aspects
            save_annotations()
            st.rerun()

    with col2:
        if st.button("Save progress", type="primary", key=f"{dataset_key}_save"):
            record["status"] = "todo"
            save_annotations()
            st.success("Progress saved.")

    with col3:
        if st.button("Mark complete", key=f"{dataset_key}_complete"):
            try:
                record["status"] = "complete"
                GoldRecord.model_validate(record)
                save_annotations()
                st.success("Annotation completed and saved.")
                st.rerun()
            except Exception as exc:
                st.error(f"Cannot complete annotation: {exc}")

    st.caption(f"Saved locally in {gold_path.relative_to(ROOT)}.")

    if completed == len(records):
        st.success("All reviews in this dataset have been annotated.")

    st.divider()

    # Final evaluation metrics are shown only in the evaluation view.
    if dataset_key == "eval":
        m = load_json(ROOT / "reports/evaluation/evaluation_metrics.json")
        if m:
            st.subheader("Evaluation results")
            a = m["aspect_extraction"]
            s = m["sentiment_on_matched_aspects"]
            st.metric("Aspect extraction macro F1", a["macro_f1"])
            st.metric("Sentiment macro F1", s["macro_f1"])
        else:
            st.info(
                "Evaluation metrics will appear here after completed annotations and evaluation."
            )



else:
    st.header("Pipeline status")
    s = pipeline_status(ROOT)
    st.json(s)
    if not s["api_key_configured"]:
        st.warning("OPENAI_API_KEY is not configured: extraction cannot run (existing outputs can still be browsed).")
    if s["pilot_predictions"] and not s["pilot_approved"]:
        st.info("Pilot predictions exist but are not yet approved. Review reports/pilot_review.md.")
    if s["logged_failures"]:
        st.warning(f"{s['logged_failures']} extraction failure(s) logged in data/extracted/*failures.jsonl")

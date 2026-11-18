"""ABSA evaluation against a human-annotated reference. Pure Python, deterministic."""
import random
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from src.analysis.aspect_normalizer import normalize_aspect

LABELS = ["positive", "negative", "neutral"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def term_similarity(a, b):
    a, b = normalize_aspect(a), normalize_aspect(b)
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def span_iou(text, ev_a, ev_b):
    ia, ib = text.find(ev_a), text.find(ev_b)
    if ia < 0 or ib < 0:
        return 0.0
    a0, a1, b0, b1 = ia, ia + len(ev_a), ib, ib + len(ev_b)
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union else 0.0


def is_match(text, gold, pred, mode):
    sim = term_similarity(gold["aspect_term"], pred["aspect_term"])
    if sim == 1.0:
        return True
    if mode == "strict":
        return False
    iou = span_iou(text, gold["evidence"], pred["evidence"])
    return sim >= 0.8 or (iou >= 0.5 and sim >= 0.4)


def match_review(text, gold_aspects, pred_aspects, mode="lenient"):
    """Greedy one-to-one matching by (term similarity, span IoU). Returns pairs, FP list, FN list."""
    cands = []
    for gi, g in enumerate(gold_aspects):
        for pi, p in enumerate(pred_aspects):
            if is_match(text, g, p, mode):
                cands.append((-term_similarity(g["aspect_term"], p["aspect_term"]),
                              -span_iou(text, g["evidence"], p["evidence"]), gi, pi))
    cands.sort()
    ug, up, pairs = set(), set(), []
    for _, _, gi, pi in cands:
        if gi in ug or pi in up:
            continue
        ug.add(gi); up.add(pi); pairs.append((gi, pi))
    fp = [pi for pi in range(len(pred_aspects)) if pi not in up]
    fn = [gi for gi in range(len(gold_aspects)) if gi not in ug]
    return pairs, fp, fn


def confusion(y_true, y_pred, labels):
    m = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        m[t][p] += 1
    return m


def classification_report(y_true, y_pred, labels):
    cm = confusion(y_true, y_pred, labels)
    per, n = {}, len(y_true)
    for l in labels:
        tp = cm[l][l]
        fp = sum(cm[t][l] for t in labels if t != l)
        fn = sum(cm[l][p] for p in labels if p != l)
        per[l] = {**prf(tp, fp, fn), "support": tp + fn}
    sup = sum(v["support"] for v in per.values())
    macro = {k: round(sum(per[l][k] for l in labels) / len(labels), 4) for k in ("precision", "recall", "f1")}
    weighted = {k: round(sum(per[l][k] * per[l]["support"] for l in labels) / sup, 4) if sup else 0.0
                for k in ("precision", "recall", "f1")}
    acc = round(sum(cm[l][l] for l in labels) / n, 4) if n else 0.0
    return {"labels": labels, "confusion_matrix": cm, "per_class": per, "macro_avg": macro,
            "weighted_avg": weighted, "accuracy": acc, "n": n, "cohen_kappa": cohen_kappa(y_true, y_pred, labels)}


def cohen_kappa(a, b, labels):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def evaluate(gold_records, pred_records, *, mode="lenient"):
    """gold_records/pred_records: dicts keyed by review_id -> record dict (schema dumps).

    Only reviews present in BOTH are scored; coverage is reported.
    """
    ids = sorted(set(gold_records) & set(pred_records))
    tp = fp = fn = 0
    sent_t, sent_p, cmp_t, cmp_p = [], [], [], []
    joint_tp = 0
    per_aspect = defaultdict(lambda: [0, 0, 0])
    errors = []
    rev_gold_cmp, rev_pred_cmp = [], []
    per_review = []
    for rid in ids:
        g, p = gold_records[rid], pred_records[rid]
        text = g["review_text"]
        pairs, fpi, fni = match_review(text, g["aspects"], p["aspects"], mode)
        tp += len(pairs); fp += len(fpi); fn += len(fni)
        per_review.append((len(pairs), len(fpi), len(fni)))
        for gi, pi in pairs:
            ga, pa = g["aspects"][gi], p["aspects"][pi]
            c = normalize_aspect(ga["aspect_term"])
            per_aspect[c][0] += 1
            sent_t.append(ga["sentiment"]); sent_p.append(pa["sentiment"])
            cmp_t.append(bool(ga["complaint_present"])); cmp_p.append(bool(pa["complaint_present"]))
            if ga["sentiment"] == pa["sentiment"]:
                joint_tp += 1
            else:
                errors.append({"review_id": rid, "type": "sentiment_error", "gold_aspect": ga["aspect_term"],
                               "pred_aspect": pa["aspect_term"], "gold": ga["sentiment"], "pred": pa["sentiment"],
                               "gold_evidence": ga["evidence"], "pred_evidence": pa["evidence"]})
            if bool(ga["complaint_present"]) != bool(pa["complaint_present"]):
                errors.append({"review_id": rid, "type": "complaint_error", "gold_aspect": ga["aspect_term"],
                               "pred_aspect": pa["aspect_term"], "gold": ga["complaint_present"],
                               "pred": pa["complaint_present"], "gold_evidence": ga["evidence"],
                               "pred_evidence": pa["evidence"]})
        for pi in fpi:
            pa = p["aspects"][pi]
            per_aspect[normalize_aspect(pa["aspect_term"])][1] += 1
            errors.append({"review_id": rid, "type": "false_positive_aspect", "pred_aspect": pa["aspect_term"],
                           "pred": pa["sentiment"], "pred_evidence": pa["evidence"]})
        for gi in fni:
            ga = g["aspects"][gi]
            per_aspect[normalize_aspect(ga["aspect_term"])][2] += 1
            errors.append({"review_id": rid, "type": "missed_aspect", "gold_aspect": ga["aspect_term"],
                           "gold": ga["sentiment"], "gold_evidence": ga["evidence"]})
        rev_gold_cmp.append(any(a["complaint_present"] for a in g["aspects"]))
        rev_pred_cmp.append(any(a["complaint_present"] for a in p["aspects"]))
    rng = random.Random(42)
    boots = []
    for _ in range(1000 if per_review else 0):
        s = [per_review[rng.randrange(len(per_review))] for _ in per_review]
        boots.append(prf(sum(x[0] for x in s), sum(x[1] for x in s), sum(x[2] for x in s))["f1"])
    boots.sort()
    ci = [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]] if boots else None
    cmp_cm = confusion(cmp_t, cmp_p, [True, False])
    result = {
        "mode": mode, "reviews_scored": len(ids),
        "coverage": {"gold_reviews": len(gold_records), "predicted_reviews": len(pred_records),
                     "missing_predictions": sorted(set(gold_records) - set(pred_records))},
        "aspect_extraction": {**prf(tp, fp, fn), "f1_bootstrap_95ci_over_reviews": ci},
        "aspect_plus_sentiment_joint": prf(joint_tp, fp + (tp - joint_tp), fn + (tp - joint_tp)),
        "sentiment_on_matched_aspects": classification_report(sent_t, sent_p, LABELS),
        "complaint_on_matched_aspects": {"confusion_matrix_true_false": {str(k): {str(a): b for a, b in v.items()} for k, v in cmp_cm.items()},
                                         **prf(sum(t and p for t, p in zip(cmp_t, cmp_p)),
                                               sum((not t) and p for t, p in zip(cmp_t, cmp_p)),
                                               sum(t and (not p) for t, p in zip(cmp_t, cmp_p))),
                                         "n": len(cmp_t)},
        "complaint_review_level": prf(sum(t and p for t, p in zip(rev_gold_cmp, rev_pred_cmp)),
                                       sum((not t) and p for t, p in zip(rev_gold_cmp, rev_pred_cmp)),
                                       sum(t and (not p) for t, p in zip(rev_gold_cmp, rev_pred_cmp))),
        "per_aspect_extraction": {k: prf(*v) for k, v in sorted(per_aspect.items(), key=lambda kv: -sum(kv[1]))},
        "error_counts": dict(Counter(e["type"] for e in errors)),
    }
    gold_aspect_classes = {
        normalize_aspect(aspect["aspect_term"])
        for record in gold_records.values()
        for aspect in record["aspects"]
    }
    macro = {
        metric: [
            result["per_aspect_extraction"].get(
                aspect_class, prf(0, 0, 0)
            )[metric]
            for aspect_class in sorted(gold_aspect_classes)
        ]
        for metric in ("precision", "recall", "f1")
    }
    result["aspect_extraction"]["macro_precision_over_gold_aspect_classes"] = (
        round(sum(macro["precision"]) / len(macro["precision"]), 4)
        if macro["precision"] else 0.0
    )
    result["aspect_extraction"]["macro_recall_over_gold_aspect_classes"] = (
        round(sum(macro["recall"]) / len(macro["recall"]), 4)
        if macro["recall"] else 0.0
    )
    result["aspect_extraction"]["macro_f1_over_gold_aspect_classes"] = (
        round(sum(macro["f1"]) / len(macro["f1"]), 4)
        if macro["f1"] else 0.0
    )
    return result, errors

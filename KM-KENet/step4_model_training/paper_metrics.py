"""Evaluation measures reported in the paper (Table 3): Accuracy, Precision,
Recall, Specificity, F1-score, and the area under the ROC curve (AUC).

They are computed one-vs-rest for every class and macro-averaged over the
*positive* classes (entity types for NER, relation types for RE), so the large
"O" / NO_RELATION class does not dominate the averages:

* Accuracy    = (TP + TN) / (TP + TN + FP + FN)   (fraction of correctly classified items)
* Precision   = TP / (TP + FP)
* Recall      = TP / (TP + FN)
* Specificity = TN / (TN + FP)
* F1-score    = 2 * Precision * Recall / (Precision + Recall)
* AUC         = area under the one-vs-rest ROC curve of the predicted probabilities

NER is evaluated at word level on IO entity types (B-X and I-X are merged into X);
the seqeval span-level scores are reported next to these in `test_metrics.json`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_measures(
    gold: Sequence[int],
    predicted: Sequence[int],
    probabilities: np.ndarray | None,
    class_names: Sequence[str],
    negative_class: int,
) -> dict[str, Any]:
    """Accuracy, macro P/R/Specificity/F1 and macro AUC over the positive classes."""
    gold_arr = np.asarray(gold, dtype=np.int64)
    pred_arr = np.asarray(predicted, dtype=np.int64)
    n = int(gold_arr.size)
    result: dict[str, Any] = {"items": n, "per_class": {}}
    if n == 0:
        result.update(accuracy=0.0, precision=0.0, recall=0.0, specificity=0.0, f1=0.0, auc=None)
        return result

    result["accuracy"] = float((gold_arr == pred_arr).mean())
    positive_classes = [index for index in range(len(class_names)) if index != negative_class]
    aucs: list[float] = []
    for index in positive_classes:
        g = gold_arr == index
        p = pred_arr == index
        tp = int((g & p).sum())
        fp = int((~g & p).sum())
        fn = int((g & ~p).sum())
        tn = int((~g & ~p).sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        auc = None
        if probabilities is not None and 0 < g.sum() < n:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(g.astype(int), probabilities[:, index]))
            aucs.append(auc)
        result["per_class"][class_names[index]] = {
            "support": int(g.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": _safe_div(tp + tn, n),
            "precision": precision,
            "recall": recall,
            "specificity": _safe_div(tn, tn + fp),
            "f1": _safe_div(2 * precision * recall, precision + recall),
            "auc": auc,
        }

    present = [c for c in result["per_class"].values() if c["support"] > 0]
    for key in ("precision", "recall", "specificity", "f1"):
        result[key] = float(np.mean([c[key] for c in present])) if present else 0.0
    result["auc"] = float(np.mean(aucs)) if aucs else None
    return result


def bio_to_io_probabilities(
    bio_probabilities: np.ndarray, bio_labels: Sequence[str], io_classes: Sequence[str]
) -> np.ndarray:
    """Merge B-X / I-X probability columns into one column per IO class."""
    index = {name: i for i, name in enumerate(io_classes)}
    merged = np.zeros((bio_probabilities.shape[0], len(io_classes)), dtype=np.float64)
    for column, label in enumerate(bio_labels):
        entity_type = label.split("-", 1)[1] if "-" in label else label
        merged[:, index[entity_type]] += bio_probabilities[:, column]
    return merged


def summarize(ner: dict[str, Any], relations: dict[str, Any]) -> dict[str, Any]:
    """Overall KM-KENet score: mean of the NER and RE macro measures (RE only if it has gold relations)."""
    parts = [ner] + ([relations] if relations.get("items") and relations.get("per_class") and any(
        c["support"] for c in relations["per_class"].values()) else [])
    summary: dict[str, Any] = {}
    for key in ("accuracy", "precision", "recall", "specificity", "f1", "auc"):
        values = [part[key] for part in parts if part.get(key) is not None]
        summary[key] = float(np.mean(values)) if values else None
    summary["components"] = ["ner"] + (["relations"] if len(parts) > 1 else [])
    return summary


def save_roc_curves(
    gold: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    negative_class: int,
    path: Path,
    title: str,
) -> bool:
    """One-vs-rest ROC curve per positive class. Returns False if nothing could be drawn."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_auc_score, roc_curve
    except ImportError:
        logger.warning("matplotlib / scikit-learn missing; ROC curves not saved")
        return False

    gold_arr = np.asarray(gold)
    figure, axis = plt.subplots(figsize=(6, 5))
    drawn = 0
    for index, name in enumerate(class_names):
        if index == negative_class:
            continue
        g = (gold_arr == index).astype(int)
        if g.sum() == 0 or g.sum() == len(g):
            continue
        fpr, tpr, _ = roc_curve(g, probabilities[:, index])
        axis.plot(fpr, tpr, linewidth=1.5, label=f"{name} (AUC = {roc_auc_score(g, probabilities[:, index]):.4f})")
        drawn += 1
    if not drawn:
        plt.close(figure)
        return False
    axis.plot([0, 1], [0, 1], "k--", linewidth=1)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title(title)
    axis.legend(loc="lower right", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return True

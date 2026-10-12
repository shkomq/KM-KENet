"""Evaluation metrics: entity-level NER (seqeval) and micro relation F1."""

from __future__ import annotations

from typing import Any, Sequence

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from common.schema import repair_bio


def ner_metrics(gold: Sequence[Sequence[str]], predicted: Sequence[Sequence[str]]) -> dict[str, Any]:
    predicted = [repair_bio(sequence) for sequence in predicted]
    gold = [list(sequence) for sequence in gold]
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "per_type": {}}
    report = classification_report(gold, predicted, output_dict=True, zero_division=0)
    per_type = {
        name: {key: float(value) for key, value in scores.items()}
        for name, scores in report.items()
        if not name.endswith("avg")
    }
    return {
        "precision": float(precision_score(gold, predicted, zero_division=0)),
        "recall": float(recall_score(gold, predicted, zero_division=0)),
        "f1": float(f1_score(gold, predicted, zero_division=0)),
        "per_type": per_type,
    }


def relation_metrics(
    gold: Sequence[int], predicted: Sequence[int], no_relation_id: int, id2label: Sequence[str]
) -> dict[str, Any]:
    """Micro P/R/F1 over positive relation labels (NO_RELATION excluded)."""
    true_positive = sum(1 for g, p in zip(gold, predicted) if g == p and p != no_relation_id)
    predicted_positive = sum(1 for p in predicted if p != no_relation_id)
    gold_positive = sum(1 for g in gold if g != no_relation_id)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / gold_positive if gold_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_type: dict[str, dict[str, float]] = {}
    for label_id, name in enumerate(id2label):
        if label_id == no_relation_id:
            continue
        tp = sum(1 for g, p in zip(gold, predicted) if g == p == label_id)
        pp = sum(1 for p in predicted if p == label_id)
        gp = sum(1 for g in gold if g == label_id)
        p_ = tp / pp if pp else 0.0
        r_ = tp / gp if gp else 0.0
        per_type[name] = {
            "precision": p_,
            "recall": r_,
            "f1": 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0,
            "support": float(gp),
        }
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gold_relations": gold_positive,
        "per_type": per_type,
    }

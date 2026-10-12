"""Result files for an evaluated model: metrics JSON, a paper-style table and ROC curves."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.io_utils import atomic_open, write_json
from step4_model_training.paper_metrics import save_roc_curves

logger = logging.getLogger(__name__)

MEASURES = ("accuracy", "precision", "recall", "specificity", "f1", "auc")
HEADERS = ("Accuracy", "Precision", "Recall", "Specificity", "F1-score", "AUC")


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def results_table(paper: Mapping[str, Any]) -> str:
    """Markdown table: one row per component (NER, RE, overall)."""
    lines = ["| Component | " + " | ".join(HEADERS) + " |", "| --- |" + " --- |" * len(HEADERS)]
    for key, name in (("ner", "NER (word level)"), ("relations", "Relation extraction"), ("overall", "KM-KENet overall")):
        part = paper.get(key) or {}
        lines.append(f"| {name} | " + " | ".join(_fmt(part.get(m)) for m in MEASURES) + " |")
    return "\n".join(lines) + "\n"


def write_report(
    metrics: Mapping[str, Any],
    last_predictions: Mapping[str, Any],
    label_names: Mapping[str, Sequence[str]],
    output_dir: Path,
    prefix: str = "test",
    roc_curves: bool = True,
) -> None:
    """Write `<prefix>_metrics.json`, `<prefix>_results.md` and ROC curve images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{prefix}_metrics.json", metrics)
    paper = metrics.get("paper_metrics", {})
    with atomic_open(output_dir / f"{prefix}_results.md") as handle:
        handle.write(f"# KM-KENet — {prefix} results\n\n")
        handle.write("Measures as defined in Table 3 of the paper (macro-averaged over the positive classes).\n\n")
        handle.write(results_table(paper))
        ner = metrics.get("ner", {})
        rel = metrics.get("relations", {})
        handle.write(
            "\nSpan-level (seqeval) NER: P={:.4f} R={:.4f} F1={:.4f} · Relation micro P={:.4f} R={:.4f} F1={:.4f}\n".format(
                ner.get("precision", 0), ner.get("recall", 0), ner.get("f1", 0),
                rel.get("precision", 0), rel.get("recall", 0), rel.get("f1", 0),
            )
        )
    if roc_curves:
        for key, negative, title in (("ner", 0, "KM-KENet NER — ROC (one-vs-rest)"),
                                     ("relations", 0, "KM-KENet relation extraction — ROC (one-vs-rest)")):
            if key not in last_predictions:
                continue
            gold, probabilities = last_predictions[key]
            if len(gold) and save_roc_curves(gold, probabilities, label_names[key], negative,
                                             output_dir / f"{prefix}_roc_{key}.png", title):
                logger.info("ROC curves saved to %s", output_dir / f"{prefix}_roc_{key}.png")
    logger.info("\n%s", results_table(paper))

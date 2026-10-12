"""k-fold cross-validation of KM-KENet (Table 4 of the paper).

All weakly labelled sentences (train + dev + test) are pooled and split into k
folds *by source document group*, so sentences of one book never appear in both
the training and the evaluation fold. For every fold a fresh model is trained
on the other k-1 folds (10 % of their groups are held out for early stopping)
and evaluated on the held-out fold. The script reports accuracy, precision,
recall, specificity and F1-score per fold and their mean ± standard deviation.

    python -m step4_model_training.cross_validation --config configs/config.yaml
    python -m step4_model_training.cross_validation --override evaluation.cross_validation.folds=5
"""

from __future__ import annotations

import dataclasses
import random
import sys
from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from common.cli import build_arg_parser, init_step
from common.io_utils import atomic_open, write_json
from step4_model_training.main import TrainingSetup, load_records
from step4_model_training.reporting import write_report

STEP_NAME = "step4_cross_validation"
MEASURES = ("accuracy", "precision", "recall", "specificity", "f1")
HEADERS = ("Accuracy", "Precision", "Recall", "Specificity", "F1-Score")


def group_folds(records: Sequence[dict[str, Any]], k: int, rng: random.Random) -> list[list[dict[str, Any]]]:
    """Assign whole source groups to k folds, balancing the number of sentences."""
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record.get("group") or record["doc_id"]].append(record)
    if len(by_group) < k:
        # Too few groups: fall back to a sentence-level split.
        shuffled = list(records)
        rng.shuffle(shuffled)
        return [shuffled[i::k] for i in range(k)]
    groups = sorted(by_group)
    rng.shuffle(groups)
    groups.sort(key=lambda g: len(by_group[g]), reverse=True)  # largest first, greedy balancing
    folds: list[list[dict[str, Any]]] = [[] for _ in range(k)]
    for group in groups:
        smallest = min(range(k), key=lambda i: len(folds[i]))
        folds[smallest].extend(by_group[group])
    return folds


def split_dev(records: Sequence[dict[str, Any]], share: float, rng: random.Random) -> tuple[list, list]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record.get("group") or record["doc_id"]].append(record)
    groups = sorted(by_group)
    rng.shuffle(groups)
    target = share * len(records)
    dev: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for group in groups:
        (dev if len(dev) < target and len(groups) > 1 else train).extend(by_group[group])
    if not train:
        train, dev = dev, []
    return train, dev


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("k-fold cross-validation of KM-KENet (paper Table 4)")
    args = parser.parse_args(argv)
    cfg, logger = init_step(args, STEP_NAME)
    cv_cfg = (cfg.raw.get("evaluation", {}) or {}).get("cross_validation", {}) or {}
    k = int(cv_cfg.get("folds", 5))
    if k < 2:
        raise ValueError("evaluation.cross_validation.folds must be >= 2")
    rng = random.Random(cfg.seed)

    setup = TrainingSetup.from_config(cfg)
    train_cfg = setup.train_cfg
    if cv_cfg.get("epochs"):
        train_cfg = dataclasses.replace(train_cfg, epochs=int(cv_cfg["epochs"]))

    weak_dir = cfg.path("weak_dir")
    records = [r for split in ("train", "dev", "test") for r in load_records(weak_dir / f"{split}.jsonl", None)]
    if cv_cfg.get("max_examples"):
        rng.shuffle(records)
        records = records[: int(cv_cfg["max_examples"])]
    if len(records) < k:
        logger.error("Not enough weakly labelled sentences (%d) for %d folds; run step 3 first", len(records), k)
        return 1
    folds = group_folds(records, k, rng)
    logger.info("Cross-validation: %d sentences in %d folds %s", len(records), k, [len(f) for f in folds])

    out_dir = cfg.path("model_dir") / "cross_validation"
    rows: list[dict[str, Any]] = []
    for index in range(k):
        held_out = folds[index]
        rest = [r for i, fold in enumerate(folds) if i != index for r in fold]
        train, dev = split_dev(rest, 0.1, rng)
        fold_dir = out_dir / f"fold_{index + 1}"
        logger.info("Fold %d/%d: train=%d dev=%d eval=%d", index + 1, k, len(train), len(dev), len(held_out))
        trainer = setup.train(train, dev, fold_dir, logger, train_cfg)
        metrics = trainer.evaluate(setup.loader(setup.dataset(held_out, training=False), False))
        write_report(metrics, trainer.last_predictions, setup.label_names(), fold_dir, "fold", roc_curves=False)
        overall = metrics["paper_metrics"]["overall"]
        rows.append({"fold": index + 1, **{m: overall[m] for m in MEASURES}, "auc": overall.get("auc")})
        logger.info("Fold %d: %s", index + 1, {m: round(overall[m], 4) for m in MEASURES})

    mean = {m: float(np.mean([r[m] for r in rows])) for m in MEASURES}
    std = {m: float(np.std([r[m] for r in rows], ddof=1)) for m in MEASURES}
    write_json(out_dir / "cross_validation.json", {"folds": rows, "mean": mean, "std": std, "k": k})
    with atomic_open(out_dir / "cross_validation.md") as handle:
        handle.write("# Five-fold cross-validation of KM-KENet\n\n" if k == 5 else f"# {k}-fold cross-validation\n\n")
        handle.write("| Fold | " + " | ".join(HEADERS) + " |\n| --- |" + " --- |" * len(HEADERS) + "\n")
        for row in rows:
            handle.write(f"| Fold {row['fold']} | " + " | ".join(f"{row[m]:.4f}" for m in MEASURES) + " |\n")
        handle.write("| **Mean ± Std Dev** | " + " | ".join(f"**{mean[m]:.4f} ± {std[m]:.4f}**" for m in MEASURES) + " |\n")
    logger.info("Mean: %s", {m: round(v, 4) for m, v in mean.items()})
    logger.info("Results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

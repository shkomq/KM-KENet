"""Noisy label generation: apply every token LF and build the label matrix."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from tqdm import tqdm

from step3_weak_supervision.labeling_functions.base import TokenLabelingFunction

logger = logging.getLogger(__name__)

ABSTAIN = -1


@dataclass
class LabelMatrix:
    """`matrix[i, j]` is LF j's class index for token i, or ABSTAIN.

    Tokens of all sentences are concatenated; `offsets[k]` gives the token range of sentence k.
    """

    matrix: np.ndarray
    lf_names: list[str]
    classes: list[str]
    offsets: list[tuple[int, int]]

    @property
    def covered(self) -> np.ndarray:
        return (self.matrix != ABSTAIN).any(axis=1)


class NoisyLabelGenerator:
    def __init__(self, lfs: Sequence[TokenLabelingFunction], classes: Sequence[str]) -> None:
        if len(classes) > np.iinfo(np.int8).max:
            raise ValueError("Too many classes for an int8 label matrix")
        self.lfs = list(lfs)
        self.classes = list(classes)
        self.class_index = {label: index for index, label in enumerate(self.classes)}

    def generate(self, sentences: Sequence[Sequence[str]], show_progress: bool = True) -> LabelMatrix:
        total = sum(len(tokens) for tokens in sentences)
        matrix = np.full((total, len(self.lfs)), ABSTAIN, dtype=np.int8)
        offsets: list[tuple[int, int]] = []
        position = 0
        for tokens in tqdm(sentences, desc="Applying labeling functions", disable=not show_progress, mininterval=5):
            end = position + len(tokens)
            offsets.append((position, end))
            for column, lf in enumerate(self.lfs):
                for index, vote in enumerate(lf(tokens)):
                    if vote is None:
                        continue
                    try:
                        matrix[position + index, column] = self.class_index[vote]
                    except KeyError as exc:
                        raise ValueError(f"LF '{lf.name}' emitted unknown label '{vote}'") from exc
            position = end
        return LabelMatrix(matrix, [lf.name for lf in self.lfs], self.classes, offsets)


def summarize_label_matrix(label_matrix: LabelMatrix) -> dict[str, Any]:
    """Snorkel-style LF analysis: coverage, overlaps, conflicts and emitted labels."""
    matrix = label_matrix.matrix
    n_tokens = matrix.shape[0]
    voted = matrix != ABSTAIN
    summary: dict[str, Any] = {
        "tokens": int(n_tokens),
        "coverage_any_lf": float(voted.any(axis=1).mean()) if n_tokens else 0.0,
        "labeling_functions": [],
    }
    for column, name in enumerate(label_matrix.lf_names):
        own = voted[:, column]
        others = np.delete(matrix, column, axis=1)
        others_voted = others != ABSTAIN
        overlap = own & others_voted.any(axis=1)
        conflict = own & (others_voted & (others != matrix[:, [column]])).any(axis=1)
        values, counts = np.unique(matrix[own, column], return_counts=True)
        summary["labeling_functions"].append({
            "name": name,
            "votes": int(own.sum()),
            "coverage": float(own.mean()) if n_tokens else 0.0,
            "overlap": float(overlap.mean()) if n_tokens else 0.0,
            "conflict": float(conflict.mean()) if n_tokens else 0.0,
            "labels": {label_matrix.classes[int(v)]: int(c) for v, c in zip(values, counts)},
        })
    return summary

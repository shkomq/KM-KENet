"""Label aggregation models: majority vote and Dawid–Skene (EM).

The Dawid–Skene model learns, for every labeling function j, a confusion
matrix P(LF_j outputs l | true class k) together with the class prior, and
returns a posterior over the true class of every token. With
`model_abstains=True` abstention is treated as an extra output symbol, so the
model also learns how selective each LF is (a dictionary LF that rarely fires
on O tokens becomes strong evidence when it does fire).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np

from step3_weak_supervision.label_matrix import ABSTAIN

logger = logging.getLogger(__name__)


def _logsumexp(values: np.ndarray) -> np.ndarray:
    peak = values.max(axis=1, keepdims=True)
    return (peak + np.log(np.exp(values - peak).sum(axis=1, keepdims=True))).ravel()


class LabelModel(ABC):
    def __init__(self, num_classes: int, default_class: int = 0) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if not 0 <= default_class < num_classes:
            raise ValueError("default_class out of range")
        self.num_classes = num_classes
        self.default_class = default_class

    def _validate(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2:
            raise ValueError("Label matrix must be 2-D")
        if matrix.size and (matrix.min() < ABSTAIN or matrix.max() >= self.num_classes):
            raise ValueError("Label matrix contains out-of-range class indices")
        return matrix.astype(np.int64, copy=False)

    @abstractmethod
    def fit(self, matrix: np.ndarray) -> "LabelModel":
        ...

    @abstractmethod
    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        ...

    def predict(self, matrix: np.ndarray, chunk_size: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
        """Hard labels and their posterior probability, computed in chunks to bound memory."""
        labels = np.empty(matrix.shape[0], dtype=np.int64)
        confidences = np.empty(matrix.shape[0], dtype=np.float32)
        for start in range(0, matrix.shape[0], chunk_size):
            proba = self.predict_proba(matrix[start : start + chunk_size])
            labels[start : start + len(proba)] = proba.argmax(axis=1)
            confidences[start : start + len(proba)] = proba.max(axis=1)
        return labels, confidences

    def _default_rows(self, proba: np.ndarray, rows: np.ndarray) -> None:
        proba[rows] = 0.0
        proba[rows, self.default_class] = 1.0


class MajorityVoteLabelModel(LabelModel):
    def fit(self, matrix: np.ndarray) -> "MajorityVoteLabelModel":
        self._validate(matrix)
        return self

    def vote_counts(self, matrix: np.ndarray) -> np.ndarray:
        matrix = self._validate(matrix)
        counts = np.zeros((matrix.shape[0], self.num_classes), dtype=np.float64)
        for column in range(matrix.shape[1]):
            votes = matrix[:, column]
            rows = np.nonzero(votes != ABSTAIN)[0]
            np.add.at(counts, (rows, votes[rows]), 1.0)
        return counts

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        counts = self.vote_counts(matrix)
        totals = counts.sum(axis=1, keepdims=True)
        proba = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        self._default_rows(proba, totals.ravel() == 0)
        return proba


class DawidSkeneLabelModel(LabelModel):
    def __init__(
        self,
        num_classes: int,
        default_class: int = 0,
        max_iter: int = 100,
        tol: float = 1e-6,
        smoothing: float = 1.0,
        model_abstains: bool = True,
        fit_sample_size: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(num_classes, default_class)
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.smoothing = float(smoothing)
        self.model_abstains = bool(model_abstains)
        self.fit_sample_size = fit_sample_size
        self.seed = seed
        self.class_prior_: np.ndarray | None = None
        self.confusion_: np.ndarray | None = None  # (n_lfs, K, n_outputs)
        self.n_iter_ = 0

    @property
    def _n_outputs(self) -> int:
        return self.num_classes + 1 if self.model_abstains else self.num_classes

    def _observations(self, matrix: np.ndarray) -> np.ndarray:
        """Map ABSTAIN to an explicit output symbol when abstentions are modelled."""
        if not self.model_abstains:
            return matrix
        observed = matrix.copy()
        observed[observed == ABSTAIN] = self.num_classes
        return observed

    def fit(self, matrix: np.ndarray) -> "DawidSkeneLabelModel":
        # Sub-sample before validation so the full (int8) matrix is never up-cast in memory.
        if not self.model_abstains:
            matrix = matrix[(matrix != ABSTAIN).any(axis=1)]
        if self.fit_sample_size and matrix.shape[0] > self.fit_sample_size:
            rng = np.random.default_rng(self.seed)
            matrix = matrix[np.sort(rng.choice(matrix.shape[0], self.fit_sample_size, replace=False))]
        matrix = self._validate(matrix)
        if matrix.shape[0] == 0:
            raise ValueError("Cannot fit the label model on an empty label matrix")

        posterior = MajorityVoteLabelModel(self.num_classes, self.default_class).predict_proba(matrix)
        observed = self._observations(matrix)
        previous = mean_ll = -np.inf
        for iteration in range(1, self.max_iter + 1):
            self._m_step(observed, posterior)
            log_joint = self._log_joint(observed)
            log_norm = _logsumexp(log_joint)
            posterior = np.exp(log_joint - log_norm[:, None])
            mean_ll = float(log_norm.mean())
            self.n_iter_ = iteration
            if abs(mean_ll - previous) < self.tol:
                break
            previous = mean_ll
        logger.info("Dawid-Skene stopped after %d iterations (mean log-likelihood %.5f)", self.n_iter_, mean_ll)
        return self

    def _m_step(self, observed: np.ndarray, posterior: np.ndarray) -> None:
        k, n_out, s = self.num_classes, self._n_outputs, self.smoothing
        self.class_prior_ = (posterior.sum(axis=0) + s) / (posterior.sum() + k * s)
        confusion = np.empty((observed.shape[1], k, n_out), dtype=np.float64)
        for column in range(observed.shape[1]):
            votes = observed[:, column]
            valid = votes != ABSTAIN
            counts = np.full((k, n_out), s, dtype=np.float64)
            for true_class in range(k):
                counts[true_class] += np.bincount(
                    votes[valid], weights=posterior[valid, true_class], minlength=n_out
                )
            confusion[column] = counts / counts.sum(axis=1, keepdims=True)
        self.confusion_ = confusion

    def _log_joint(self, observed: np.ndarray) -> np.ndarray:
        assert self.class_prior_ is not None and self.confusion_ is not None
        log_joint = np.tile(np.log(self.class_prior_), (observed.shape[0], 1))
        log_confusion = np.log(self.confusion_)
        for column in range(observed.shape[1]):
            votes = observed[:, column]
            valid = votes != ABSTAIN
            log_joint[valid] += log_confusion[column][:, votes[valid]].T
        return log_joint

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        if self.confusion_ is None:
            raise RuntimeError("DawidSkeneLabelModel must be fitted first")
        matrix = self._validate(matrix)
        log_joint = self._log_joint(self._observations(matrix))
        proba = np.exp(log_joint - _logsumexp(log_joint)[:, None])
        if not self.model_abstains:
            self._default_rows(proba, ~(matrix != ABSTAIN).any(axis=1))
        return proba


def build_label_model(section: Mapping[str, Any], num_classes: int, seed: int) -> LabelModel:
    name = str(section.get("label_model", "dawid_skene")).lower()
    if name == "majority_vote":
        return MajorityVoteLabelModel(num_classes)
    if name == "dawid_skene":
        ds = section.get("dawid_skene", {}) or {}
        sample = ds.get("fit_sample_size")
        return DawidSkeneLabelModel(
            num_classes,
            max_iter=int(ds.get("max_iter", 100)),
            tol=float(ds.get("tol", 1e-6)),
            smoothing=float(ds.get("smoothing", 1.0)),
            model_abstains=bool(ds.get("model_abstains", True)),
            fit_sample_size=int(sample) if sample else None,
            seed=seed,
        )
    raise ValueError(f"Unknown label model '{name}' (expected dawid_skene or majority_vote)")


def apply_anchor_votes(
    matrix: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    anchor_columns: list[int],
    outside_class: int = 0,
    confidence: float = 0.95,
) -> int:
    """Let deterministic "anchor" LFs override the label model where nothing contradicts them.

    A class voted by a single LF that no other LF ever votes on (e.g. DOSAGE from the
    `<number> <unit>` rule) is not identifiable by an unsupervised label model: its
    tokens are explained equally well by "true O, noisy LF", and EM tends to merge the
    class into O. For such high-precision rules the LF vote is kept whenever no other
    LF votes a different entity class for the token. Returns the number of tokens changed.
    """
    changed = 0
    for column in anchor_columns:
        votes = matrix[:, column]
        fired = votes != ABSTAIN
        if not fired.any():
            continue
        conflicting = np.zeros(matrix.shape[0], dtype=bool)
        for other in range(matrix.shape[1]):  # column by column: no copy of the full matrix
            if other == column:
                continue
            others = matrix[:, other]
            conflicting |= (others != ABSTAIN) & (others != outside_class) & (others != votes)
        rows = np.nonzero(fired & ~conflicting & (labels != votes))[0]
        labels[rows] = votes[rows]
        confidences[rows] = np.maximum(confidences[rows], confidence)
        changed += int(rows.size)
    return changed

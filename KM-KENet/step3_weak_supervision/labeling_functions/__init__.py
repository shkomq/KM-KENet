"""Labeling-function registry (dictionary-, pattern- and heuristic-based)."""

from __future__ import annotations

from step2_preprocessing.stemmer import KurdishStemmer
from step3_weak_supervision.labeling_functions.base import TokenLabelingFunction
from step3_weak_supervision.labeling_functions.dictionary import DictionaryLF, LexiconMatcher
from step3_weak_supervision.labeling_functions.heuristic import build_heuristic_lfs
from step3_weak_supervision.labeling_functions.pattern import build_pattern_lfs
from step3_weak_supervision.labeling_functions.relation import build_relation_lfs


def build_ner_labeling_functions(
    matcher: LexiconMatcher,
    stemmer: KurdishStemmer,
    stopwords: frozenset[str],
) -> list[TokenLabelingFunction]:
    lfs: list[TokenLabelingFunction] = [DictionaryLF(matcher)]
    lfs.extend(build_pattern_lfs(stopwords))
    lfs.extend(build_heuristic_lfs(stemmer, stopwords))
    names = [lf.name for lf in lfs]
    if len(names) != len(set(names)):
        raise ValueError(f"Labeling function names must be unique: {names}")
    return lfs


__all__ = [
    "LexiconMatcher",
    "build_ner_labeling_functions",
    "build_relation_lfs",
]

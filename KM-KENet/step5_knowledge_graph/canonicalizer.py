"""Entity canonicalisation: map surface mentions onto knowledge-graph node ids.

Mentions that match a lexicon entry are linked to that entry (and inherit its
English label). Other mentions are grouped by their stemmed surface form, so
inflected variants (e.g. گورچیلەکان / گورچیلە) collapse into one node.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from step2_preprocessing.stemmer import KurdishStemmer
from step3_weak_supervision.labeling_functions.dictionary import LexiconMatcher

_UNSAFE_ID_CHARS = re.compile(r"[^\w]+")


@dataclass(frozen=True)
class CanonicalEntity:
    node_id: str
    entity_type: str
    label_ckb: str
    label_en: str | None
    lexicon_id: str | None


class EntityCanonicalizer:
    def __init__(self, matcher: LexiconMatcher, stemmer: KurdishStemmer, cache_size: int = 200_000) -> None:
        self.matcher = matcher
        self.stemmer = stemmer
        self._canonicalize_cached = lru_cache(maxsize=cache_size)(self._canonicalize)

    def canonicalize(self, tokens: Sequence[str], entity_type: str) -> CanonicalEntity:
        return self._canonicalize_cached(tuple(tokens), entity_type)

    def _canonicalize(self, tokens: tuple[str, ...], entity_type: str) -> CanonicalEntity:
        entry = self.matcher.lookup(tokens)
        if entry is not None:
            return CanonicalEntity(
                node_id=f"{entity_type}:{entry.entry_id}",
                entity_type=entity_type,
                label_ckb=entry.kurdish,
                label_en=entry.english,
                lexicon_id=entry.entry_id,
            )
        key = "_".join(self.stemmer.stem(token) for token in tokens)
        key = _UNSAFE_ID_CHARS.sub("_", key).strip("_") or "unknown"
        return CanonicalEntity(
            node_id=f"{entity_type}:ckb:{key}",
            entity_type=entity_type,
            label_ckb=" ".join(tokens),
            label_en=None,
            lexicon_id=None,
        )

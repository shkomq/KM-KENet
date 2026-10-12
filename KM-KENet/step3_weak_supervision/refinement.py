"""Label refinement and aggregation.

1. Token posteriors from the label model become IO types (low-confidence
   entity tokens fall back to O).
2. Entity spans are trimmed of function words and punctuation at their edges.
3. Corpus-level propagation: surface forms that are consistently labelled
   with one type are projected onto their unlabelled occurrences.
4. Relation LFs vote on every schema-compatible entity pair; weighted votes
   are aggregated into relation labels with a confidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from common.schema import NO_RELATION, OUTSIDE, candidate_pairs, io_to_spans, spans_to_bio
from step2_preprocessing.stemmer import KurdishStemmer
from step2_preprocessing.tokenizer import is_punctuation
from step3_weak_supervision.labeling_functions.base import (
    EntitySpan,
    RelationCandidate,
    RelationLabelingFunction,
)


@dataclass
class SentenceAnnotation:
    sentence_id: str
    doc_id: str
    group: str
    text: str
    tokens: list[str]
    types: list[str]            # IO type per token
    confidences: list[float]    # confidence of the type per token
    entities: list[EntitySpan] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)

    def rebuild_entities(self) -> None:
        self.entities = [
            EntitySpan(start, end, entity_type, float(np.mean(self.confidences[start:end])))
            for start, end, entity_type in io_to_spans(self.types)
        ]

    def to_record(self) -> dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "doc_id": self.doc_id,
            "group": self.group,
            "text": self.text,
            "tokens": self.tokens,
            "ner_tags": spans_to_bio([(e.start, e.end, e.type) for e in self.entities], len(self.tokens)),
            "token_confidence": [round(float(c), 4) for c in self.confidences],
            "entities": [
                {
                    "start": e.start,
                    "end": e.end,
                    "type": e.type,
                    "text": e.text(self.tokens),
                    "confidence": round(e.confidence, 4),
                }
                for e in self.entities
            ],
            "relations": self.relations,
        }


class NERLabelRefiner:
    def __init__(
        self,
        classes: Sequence[str],
        stopwords: frozenset[str],
        stemmer: KurdishStemmer,
        min_confidence: float = 0.5,
        uncovered_as_outside: bool = True,
    ) -> None:
        self.classes = list(classes)
        self.stopwords = stopwords
        self.stemmer = stemmer
        self.min_confidence = float(min_confidence)
        self.uncovered_as_outside = uncovered_as_outside

    def build_annotations(
        self,
        sentences: Sequence[dict[str, Any]],
        labels: np.ndarray,
        confidences: np.ndarray,
        covered: np.ndarray,
        offsets: Sequence[tuple[int, int]],
    ) -> list[SentenceAnnotation]:
        annotations: list[SentenceAnnotation] = []
        for sentence, (start, end) in zip(sentences, offsets):
            types: list[str] = []
            token_conf: list[float] = []
            for index in range(start, end):
                label = self.classes[int(labels[index])]
                confidence = float(confidences[index])
                if self.uncovered_as_outside and not covered[index]:
                    label, confidence = OUTSIDE, 1.0
                elif label != OUTSIDE and confidence < self.min_confidence:
                    # Probability that the token is *not* of the rejected type.
                    label, confidence = OUTSIDE, max(1.0 - confidence, 0.05)
                types.append(label)
                token_conf.append(confidence)
            annotation = SentenceAnnotation(
                sentence_id=sentence["sentence_id"],
                doc_id=sentence["doc_id"],
                group=sentence.get("group") or sentence["doc_id"],
                text=sentence["text"],
                tokens=list(sentence["tokens"]),
                types=types,
                confidences=token_conf,
            )
            self._trim_spans(annotation)
            annotation.rebuild_entities()
            annotations.append(annotation)
        return annotations

    def _is_edge_noise(self, token: str) -> bool:
        return token in self.stopwords or is_punctuation(token)

    def _trim_spans(self, annotation: SentenceAnnotation) -> None:
        for start, end, _ in io_to_spans(annotation.types):
            while start < end and self._is_edge_noise(annotation.tokens[start]):
                annotation.types[start] = OUTSIDE
                start += 1
            while end > start and self._is_edge_noise(annotation.tokens[end - 1]):
                annotation.types[end - 1] = OUTSIDE
                end -= 1

    def _key(self, tokens: Sequence[str]) -> tuple[str, ...]:
        return tuple(self.stemmer.stem(token) for token in tokens)

    def propagate(
        self,
        annotations: Sequence[SentenceAnnotation],
        min_count: int = 3,
        min_purity: float = 0.85,
        confidence_scale: float = 0.9,
    ) -> int:
        """Label unlabelled occurrences of consistently typed surface forms. Returns #spans added."""
        type_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for annotation in annotations:
            for entity in annotation.entities:
                type_counts[self._key(annotation.tokens[entity.start : entity.end])][entity.type] += 1

        reliable: dict[tuple[str, ...], tuple[str, float]] = {}
        for key, counter in type_counts.items():
            total = sum(counter.values())
            entity_type, count = counter.most_common(1)[0]
            if total >= min_count and count / total >= min_purity:
                reliable[key] = (entity_type, count / total)
        if not reliable:
            return 0
        max_length = max(len(key) for key in reliable)

        added = 0
        for annotation in annotations:
            stems = self._key(annotation.tokens)
            n = len(stems)
            position = 0
            changed = False
            while position < n:
                matched = 0
                for size in range(min(max_length, n - position), 0, -1):
                    hit = reliable.get(stems[position : position + size])
                    if hit is None:
                        continue
                    if any(annotation.types[k] != OUTSIDE for k in range(position, position + size)):
                        continue
                    entity_type, purity = hit
                    for k in range(position, position + size):
                        annotation.types[k] = entity_type
                        annotation.confidences[k] = purity * confidence_scale
                    matched = size
                    break
                if matched:
                    added += 1
                    changed = True
                    position += matched
                else:
                    position += 1
            if changed:
                annotation.rebuild_entities()
        return added


class RelationLabeler:
    """Weighted vote of relation LFs over schema-compatible entity pairs."""

    def __init__(
        self,
        lfs: Sequence[RelationLabelingFunction],
        min_confidence: float = 0.5,
        no_relation_prior: float = 1.0,
    ) -> None:
        self.lfs = list(lfs)
        self.min_confidence = float(min_confidence)
        # Pseudo-vote for NO_RELATION: a single weak LF can never assert a relation on its own.
        self.no_relation_prior = float(no_relation_prior)
        self.vote_counts: Counter[str] = Counter()

    def label(self, annotation: SentenceAnnotation) -> list[dict[str, Any]]:
        tokens = tuple(annotation.tokens)
        entities = annotation.entities
        relations: list[dict[str, Any]] = []
        for head_index, tail_index, relation in candidate_pairs([e.type for e in entities]):
            candidate = RelationCandidate(tokens, entities[head_index], entities[tail_index], relation)
            scores: dict[str, float] = defaultdict(float)
            scores[NO_RELATION] = self.no_relation_prior
            for lf in self.lfs:
                vote = lf(candidate)
                if vote is not None:
                    scores[vote] += lf.weight
                    self.vote_counts[f"{lf.name}:{vote}"] += 1
            # Ties are resolved in favour of NO_RELATION (precision first).
            total = sum(scores.values())
            if total <= 0:
                continue
            winner = max(scores, key=lambda label: (scores[label], label == NO_RELATION))
            confidence = scores[winner] / total
            if winner != NO_RELATION and confidence >= self.min_confidence:
                relations.append({
                    "head": head_index,
                    "tail": tail_index,
                    "type": winner,
                    "confidence": round(confidence, 4),
                })
        return relations

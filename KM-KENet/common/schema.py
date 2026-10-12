"""Annotation schema shared by weak supervision, the model and the knowledge graph.

Entities use the BIO scheme at word level. Relations are directed and only
allowed between specific entity-type pairs (see `RELATION_SCHEMA`).
"""

from __future__ import annotations

from typing import Sequence

OUTSIDE = "O"
NO_RELATION = "NO_RELATION"

# The six medical entity categories of the paper (Disease, Symptom, Drug,
# Dosage, Procedure, Body Part). PROCEDURE covers therapeutic procedures
# (surgery, radiotherapy, ...) as well as diagnostic tests (blood test, X-ray, ...).
ENTITY_TYPES: tuple[str, ...] = ("DISEASE", "SYMPTOM", "DRUG", "DOSAGE", "PROCEDURE", "BODY_PART")

RELATION_SCHEMA: dict[str, tuple[tuple[str, str], ...]] = {
    "TREATS": (("DRUG", "DISEASE"), ("DRUG", "SYMPTOM"), ("PROCEDURE", "DISEASE"), ("PROCEDURE", "SYMPTOM")),
    "HAS_SYMPTOM": (("DISEASE", "SYMPTOM"),),
    "AFFECTS": (("DISEASE", "BODY_PART"),),
    "DIAGNOSED_BY": (("DISEASE", "PROCEDURE"),),
    "HAS_DOSAGE": (("DRUG", "DOSAGE"),),
}

RELATION_TYPES: tuple[str, ...] = (NO_RELATION, *RELATION_SCHEMA)

TYPE_PAIR_TO_RELATION: dict[tuple[str, str], str] = {
    pair: relation for relation, pairs in RELATION_SCHEMA.items() for pair in pairs
}

Span = tuple[int, int, str]  # (start, end_exclusive, entity_type)


def bio_labels(entity_types: Sequence[str] = ENTITY_TYPES) -> list[str]:
    labels = [OUTSIDE]
    for entity_type in entity_types:
        labels.extend((f"B-{entity_type}", f"I-{entity_type}"))
    return labels


def split_tag(tag: str) -> tuple[str, str | None]:
    if tag == OUTSIDE or "-" not in tag:
        return OUTSIDE, None
    prefix, entity_type = tag.split("-", 1)
    return prefix, entity_type


def repair_bio(tags: Sequence[str]) -> list[str]:
    """Turn dangling `I-X` tags (not preceded by X) into `B-X`."""
    repaired: list[str] = []
    previous_type: str | None = None
    for tag in tags:
        prefix, entity_type = split_tag(tag)
        if entity_type is None:
            repaired.append(OUTSIDE)
            previous_type = None
            continue
        if prefix == "I" and previous_type != entity_type:
            prefix = "B"
        repaired.append(f"{prefix}-{entity_type}")
        previous_type = entity_type
    return repaired


def bio_to_spans(tags: Sequence[str]) -> list[Span]:
    spans: list[Span] = []
    start: int | None = None
    current: str | None = None
    for index, tag in enumerate(repair_bio(tags)):
        prefix, entity_type = split_tag(tag)
        if prefix == "B" or entity_type is None:
            if current is not None and start is not None:
                spans.append((start, index, current))
            start, current = (index, entity_type) if entity_type else (None, None)
    if current is not None and start is not None:
        spans.append((start, len(tags), current))
    return spans


def spans_to_bio(spans: Sequence[Span], length: int) -> list[str]:
    tags = [OUTSIDE] * length
    for start, end, entity_type in sorted(spans):
        if not 0 <= start < end <= length:
            raise ValueError(f"Invalid span ({start}, {end}) for sequence of length {length}")
        tags[start] = f"B-{entity_type}"
        for index in range(start + 1, end):
            tags[index] = f"I-{entity_type}"
    return tags


def io_to_spans(types: Sequence[str]) -> list[Span]:
    """Contiguous runs of the same non-O type become one span."""
    spans: list[Span] = []
    start = 0
    for index in range(1, len(types) + 1):
        if index == len(types) or types[index] != types[start]:
            if types[start] != OUTSIDE:
                spans.append((start, index, types[start]))
            start = index
    return spans


def candidate_pairs(entity_types: Sequence[str]) -> list[tuple[int, int, str]]:
    """All ordered entity pairs whose type pair admits a relation."""
    pairs: list[tuple[int, int, str]] = []
    for i, head_type in enumerate(entity_types):
        for j, tail_type in enumerate(entity_types):
            if i == j:
                continue
            relation = TYPE_PAIR_TO_RELATION.get((head_type, tail_type))
            if relation is not None:
                pairs.append((i, j, relation))
    return pairs

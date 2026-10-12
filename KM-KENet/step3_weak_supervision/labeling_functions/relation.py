"""Relation labeling functions over entity-pair candidates."""

from __future__ import annotations

from typing import Mapping, Sequence

from common.schema import NO_RELATION
from step3_weak_supervision.labeling_functions.base import (
    EntitySpan,
    RelationCandidate,
    RelationLabelingFunction,
    find_subsequence,
)

RELATION_TRIGGERS: dict[str, tuple[str, ...]] = {
    "TREATS": ("چارەسەر", "بەکاردێ", "بەکاردەهێنرێ", "بەکاردەبر", "دەدرێت", "وەردەگیرێت", "کەمکردنەوە", "دەرمان"),
    "HAS_SYMPTOM": ("نیشانە", "بریتین", "بریتییە", "دەردەکەوێت", "هۆی"),
    "AFFECTS": ("کاریگەری", "لەسەر", "زیان", "تووش", "تێکدەدات"),
    "DIAGNOSED_BY": ("دەستنیشان", "پشکنین", "دەدۆزرێتەوە", "دیاری"),
}

NEGATION_CUES = frozenset({"نەک", "نییە", "ناکات", "نابێت", "ناتوانێت", "نەبوو", "نەکراوە", "بەبێ", "ناکرێت"})


def _on_side(span: EntitySpan, anchor_start: int, anchor_end: int, side: str) -> bool:
    return span.end <= anchor_start if side == "before" else span.start >= anchor_end


class ArgumentSlotLF(RelationLabelingFunction):
    """Head and tail occupy the argument slots around an anchor phrase.

    e.g. TREATS: `<DRUG> بۆ چارەسەری <DISEASE>` -> head before, tail after the anchor.
    """

    def __init__(
        self,
        name: str,
        relation: str,
        anchors: Sequence[Sequence[str]],
        head_side: str,
        tail_side: str,
        weight: float = 2.0,
        context_prefixes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(name, weight)
        if head_side not in ("before", "after") or tail_side not in ("before", "after"):
            raise ValueError("head_side/tail_side must be 'before' or 'after'")
        self.relation = relation
        self.anchors = tuple(tuple(anchor) for anchor in anchors)
        self.head_side = head_side
        self.tail_side = tail_side
        self.context_prefixes = context_prefixes

    def apply(self, candidate: RelationCandidate) -> str | None:
        if candidate.relation != self.relation:
            return None
        tokens = candidate.tokens
        if self.context_prefixes and not any(token.startswith(self.context_prefixes) for token in tokens):
            return None
        for anchor in self.anchors:
            for start in find_subsequence(tokens, anchor):
                end = start + len(anchor)
                if _on_side(candidate.head, start, end, self.head_side) and _on_side(
                    candidate.tail, start, end, self.tail_side
                ):
                    return self.relation
        return None


class TriggerCooccurrenceLF(RelationLabelingFunction):
    """A trigger word of the candidate relation appears in the sentence."""

    def __init__(self, triggers: Mapping[str, tuple[str, ...]], weight: float = 1.0) -> None:
        super().__init__("relation_trigger_cooccurrence", weight)
        self.triggers = triggers

    def apply(self, candidate: RelationCandidate) -> str | None:
        prefixes = self.triggers.get(candidate.relation, ())
        if prefixes and any(token.startswith(prefixes) for token in candidate.tokens):
            return candidate.relation
        return None


class MissingTriggerLF(RelationLabelingFunction):
    """No trigger word of the candidate relation anywhere in the sentence."""

    def __init__(self, triggers: Mapping[str, tuple[str, ...]], weight: float = 1.0) -> None:
        super().__init__("relation_missing_trigger", weight)
        self.triggers = triggers

    def apply(self, candidate: RelationCandidate) -> str | None:
        prefixes = self.triggers.get(candidate.relation, ())
        if prefixes and not any(token.startswith(prefixes) for token in candidate.tokens):
            return NO_RELATION
        return None


class DistanceLF(RelationLabelingFunction):
    def __init__(self, max_distance: int, weight: float = 2.0) -> None:
        super().__init__("relation_long_distance", weight)
        self.max_distance = max_distance

    def apply(self, candidate: RelationCandidate) -> str | None:
        return NO_RELATION if candidate.distance > self.max_distance else None


class NegationLF(RelationLabelingFunction):
    """A negation cue between the arguments cancels the relation."""

    def __init__(self, cues: frozenset[str] = NEGATION_CUES, weight: float = 4.0) -> None:
        super().__init__("relation_negation", weight)
        self.cues = cues

    def apply(self, candidate: RelationCandidate) -> str | None:
        tokens = candidate.tokens
        if any(tokens[index] in self.cues for index in candidate.gap):
            return NO_RELATION
        return None


class AdjacentDosageLF(RelationLabelingFunction):
    """HAS_DOSAGE: the dosage follows the drug within a few tokens (`پاراسیتامۆڵ 500 mg`)."""

    def __init__(self, max_gap: int = 3, weight: float = 3.0) -> None:
        super().__init__("relation_adjacent_dosage", weight)
        self.max_gap = max_gap

    def apply(self, candidate: RelationCandidate) -> str | None:
        if candidate.relation != "HAS_DOSAGE":
            return None
        if candidate.tail.start >= candidate.head.end and candidate.distance <= self.max_gap:
            return "HAS_DOSAGE"
        return None


def build_relation_lfs(max_distance: int) -> list[RelationLabelingFunction]:
    return [
        ArgumentSlotLF(
            "relation_treats_slots", "TREATS",
            [
                ("بۆ", "چارەسەری"), ("بۆ", "چارەسەرکردنی"), ("لە", "چارەسەرکردنی"), ("بۆ", "کەمکردنەوەی"),
                ("بەکاردێت", "بۆ"), ("بەکاردەبرێت", "بۆ"), ("بەکاردەهێنرێت", "بۆ"),
            ],
            head_side="before", tail_side="after",
        ),
        ArgumentSlotLF(
            "relation_causes_symptom_slots", "HAS_SYMPTOM", [("دەبێتە", "هۆی"), ("دەبنە", "هۆی")],
            head_side="before", tail_side="after",
        ),
        ArgumentSlotLF(
            "relation_symptom_list_slots", "HAS_SYMPTOM",
            [("بریتین", "لە"), ("بریتییە", "لە")],
            head_side="before", tail_side="after", context_prefixes=("نیشانە",),
        ),
        ArgumentSlotLF(
            "relation_affects_slots", "AFFECTS", [("کاریگەری", "لەسەر")],
            head_side="before", tail_side="after",
        ),
        ArgumentSlotLF(
            "relation_diagnosis_slots", "DIAGNOSED_BY", [("بە",)],
            head_side="before", tail_side="after", context_prefixes=("دەستنیشان",),
        ),
        AdjacentDosageLF(),
        TriggerCooccurrenceLF(RELATION_TRIGGERS),
        MissingTriggerLF(RELATION_TRIGGERS),
        DistanceLF(max_distance),
        NegationLF(),
    ]

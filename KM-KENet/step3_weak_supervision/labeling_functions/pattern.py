"""Pattern-based labeling: Sorani constructions that introduce medical entities.

Examples (Sorani is SOV, so arguments precede the verb):

* `X بۆ چارەسەری Y بەکاردێت`          X = DRUG, Y = DISEASE
* `نیشانەکانی Y بریتین لە A و B`       A, B = SYMPTOM
* `Y بە X دەستنیشان دەکرێت`           X = PROCEDURE (diagnostic test)
* `Y کاریگەری لەسەر A و B هەیە`        A, B = BODY_PART
"""

from __future__ import annotations

from typing import Iterable, Sequence

from step2_preprocessing.tokenizer import is_punctuation
from step3_weak_supervision.labeling_functions.base import (
    TokenLabelingFunction,
    TokenVote,
    find_subsequence,
)

CLAUSE_BOUNDARIES = frozenset({"کە", "بەڵام", "چونکە", "ئەگەر", "نەک", "هەروەها", "بۆیە"})

COMMON_PREDICATES = frozenset({
    "هەیە", "هەن", "هەبوو", "نییە", "بوو", "بێت", "یە", "ە",
    "بەکاردێت", "بەکاردێن", "بەکاردەهێنرێت", "بەکاردەهێنرێن", "بەکاردەبرێت", "بەکاردەبرێن",
    "دەبێتە", "دەبنە", "باوە", "بڵاوە", "بڵاون",
})
# Words that introduce a treatment phrase; they end a DISEASE window (e.g. "بەکاردەبرێت بۆ چارەسەری ...").
TREATMENT_WORDS = frozenset({"چارەسەری", "چارەسەرکردنی", "کەمکردنەوەی", "ڕێگریکردن"})
_VERB_PREFIXES = ("دە", "نا", "ئە")
_VERB_ENDINGS = ("ێتەوە", "ێنەوە", "اتەوە", "ێت", "ێن", "ات")


def is_verb_like(token: str) -> bool:
    """Heuristic for finite Sorani verbs (present-tense prefix + person ending)."""
    if token in COMMON_PREDICATES:
        return True
    return len(token) >= 4 and token.startswith(_VERB_PREFIXES) and token.endswith(_VERB_ENDINGS)


class TriggerPatternLF(TokenLabelingFunction):
    """Labels up to `window` tokens before/after a trigger phrase.

    * `skip_tokens` are passed over without being labelled (e.g. "و" in lists).
    * Punctuation, clause boundaries, `stop_tokens` and (optionally) verbs end the window.
    * `context_prefixes`: the LF only fires if some token starts with one of them.
    """

    def __init__(
        self,
        name: str,
        label: str,
        triggers: Iterable[Sequence[str]],
        *,
        direction: str = "after",
        window: int = 1,
        include_trigger: bool = False,
        skip_tokens: frozenset[str] = frozenset(),
        stop_tokens: frozenset[str] = frozenset(),
        stop_at_verbs: bool = True,
        context_prefixes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(name)
        if direction not in ("after", "before"):
            raise ValueError("direction must be 'after' or 'before'")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.label = label
        self.triggers = tuple(tuple(trigger) for trigger in triggers)
        self.direction = direction
        self.window = window
        self.include_trigger = include_trigger
        self.skip_tokens = skip_tokens
        self.stop_tokens = stop_tokens
        self.stop_at_verbs = stop_at_verbs
        self.context_prefixes = context_prefixes

    def _is_stop(self, token: str) -> bool:
        return (
            is_punctuation(token)
            or token in self.stop_tokens
            or token in CLAUSE_BOUNDARIES
            or (self.stop_at_verbs and is_verb_like(token))
        )

    def apply(self, tokens: Sequence[str]) -> list[TokenVote]:
        votes: list[TokenVote] = [None] * len(tokens)
        if self.context_prefixes and not any(token.startswith(self.context_prefixes) for token in tokens):
            return votes
        for trigger in self.triggers:
            for start in find_subsequence(tokens, trigger):
                end = start + len(trigger)
                if self.include_trigger:
                    for index in range(start, end):
                        votes[index] = self.label
                self._label_window(tokens, votes, start, end)
        return votes

    def _label_window(self, tokens: Sequence[str], votes: list[TokenVote], start: int, end: int) -> None:
        positions = range(end, len(tokens)) if self.direction == "after" else range(start - 1, -1, -1)
        labelled = 0
        for index in positions:
            token = tokens[index]
            if self._is_stop(token):
                break
            if token in self.skip_tokens:
                continue
            votes[index] = self.label
            labelled += 1
            if labelled >= self.window:
                break


def build_pattern_lfs(stopwords: frozenset[str]) -> list[TokenLabelingFunction]:
    # Constructions observed in the KMC corpus (approximate counts in kurdish_medical_corpus_kmc.json):
    #   بۆ چارەسەری (5.7k) · بۆ چارەسەرکردنی (2.3k) · لە چارەسەرکردنی (1.5k)
    #   بەکاردێت / بەکاردەبرێت / بەکاردەهێنرێت بۆ (2.3k) · نیشانەکانی (12.6k) · هۆکارەکانی (4.8k)
    #   تووشبوون بە (1.9k) · بە دەرمانی (0.4k)
    treat_triggers = [("بۆ", "چارەسەری"), ("بۆ", "چارەسەرکردنی"), ("لە", "چارەسەرکردنی")]
    used_for_triggers = [("بەکاردێت", "بۆ"), ("بەکاردەبرێت", "بۆ"), ("بەکاردەهێنرێت", "بۆ")]
    return [
        TriggerPatternLF(
            "pattern_disease_after_nexoshi", "DISEASE", [("نەخۆشی",), ("نەخۆشیی",)],
            window=1, stop_tokens=stopwords,
        ),
        TriggerPatternLF(
            "pattern_drug_before_treat", "DRUG",
            [*treat_triggers, ("بۆ", "کەمکردنەوەی")],
            direction="before", window=1, stop_tokens=stopwords,
        ),
        TriggerPatternLF(
            "pattern_condition_after_treat", "DISEASE", treat_triggers,
            window=2, stop_tokens=stopwords,
        ),
        TriggerPatternLF(
            "pattern_drug_before_used_for", "DRUG", used_for_triggers,
            direction="before", window=1, stop_tokens=stopwords,
        ),
        TriggerPatternLF(
            "pattern_condition_after_used_for", "DISEASE", used_for_triggers,
            window=2, stop_tokens=stopwords | TREATMENT_WORDS,
        ),
        TriggerPatternLF(
            "pattern_drug_after_with_drug", "DRUG", [("بە", "دەرمانی")],
            window=1, stop_tokens=stopwords,
        ),
        TriggerPatternLF(
            "pattern_disease_after_symptoms_of", "DISEASE", [("نیشانەکانی",), ("هۆکارەکانی",)],
            window=2, stop_tokens=stopwords | {"بریتین", "بریتییە"},
        ),
        TriggerPatternLF(
            "pattern_disease_after_infected_with", "DISEASE",
            [("تووشبوون", "بە"), ("بۆ", "دەستنیشانکردنی")],
            window=2, stop_tokens=stopwords,
        ),
        TriggerPatternLF(
            "pattern_symptom_list", "SYMPTOM", [("بریتین", "لە"), ("بریتییە", "لە")],
            window=12, skip_tokens=stopwords, context_prefixes=("نیشانە",),
        ),
        TriggerPatternLF(
            "pattern_procedure_after_examination", "PROCEDURE", [("پشکنینی",)],
            window=1, include_trigger=True, stop_tokens=stopwords,
        ),
        TriggerPatternLF(
            "pattern_procedure_before_diagnosis", "PROCEDURE", [("دەستنیشان",)],
            direction="before", window=5, skip_tokens=stopwords, stop_tokens=frozenset({"بە"}),
        ),
        TriggerPatternLF(
            "pattern_procedure_surgery", "PROCEDURE", [("نەشتەرگەری",), ("نەشتەرگەریی",)],
            window=1, include_trigger=True, stop_tokens=stopwords,
        ),
        TriggerPatternLF(
            "pattern_body_part_after_affects", "BODY_PART", [("کاریگەری", "لەسەر")],
            window=5, skip_tokens=stopwords,
        ),
        TriggerPatternLF(
            "pattern_pain_symptom", "SYMPTOM", [("ئازاری",)],
            window=1, include_trigger=True, stop_tokens=stopwords,
        ),
        TriggerPatternLF(
            "pattern_disease_after_afflicted", "DISEASE", [("تووشی",)],
            window=2, stop_tokens=stopwords | {"بوون", "دەبن", "دەبێت", "بووە", "بوو"},
        ),
    ]

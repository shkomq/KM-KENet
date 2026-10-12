"""Heuristic labeling: morphology of drug/disease names, dosage expressions, function words."""

from __future__ import annotations

import re
from typing import Sequence

from common.schema import OUTSIDE
from step2_preprocessing.stemmer import KurdishStemmer
from step2_preprocessing.tokenizer import is_number, is_punctuation
from step3_weak_supervision.labeling_functions.base import TokenLabelingFunction, TokenVote

DRUG_SUFFIXES_CKB: tuple[str, ...] = (
    "سیلین", "مایسین", "سایکلین", "پیسین", "ازۆل", "پرۆفین", "ستاتین", "پریل", "ۆلۆل",
    "تامۆڵ", "تامۆل", "دیپین", "سولۆسین", "نیازید", "ۆکسامین", "فۆرمین", "سۆلین", "سولین",
    "ازیپام", "سارتان", "پرازۆل", "فلۆکساسین",
)
DRUG_PATTERN_LATIN = re.compile(
    r"[a-z]+(?:mycin|cillin|cycline|azole|profen|statin|pril|olol|sartan|dipine|formin|"
    r"amol|azepam|prazole|floxacin|mab|vir|sulin)"
)

DISEASE_SUFFIXES_CKB: tuple[str, ...] = ("یمیا", "ایتیس", "ایتس", "ۆسیس", "ۆما", "پاسی")
DISEASE_PATTERN_LATIN = re.compile(r"[a-z]+(?:itis|osis|emia|aemia|oma|pathy|algia)")

DOSAGE_UNITS = frozenset({
    "mg", "ml", "g", "mcg", "µg", "iu", "kg",
    "ملگم", "ملیگرام", "میلیگرام", "مایکرۆگرام", "گرام", "مل", "میلیلیتر", "یەکە", "دانە", "حەب", "کەپسوول",
    "قوم", "کەوچک", "دڵۆپ", "دەرزی",
})
# Sorani number words that can stand for the dose amount ("دوو حەب" = two tablets).
NUMBER_WORDS_CKB = frozenset({"یەک", "دوو", "سێ", "چوار", "پێنج", "شەش", "حەوت", "هەشت", "نۆ", "دە", "نیو"})


def _is_amount(token: str) -> bool:
    return is_number(token) or token in NUMBER_WORDS_CKB


def dosage_spans(tokens: Sequence[str]) -> list[tuple[int, int]]:
    """`<amount> <unit>` spans, e.g. `500 mg`, `دوو حەب` (two tablets)."""
    spans: list[tuple[int, int]] = []
    for index in range(len(tokens) - 1):
        if _is_amount(tokens[index]) and tokens[index + 1] in DOSAGE_UNITS:
            spans.append((index, index + 2))
    return spans


class SuffixMorphologyLF(TokenLabelingFunction):
    """Votes `label` for tokens whose (stripped) form ends with a characteristic suffix."""

    def __init__(
        self,
        name: str,
        label: str,
        kurdish_suffixes: tuple[str, ...],
        latin_pattern: re.Pattern[str],
        stemmer: KurdishStemmer,
        min_length: int = 6,
    ) -> None:
        super().__init__(name)
        self.label = label
        self.kurdish_suffixes = kurdish_suffixes
        self.latin_pattern = latin_pattern
        self.stemmer = stemmer
        self.min_length = min_length

    def apply(self, tokens: Sequence[str]) -> list[TokenVote]:
        votes: list[TokenVote] = [None] * len(tokens)
        for index, token in enumerate(tokens):
            if len(token) < self.min_length:
                continue
            if token.isascii():
                if self.latin_pattern.fullmatch(token):
                    votes[index] = self.label
            elif any(form.endswith(self.kurdish_suffixes) for form in self.stemmer.candidates(token)):
                votes[index] = self.label
        return votes


class DosageContextLF(TokenLabelingFunction):
    """`<drug> <amount> <unit>` -> the word before the dose is a DRUG."""

    def __init__(self, name: str = "heuristic_dosage_context") -> None:
        super().__init__(name)

    def apply(self, tokens: Sequence[str]) -> list[TokenVote]:
        votes: list[TokenVote] = [None] * len(tokens)
        for start, _ in dosage_spans(tokens):
            if start == 0:
                continue
            token = tokens[start - 1]
            if not (_is_amount(token) or is_punctuation(token)):
                votes[start - 1] = "DRUG"
        return votes


class DosageSpanLF(TokenLabelingFunction):
    """`<amount> <unit>` (e.g. `500 mg`) is a DOSAGE entity."""

    def __init__(self, name: str = "heuristic_dosage_span") -> None:
        super().__init__(name)

    def apply(self, tokens: Sequence[str]) -> list[TokenVote]:
        votes: list[TokenVote] = [None] * len(tokens)
        for start, end in dosage_spans(tokens):
            for index in range(start, end):
                votes[index] = "DOSAGE"
        return votes


class FunctionWordLF(TokenLabelingFunction):
    """Function words, punctuation and numbers are never entities (except inside a dosage)."""

    def __init__(self, stopwords: frozenset[str], name: str = "heuristic_function_words") -> None:
        super().__init__(name)
        self.stopwords = stopwords

    def apply(self, tokens: Sequence[str]) -> list[TokenVote]:
        in_dosage = {index for start, end in dosage_spans(tokens) for index in range(start, end)}
        return [
            None
            if index in in_dosage
            else OUTSIDE if token in self.stopwords or is_punctuation(token) or is_number(token) else None
            for index, token in enumerate(tokens)
        ]


def build_heuristic_lfs(stemmer: KurdishStemmer, stopwords: frozenset[str]) -> list[TokenLabelingFunction]:
    return [
        SuffixMorphologyLF(
            "heuristic_drug_morphology", "DRUG", DRUG_SUFFIXES_CKB, DRUG_PATTERN_LATIN, stemmer,
        ),
        SuffixMorphologyLF(
            "heuristic_disease_morphology", "DISEASE", DISEASE_SUFFIXES_CKB, DISEASE_PATTERN_LATIN, stemmer,
        ),
        DosageContextLF(),
        DosageSpanLF(),
        FunctionWordLF(stopwords),
    ]

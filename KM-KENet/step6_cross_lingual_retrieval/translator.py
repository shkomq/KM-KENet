"""Lexicon-based English <-> Sorani term translation (fallback when Aya is unavailable).

It produces keyword-style translations (the medical terms and key words of
the query), which is what the BM25 and knowledge-graph retrievers need.
"""

from __future__ import annotations

import re
from typing import Sequence

from common.lexicon import LexiconEntry
from step2_preprocessing.text_tools import KurdishTextTools
from step3_weak_supervision.labeling_functions.dictionary import LexiconMatcher

_ENGLISH_WORD = re.compile(r"[a-z0-9]+")


def english_words(text: str) -> list[str]:
    return _ENGLISH_WORD.findall(text.lower())


def _singular_variants(word: str) -> list[str]:
    variants = [word]
    if word.endswith("ies") and len(word) > 4:
        variants.append(word[:-3] + "y")
    if word.endswith("es") and len(word) > 3:
        variants.append(word[:-2])
    if word.endswith("s") and len(word) > 3:
        variants.append(word[:-1])
    return variants


class LexiconTranslator:
    def __init__(
        self,
        entries: Sequence[LexiconEntry],
        glossary: Sequence[tuple[str, str]],
        tools: KurdishTextTools,
    ) -> None:
        self.tools = tools
        self.matcher = LexiconMatcher(entries, tools)
        self._en_to_ckb: dict[tuple[str, ...], str] = {}
        for entry in entries:
            self._en_to_ckb.setdefault(tuple(english_words(entry.english)), entry.kurdish)
        for english, kurdish in glossary:
            self._en_to_ckb.setdefault(tuple(english_words(english)), kurdish)
        self._en_to_ckb.pop((), None)
        self._max_ngram = max((len(key) for key in self._en_to_ckb), default=1)

        self._ckb_stem_to_en: dict[str, str] = {}
        for english, kurdish in glossary:
            tokens = tools.tokenize(kurdish)
            if len(tokens) == 1:
                self._ckb_stem_to_en.setdefault(tools.stemmer.stem(tokens[0]), english)

    def translate(self, text: str, source: str, target: str) -> str:
        if source == target:
            return text
        if source == "en" and target == "ckb":
            return self._english_to_kurdish(text) or text
        if source == "ckb" and target == "en":
            return self._kurdish_to_english(text) or text
        return text

    def _lookup(self, phrase: list[str]) -> str | None:
        *head, last = phrase
        for variant in _singular_variants(last):
            hit = self._en_to_ckb.get((*head, variant))
            if hit is not None:
                return hit
        return None

    def _english_to_kurdish(self, text: str) -> str:
        words = english_words(text)
        terms: list[str] = []
        position = 0
        while position < len(words):
            for size in range(min(self._max_ngram, len(words) - position), 0, -1):
                hit = self._lookup(words[position : position + size])
                if hit is not None:
                    terms.append(hit)
                    position += size
                    break
            else:
                position += 1
        return " ".join(dict.fromkeys(terms))

    def _kurdish_to_english(self, text: str) -> str:
        tokens = self.tools.tokenize(text)
        covered: set[int] = set()
        terms: list[str] = []
        for match in self.matcher.find(tokens):
            terms.append(match.entry.english)
            covered.update(range(match.start, match.end))
        for index, token in enumerate(tokens):
            if index not in covered:
                english = self._ckb_stem_to_en.get(self.tools.stemmer.stem(token))
                if english:
                    terms.append(english)
        return " ".join(dict.fromkeys(terms))

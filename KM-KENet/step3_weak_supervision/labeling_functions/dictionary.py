"""Dictionary-based labeling: suffix-tolerant longest-match over the medical lexicon."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from common.lexicon import LexiconEntry
from step2_preprocessing.text_tools import KurdishTextTools
from step3_weak_supervision.labeling_functions.base import TokenLabelingFunction, TokenVote


@dataclass(frozen=True)
class LexiconMatch:
    start: int
    end: int
    entry: LexiconEntry


class LexiconMatcher:
    """Finds lexicon terms in tokenised text.

    Each token is expanded into its stem candidates (token + suffix-stripped
    variants), so `گورچیلەکان` matches the entry `گورچیلە`. Overlaps are
    resolved by preferring the longest term at the left-most position.

    With `include_english=True` the English name of each entry is matched too:
    KMC writes many drug/disease names in Latin script inside Sorani sentences
    (e.g. "Amoxicillin بەکاردەبرێت بۆ ..."), lower-cased by the normaliser.
    """

    def __init__(
        self, entries: Sequence[LexiconEntry], tools: KurdishTextTools, include_english: bool = True
    ) -> None:
        self._stemmer = tools.stemmer
        index: dict[str, list[tuple[tuple[str, ...], LexiconEntry]]] = defaultdict(list)
        max_length = 0
        for entry in entries:
            forms = (*entry.surface_forms, entry.english) if include_english else entry.surface_forms
            for form in forms:
                words = tuple(tools.tokenize(form))
                if not words:
                    continue
                index[words[0]].append((words, entry))
                max_length = max(max_length, len(words))
        for key in index:
            index[key].sort(key=lambda item: len(item[0]), reverse=True)
        self._index = dict(index)
        self.entries = tuple(entries)
        self.max_length = max_length

    def find(self, tokens: Sequence[str]) -> list[LexiconMatch]:
        n = len(tokens)
        candidate_lists = [self._stemmer.candidates(token) for token in tokens]
        candidate_sets = [set(forms) for forms in candidate_lists]
        matches: list[LexiconMatch] = []
        position = 0
        while position < n:
            best: tuple[int, LexiconEntry] | None = None
            for form in candidate_lists[position]:
                for words, entry in self._index.get(form, ()):
                    size = len(words)
                    if (best is not None and size <= best[0]) or position + size > n:
                        continue
                    if all(words[k] in candidate_sets[position + k] for k in range(1, size)):
                        best = (size, entry)
            if best is None:
                position += 1
                continue
            matches.append(LexiconMatch(position, position + best[0], best[1]))
            position += best[0]
        return matches

    def lookup(self, tokens: Sequence[str]) -> LexiconEntry | None:
        """Entry that covers exactly the whole token sequence, if any."""
        matches = self.find(tokens)
        if len(matches) == 1 and matches[0].start == 0 and matches[0].end == len(tokens):
            return matches[0].entry
        return None


class DictionaryLF(TokenLabelingFunction):
    def __init__(self, matcher: LexiconMatcher, name: str = "dictionary_lexicon") -> None:
        super().__init__(name)
        self.matcher = matcher

    def apply(self, tokens: Sequence[str]) -> list[TokenVote]:
        votes: list[TokenVote] = [None] * len(tokens)
        for match in self.matcher.find(tokens):
            for index in range(match.start, match.end):
                votes[index] = match.entry.entity_type
        return votes

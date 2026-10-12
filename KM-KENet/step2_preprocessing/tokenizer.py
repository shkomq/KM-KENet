"""Rule-based sentence splitter and word tokenizer for Kurdish text.

Sub-word tokenisation for the neural model is done later by the XLM-R
tokenizer; this module produces the word-level units that weak supervision
labels and that the model predicts tags for.
"""

from __future__ import annotations

import re
import unicodedata

# Arabic combining marks are allowed inside words (in case diacritics are kept).
_MARKS = "ؐ-ًؚ-ٰٟۖ-ۭ"
_WORD = rf"(?:[^\W\d_]|[{_MARKS}])+"

_TOKEN_PATTERN = re.compile(
    rf"""
    \d+(?:[.,]\d+)*%?                 # numbers, decimals, percentages
    | {_WORD}(?:['’\-]{_WORD})*       # words with internal apostrophes / hyphens
    | [^\w\s]                         # any other single symbol or punctuation mark
    """,
    re.VERBOSE,
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟…])\s+|\n+")
_NUMBER = re.compile(r"^\d+(?:[.,]\d+)*%?$")


class KurdishTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(text)

    def split_sentences(self, text: str) -> list[str]:
        return [sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence and sentence.strip()]


def is_punctuation(token: str) -> bool:
    return bool(token) and all(unicodedata.category(char)[0] in ("P", "S") for char in token)


def is_number(token: str) -> bool:
    return bool(_NUMBER.match(token))

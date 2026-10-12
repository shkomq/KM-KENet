"""Light suffix stripper for Sorani Kurdish.

Sorani attaches definiteness, plurality, ezafe and clitics directly to the
noun (e.g. گورچیلە -> گورچیلەکان, گورچیلەکەی, گورچیلەدا). The stemmer exposes
two operations:

* `candidates(token)` — the token plus every single-suffix-stripped variant;
  used for suffix-tolerant dictionary matching.
* `stem(token)` — the longest-suffix-stripped form; used as a retrieval key.
"""

from __future__ import annotations

SORANI_SUFFIXES: tuple[str, ...] = (
    "ەکانەوە", "ەکانیان", "ەکاندا", "ەکانی", "ەکان",
    "ەکەیان", "ەکەوە", "ەکەدا", "ەکەی", "ەکە",
    "ییەکانی", "ییەکان", "ییەکی", "ییەک",
    "یەکانی", "یەکان", "یەکەی", "یەکە", "یەکی", "یەک",
    "ێکی", "ێک", "کانی", "کان", "کەی", "کە",
    "انی", "یان", "ان", "ەوە", "وە", "یشی", "یش", "دا", "یە", "ەی", "ی", "ە",
)


def _is_latin(token: str) -> bool:
    return token.isascii()


class KurdishStemmer:
    def __init__(self, suffixes: tuple[str, ...] = SORANI_SUFFIXES, min_candidate_length: int = 2) -> None:
        self.suffixes = tuple(sorted(set(suffixes), key=len, reverse=True))
        self.min_candidate_length = min_candidate_length

    def candidates(self, token: str) -> list[str]:
        forms = [token]
        if _is_latin(token):
            return forms
        for suffix in self.suffixes:
            if token.endswith(suffix) and len(token) - len(suffix) >= self.min_candidate_length:
                stripped = token[: -len(suffix)]
                if stripped not in forms:
                    forms.append(stripped)
        return forms

    def stem(self, token: str, min_length: int = 3) -> str:
        if _is_latin(token):
            return token
        for suffix in self.suffixes:
            if token.endswith(suffix) and len(token) - len(suffix) >= min_length:
                return token[: -len(suffix)]
        return token

"""Lightweight script-based language identification for queries."""

from __future__ import annotations

LANGUAGE_NAMES = {
    "en": "English",
    "ckb": "Central Kurdish (Sorani, Arabic script)",
    "kmr": "Northern Kurdish (Kurmanji, Latin script)",
}

_KURMANJI_LETTERS = frozenset("êîûçşÊÎÛÇŞ")


def detect_language(text: str) -> str:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return "en"
    arabic = sum(1 for char in letters if "؀" <= char <= "ۿ" or "ݐ" <= char <= "ݿ")
    if arabic / len(letters) >= 0.5:
        return "ckb"
    if any(char in _KURMANJI_LETTERS for char in letters):
        return "kmr"
    return "en"

"""Orthographic normalisation for Central Kurdish (Sorani) text.

Sorani text found online mixes Arabic and Persian code points (ي/ك/ة), legacy
encodings of the vowel `ە` (heh + ZWNJ), diacritics, tatweel and several digit
systems. The normaliser maps all of them onto the standard Kurdish alphabet so
that the same word always has the same byte sequence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, fields
from typing import Any, Mapping

# Letter + combining "small v" written as two code points -> precomposed Kurdish letter.
_COMPOSITIONS: tuple[tuple[str, str], ...] = (
    ("وٚ", "ۆ"),  # و + ٚ -> ۆ
    ("یٚ", "ێ"),  # ی + ٚ -> ێ
    ("يٚ", "ێ"),  # ي + ٚ -> ێ
    ("لٚ", "ڵ"),  # ل + ٚ -> ڵ
    ("یٔ", "ئ"),  # ی + ٔ -> ئ
)

_LETTER_TABLE = str.maketrans({
    "ي": "ی",  # ي ARABIC YEH        -> ی
    "ى": "ی",  # ى ALEF MAKSURA      -> ی
    "ك": "ک",  # ك ARABIC KAF        -> ک
    "ة": "ە",  # ة TEH MARBUTA       -> ە
    "ۀ": "ە",  # ۀ HEH WITH YEH      -> ە
    "ھ": "ه",  # ھ HEH DOACHASHMEE   -> ه
    "أ": "ا",  # أ -> ا
    "إ": "ا",  # إ -> ا
    "ٱ": "ا",  # ٱ -> ا
    "آ": "ئا",  # آ -> ئا
    "٫": ".",       # Arabic decimal separator
    "٬": ",",       # Arabic thousands separator
})

_DIGIT_TABLE = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_INVISIBLE = re.compile("[​‌‍‎‏‪-‮⁦-⁩﻿­]")
_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
_TATWEEL = "ـ"
_FINAL_HEH = re.compile(r"(?<=\w)ه(?!\w)")
_HORIZONTAL_WS = re.compile(r"[^\S\n]+")


@dataclass(frozen=True)
class NormalizationConfig:
    unicode_nfkc: bool = True
    unify_arabic_letters: bool = True
    final_heh_to_ae: bool = True
    remove_diacritics: bool = True
    remove_tatweel: bool = True
    normalize_digits: bool = True
    lowercase_latin: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "NormalizationConfig":
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown normalization options: {sorted(unknown)}")
        return cls(**{key: bool(value) for key, value in data.items()})


class KurdishNormalizer:
    def __init__(self, config: NormalizationConfig | None = None) -> None:
        self.config = config or NormalizationConfig()

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        cfg = self.config
        if cfg.unicode_nfkc:
            text = unicodedata.normalize("NFKC", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if cfg.unify_arabic_letters:
            for source, target in _COMPOSITIONS:
                text = text.replace(source, target)
            text = text.translate(_LETTER_TABLE)
            # Legacy Sorani encodes the vowel ە as heh followed by a zero-width non-joiner.
            text = text.replace("ه‌", "ە")

        text = _INVISIBLE.sub("", text)
        if cfg.remove_diacritics:
            text = _DIACRITICS.sub("", text)
        if cfg.remove_tatweel:
            text = text.replace(_TATWEEL, "")
        if cfg.final_heh_to_ae:
            text = _FINAL_HEH.sub("ە", text)
        if cfg.normalize_digits:
            text = text.translate(_DIGIT_TABLE)
        if cfg.lowercase_latin:
            text = text.lower()

        text = _HORIZONTAL_WS.sub(" ", text)
        lines = (line.strip() for line in text.split("\n"))
        return "\n".join(line for line in lines if line)

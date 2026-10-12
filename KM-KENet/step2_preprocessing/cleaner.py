"""Noise removal and script/language filtering."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

_HTML_TAG = re.compile(r"<[^>]{1,500}>")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_CITATION = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")
_PAGE_MARKER = re.compile(r"^\s*(?:page|پەڕە|لاپەڕە)\s*[\d٠-٩]+\s*$", re.IGNORECASE | re.MULTILINE)
_CONTROL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")
_REPEATED_PUNCT = re.compile(r"([!?؟.,،؛;:\-_=*~])\1{2,}")
# Markdown found in KMC responses: "**bold**", "*   bullet", "- item", "1. item", "# heading", `code`.
_MD_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]*", re.MULTILINE)
_MD_BULLET = re.compile(r"^[ \t]*(?:[*\-+•▪●]|\d{1,3}[.)])[ \t]+", re.MULTILINE)
_MD_EMPHASIS = re.compile(r"\*{1,3}|_{2,3}|`+")
_REPLACEMENT_CHAR = "�"


@dataclass(frozen=True)
class CleaningConfig:
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_citation_marks: bool = True
    remove_page_markers: bool = True
    remove_markdown: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "CleaningConfig":
        data = dict(data or {})
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown cleaning options: {sorted(unknown)}")
        return cls(**{key: bool(value) for key, value in data.items()})


class TextCleaner:
    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config = config or CleaningConfig()

    def clean(self, text: str) -> str:
        if not text:
            return ""
        cfg = self.config
        if cfg.remove_html:
            text = _HTML_TAG.sub(" ", html.unescape(text))
        if cfg.remove_urls:
            text = _URL.sub(" ", text)
        if cfg.remove_emails:
            text = _EMAIL.sub(" ", text)
        if cfg.remove_citation_marks:
            text = _CITATION.sub(" ", text)
        if cfg.remove_page_markers:
            text = _PAGE_MARKER.sub("", text)
        if cfg.remove_markdown:
            # Bullets first (they start with "*"), then the remaining emphasis markers.
            text = _MD_HEADING.sub("", text)
            text = _MD_BULLET.sub("", text)
            text = _MD_EMPHASIS.sub("", text)
        text = _CONTROL.sub(" ", text).replace(_REPLACEMENT_CHAR, " ")
        text = _REPEATED_PUNCT.sub(r"\1", text)
        return text.strip()


_ARABIC_SCRIPT = re.compile(r"[ؠ-يٮ-ەۮۯۺ-ۿݐ-ݿ]")
_LATIN_SCRIPT = re.compile(r"[A-Za-zÀ-ɏ]")
_SORANI_MARKERS = frozenset("ڕڵێۆەڤ")
_KURMANJI_MARKERS = frozenset("êîûçşÊÎÛÇŞ")
_SUPPORTED_SCRIPTS = frozenset({"arabic", "latin"})


class KurdishScriptFilter:
    """Accepts text written mostly in an allowed script and containing Kurdish-specific letters.

    The Kurdish-letter check separates Sorani from Arabic/Persian text, which share
    the Arabic script but never use letters such as ڕ ڵ ێ ۆ ە.
    """

    def __init__(
        self,
        allowed_scripts: Sequence[str] = ("arabic",),
        min_script_ratio: float = 0.6,
        require_kurdish_letters: bool = True,
        min_kurdish_letter_ratio: float = 0.02,
    ) -> None:
        scripts = {s.lower() for s in allowed_scripts}
        unknown = scripts - _SUPPORTED_SCRIPTS
        if unknown or not scripts:
            raise ValueError(f"allowed_scripts must be a non-empty subset of {sorted(_SUPPORTED_SCRIPTS)}")
        self.allowed_scripts = frozenset(scripts)
        self.min_script_ratio = float(min_script_ratio)
        self.require_kurdish_letters = bool(require_kurdish_letters)
        self.min_kurdish_letter_ratio = float(min_kurdish_letter_ratio)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "KurdishScriptFilter":
        return cls(**dict(data or {}))

    def accepts(self, text: str) -> bool:
        letters = [char for char in text if char.isalpha()]
        if not letters:
            return False
        arabic = sum(1 for char in letters if _ARABIC_SCRIPT.match(char))
        latin = sum(1 for char in letters if _LATIN_SCRIPT.match(char))
        script, count, markers = (
            ("arabic", arabic, _SORANI_MARKERS) if arabic >= latin else ("latin", latin, _KURMANJI_MARKERS)
        )
        if script not in self.allowed_scripts or count == 0:
            return False
        if count / len(letters) < self.min_script_ratio:
            return False
        if self.require_kurdish_letters:
            kurdish = sum(1 for char in letters if char in markers)
            if kurdish / count < self.min_kurdish_letter_ratio:
                return False
        return True

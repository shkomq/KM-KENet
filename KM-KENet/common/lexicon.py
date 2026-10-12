"""Loading of the Kurdish medical lexicon and the English–Kurdish glossary."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from common.schema import ENTITY_TYPES


@dataclass(frozen=True)
class LexiconEntry:
    entry_id: str
    kurdish: str
    english: str
    entity_type: str
    aliases: tuple[str, ...] = ()

    @property
    def surface_forms(self) -> tuple[str, ...]:
        return (self.kurdish, *self.aliases)


def _data_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                yield line


def load_lexicon(path: Path) -> list[LexiconEntry]:
    """Read a TSV with columns: id, kurdish, english, type, aliases (`|`-separated)."""
    reader = csv.DictReader(_data_lines(path), delimiter="\t", quoting=csv.QUOTE_NONE)
    required = {"id", "kurdish", "english", "type"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{path}: missing lexicon columns {sorted(missing)}")

    entries: list[LexiconEntry] = []
    seen_ids: set[str] = set()
    for row_no, row in enumerate(reader, start=2):
        entry_id = (row.get("id") or "").strip()
        kurdish = (row.get("kurdish") or "").strip()
        english = (row.get("english") or "").strip()
        entity_type = (row.get("type") or "").strip().upper()
        if not entry_id or not kurdish or not english:
            raise ValueError(f"{path}: row {row_no} has an empty id/kurdish/english field")
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"{path}: row {row_no} has unknown entity type '{entity_type}'")
        if entry_id in seen_ids:
            raise ValueError(f"{path}: duplicate lexicon id '{entry_id}'")
        seen_ids.add(entry_id)
        aliases = tuple(alias.strip() for alias in (row.get("aliases") or "").split("|") if alias.strip())
        entries.append(LexiconEntry(entry_id, kurdish, english, entity_type, aliases))
    if not entries:
        raise ValueError(f"{path}: lexicon is empty")
    return entries


def load_glossary(path: Path) -> list[tuple[str, str]]:
    """Read a TSV of general (non-entity) English–Kurdish term pairs."""
    reader = csv.DictReader(_data_lines(path), delimiter="\t", quoting=csv.QUOTE_NONE)
    if not {"english", "kurdish"} <= set(reader.fieldnames or []):
        raise ValueError(f"{path}: glossary needs 'english' and 'kurdish' columns")
    pairs = []
    for row in reader:
        english = (row.get("english") or "").strip().lower()
        kurdish = (row.get("kurdish") or "").strip()
        if english and kurdish:
            pairs.append((english, kurdish))
    return pairs

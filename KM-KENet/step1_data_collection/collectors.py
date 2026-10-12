"""Document collectors for the Kurdish Medical Corpus.

Every collector yields `RawDocument` objects; the step's `main` module takes
care of filtering, de-duplication and persistence.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import unquote, urlparse
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    text: str
    source: str              # file path, URL or dataset id
    source_type: str         # kmc_json | kmc_books | local_files | web | huggingface
    title: str | None = None
    group: str | None = None  # originating book / article, used for leak-free splits
    url: str | None = None
    external_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseCollector(ABC):
    name: str = "base"

    @abstractmethod
    def collect(self) -> Iterator[RawDocument]:
        """Yield raw documents."""


def _first_present(record: dict[str, Any], fields: Sequence[str]) -> str:
    for name in fields:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_commented_json(path: Path) -> Any:
    """Load JSON that may start with a `/* ... */` header (as in the KMC release)."""
    text = path.read_text(encoding="utf-8-sig").lstrip()
    if text.startswith("/*"):
        end = text.find("*/")
        if end == -1:
            raise ValueError(f"{path}: unterminated /* comment header")
        text = text[end + 2 :]
    return json.loads(text)


def _source_name(url: str | None) -> str | None:
    """Source document name from a URL, e.g. '.../1 Medicine Book CKB/Book.pdf' -> 'Book'."""
    if not url:
        return None
    name = unquote(urlparse(url).path).rstrip("/").rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return stem.strip() or None


class KMCJsonCollector(BaseCollector):
    """Instruction/response records of the KMC corpus (one document per record).

    The record `id` (CKB-MED-...) is kept as `external_id`. The `group` used for
    leak-free splits is the source document title, falling back to the file
    name in `source_url` (about 2.4k KMC records have an empty title).
    """

    name = "kmc_json"

    def __init__(
        self,
        files: Sequence[Path],
        directories: Sequence[Path],
        instruction_fields: Sequence[str],
        response_fields: Sequence[str],
        include_instruction: bool = False,
        exclude_instruction_patterns: Sequence[str] = (),
    ) -> None:
        self.files = list(files)
        self.directories = list(directories)
        self.instruction_fields = list(instruction_fields)
        self.response_fields = list(response_fields)
        self.include_instruction = include_instruction
        self.exclude_instruction = (
            re.compile("|".join(f"(?:{pattern})" for pattern in exclude_instruction_patterns))
            if exclude_instruction_patterns
            else None
        )
        self.excluded = 0

    def _paths(self) -> Iterator[Path]:
        for path in self.files:
            if path.is_file():
                yield path
            else:
                logger.error(
                    "KMC JSON file not found: %s — download it with `python -m setup.download_assets --only dataset`",
                    path,
                )
        for directory in self.directories:
            if not directory.is_dir():
                logger.warning("KMC JSON directory not found: %s", directory)
                continue
            yield from sorted(directory.glob("*.json"))

    def collect(self) -> Iterator[RawDocument]:
        for path in self._paths():
            try:
                data = load_commented_json(path)
            except (OSError, ValueError) as exc:
                logger.error("Skipping %s: %s", path, exc)
                continue
            records = data.get("data", []) if isinstance(data, dict) else data
            if not isinstance(records, list):
                logger.error("Skipping %s: expected a list of records", path)
                continue

            logger.info("Reading %d records from %s", len(records), path.name)
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    continue
                response = _first_present(record, self.response_fields)
                if not response:
                    continue
                instruction = _first_present(record, self.instruction_fields)
                if self.exclude_instruction is not None and self.exclude_instruction.search(instruction):
                    self.excluded += 1
                    continue
                text = f"{instruction}\n{response}" if self.include_instruction and instruction else response
                document = record.get("document")
                document = document if isinstance(document, dict) else {}
                language = record.get("language")
                language = language if isinstance(language, dict) else {}
                title = (document.get("title") or "").strip() or None
                url = (document.get("source_url") or "").strip() or None
                yield RawDocument(
                    text=text,
                    source=str(path),
                    source_type=self.name,
                    title=title,
                    group=title or _source_name(url) or path.stem,
                    url=url,
                    external_id=str(record.get("id") or f"{path.stem}-{index}"),
                    metadata={
                        "instruction": instruction,
                        "publication_date": document.get("publication_date"),
                        "domain": document.get("domain"),
                        "language": language.get("code"),
                        "region": language.get("region"),
                    },
                )
        if self.excluded:
            logger.info("Skipped %d records whose instruction asks about the book, not medicine", self.excluded)


_PAGE_MARKER = re.compile(r"^\s*(?:page|پەڕە|لاپەڕە)\s*[\d٠-٩]+\s*$", re.IGNORECASE)
_EXTRACTION_HEADER = re.compile(r"^\s*here is the text extracted from the pdf pages:?\s*$", re.IGNORECASE)
_SENTENCE_END = (".", "!", "?", "؟", ":", "؛", "»", '"')


def join_wrapped_lines(lines: Sequence[str]) -> str:
    """Re-join lines broken by PDF layout; blank lines and sentence ends close a paragraph."""
    paragraphs: list[str] = []
    buffer: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            continue
        buffer.append(line)
        if line.endswith(_SENTENCE_END):
            paragraphs.append(" ".join(buffer))
            buffer = []
    if buffer:
        paragraphs.append(" ".join(buffer))
    return "\n".join(paragraphs)


class KMCBookTextCollector(BaseCollector):
    """Text extracted from the KMC medical books (one document per book)."""

    name = "kmc_books"

    def __init__(self, directories: Sequence[Path], join_lines: bool = True) -> None:
        self.directories = list(directories)
        self.join_lines = join_lines

    def collect(self) -> Iterator[RawDocument]:
        for directory in self.directories:
            if not directory.is_dir():
                logger.warning("Book text directory not found: %s", directory)
                continue
            paths = sorted(directory.glob("*.txt"))
            logger.info("Reading %d book text files from %s", len(paths), directory.name)
            for path in paths:
                try:
                    raw = path.read_text(encoding="utf-8-sig", errors="replace")
                except OSError as exc:
                    logger.error("Skipping %s: %s", path, exc)
                    continue
                lines = [
                    line
                    for line in raw.splitlines()
                    if not _PAGE_MARKER.match(line) and not _EXTRACTION_HEADER.match(line)
                ]
                text = join_wrapped_lines(lines) if self.join_lines else "\n".join(lines)
                yield RawDocument(
                    text=text,
                    source=str(path),
                    source_type=self.name,
                    title=path.stem,
                    group=path.stem,
                    external_id=path.stem,
                )


class LocalFileCollector(BaseCollector):
    """Arbitrary local .txt / .jsonl / .csv files."""

    name = "local_files"

    def __init__(
        self,
        directories: Sequence[Path],
        extensions: Sequence[str],
        text_field: str = "text",
        title_field: str = "title",
    ) -> None:
        self.directories = list(directories)
        self.extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
        self.text_field = text_field
        self.title_field = title_field

    def collect(self) -> Iterator[RawDocument]:
        for directory in self.directories:
            if not directory.is_dir():
                logger.warning("Local directory not found: %s", directory)
                continue
            for path in sorted(p for p in directory.rglob("*") if p.is_file()):
                suffix = path.suffix.lower()
                if suffix not in self.extensions:
                    continue
                try:
                    if suffix == ".txt":
                        yield from self._read_txt(path)
                    elif suffix == ".jsonl":
                        yield from self._read_jsonl(path)
                    elif suffix == ".csv":
                        yield from self._read_csv(path)
                except OSError as exc:
                    logger.error("Skipping %s: %s", path, exc)

    def _document(self, path: Path, text: str, title: str | None, index: int | None) -> RawDocument:
        external_id = path.stem if index is None else f"{path.stem}-{index}"
        return RawDocument(
            text=text,
            source=str(path),
            source_type=self.name,
            title=title or path.stem,
            group=path.stem,
            external_id=external_id,
        )

    def _read_txt(self, path: Path) -> Iterator[RawDocument]:
        yield self._document(path, path.read_text(encoding="utf-8-sig", errors="replace"), None, None)

    def _read_jsonl(self, path: Path) -> Iterator[RawDocument]:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("%s:%d: invalid JSON line skipped", path, index + 1)
                    continue
                text = record.get(self.text_field) if isinstance(record, dict) else None
                if isinstance(text, str) and text.strip():
                    yield self._document(path, text, record.get(self.title_field), index)

    def _read_csv(self, path: Path) -> Iterator[RawDocument]:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            if self.text_field not in (reader.fieldnames or []):
                logger.warning("%s has no '%s' column; skipped", path, self.text_field)
                return
            for index, row in enumerate(reader):
                text = row.get(self.text_field) or ""
                if text.strip():
                    yield self._document(path, text, row.get(self.title_field), index)


class WebCollector(BaseCollector):
    """Polite crawler for a fixed list of health-portal URLs."""

    name = "web"
    _DROP_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "aside", "form", "iframe")

    def __init__(
        self,
        urls: Sequence[str],
        delay_seconds: float = 1.5,
        timeout_seconds: float = 20.0,
        user_agent: str = "KM-KENet-Research-Crawler/1.0",
        respect_robots_txt: bool = True,
    ) -> None:
        self.urls = list(urls)
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent
        self.respect_robots_txt = respect_robots_txt
        self._robots: dict[str, RobotFileParser | None] = {}

    def _allowed(self, url: str) -> bool:
        if not self.respect_robots_txt:
            return True
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            parser = RobotFileParser()
            parser.set_url(f"{base}/robots.txt")
            try:
                parser.read()
                self._robots[base] = parser
            except OSError as exc:
                logger.warning("Could not read robots.txt for %s (%s); assuming allowed", base, exc)
                self._robots[base] = None
        parser = self._robots[base]
        return parser is None or parser.can_fetch(self.user_agent, url)

    def _extract(self, html: str) -> tuple[str, str | None]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(self._DROP_TAGS):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else None
        blocks = [
            node.get_text(" ", strip=True)
            for node in soup.find_all(["h1", "h2", "h3", "p", "li"])
        ]
        return "\n".join(block for block in blocks if block), title

    def collect(self) -> Iterator[RawDocument]:
        if not self.urls:
            return
        import requests

        session = requests.Session()
        session.headers["User-Agent"] = self.user_agent
        for index, url in enumerate(self.urls):
            if index:
                time.sleep(self.delay_seconds)
            if not self._allowed(url):
                logger.warning("robots.txt disallows %s; skipped", url)
                continue
            try:
                response = session.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("Failed to fetch %s: %s", url, exc)
                continue
            if not response.encoding or response.encoding.lower() == "iso-8859-1":
                response.encoding = response.apparent_encoding or "utf-8"
            text, title = self._extract(response.text)
            if text:
                yield RawDocument(
                    text=text,
                    source=url,
                    source_type=self.name,
                    title=title,
                    group=urlparse(url).netloc,
                    url=url,
                    external_id=url,
                )


class HuggingFaceCollector(BaseCollector):
    """Documents streamed from a Hugging Face dataset."""

    name = "huggingface"

    def __init__(
        self,
        dataset: str,
        config_name: str | None,
        split: str,
        text_field: str,
        title_field: str | None,
        max_documents: int | None,
        streaming: bool = True,
    ) -> None:
        self.dataset = dataset
        self.config_name = config_name
        self.split = split
        self.text_field = text_field
        self.title_field = title_field
        self.max_documents = max_documents
        self.streaming = streaming

    def collect(self) -> Iterator[RawDocument]:
        from datasets import load_dataset

        rows = load_dataset(self.dataset, self.config_name, split=self.split, streaming=self.streaming)
        for index, row in enumerate(rows):
            if self.max_documents is not None and index >= self.max_documents:
                break
            text = row.get(self.text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            title = row.get(self.title_field) if self.title_field else None
            yield RawDocument(
                text=text,
                source=f"hf://{self.dataset}/{self.config_name or ''}",
                source_type=self.name,
                title=title,
                group=title or f"{self.dataset}-{index}",
                external_id=str(row.get("id", index)),
            )


class MedicalRelevanceFilter:
    """Keeps documents mentioning at least `min_hits` distinct medical keywords."""

    def __init__(self, keywords: Sequence[str], min_hits: int = 2) -> None:
        self.keywords = tuple(dict.fromkeys(k.strip() for k in keywords if k.strip()))
        self.min_hits = max(1, int(min_hits))

    def is_relevant(self, text: str) -> bool:
        hits = 0
        for keyword in self.keywords:
            if keyword in text:
                hits += 1
                if hits >= self.min_hits:
                    return True
        return False

"""File I/O helpers (JSON / JSON Lines) with atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@contextmanager
def atomic_open(path: str | os.PathLike[str]) -> Iterator[TextIO]:
    """Write to a temporary file and move it into place only on success."""
    target = Path(path)
    ensure_dir(target.parent)
    tmp = target.with_name(target.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            yield handle
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON line") from exc


def write_jsonl(path: str | os.PathLike[str], records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with atomic_open(path) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def read_json(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | os.PathLike[str], obj: Any, indent: int = 2) -> None:
    with atomic_open(path) as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=indent, default=str)
        handle.write("\n")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def require_file(path: Path, hint: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}. {hint}")
    return path

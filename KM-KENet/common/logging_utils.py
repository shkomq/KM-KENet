"""Logging setup shared by all pipeline steps."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_NOISY_LOGGERS = ("urllib3", "filelock", "httpx", "httpcore", "matplotlib", "PIL", "fsspec")


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(str(level).upper())

    formatter = logging.Formatter(_FORMAT, _DATE_FORMAT)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


@contextmanager
def log_duration(logger: logging.Logger, task: str) -> Iterator[None]:
    start = time.perf_counter()
    logger.info("%s ...", task)
    try:
        yield
    finally:
        logger.info("%s finished in %.1fs", task, time.perf_counter() - start)

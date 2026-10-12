"""Step 1 — Data Collection.

Collects documents from the Kurdish Medical Corpus (KMC) and optional extra
sources, removes exact duplicates and writes `interim_dir/corpus.jsonl`.

    python -m step1_data_collection.main --config configs/config.yaml
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from common.cli import build_arg_parser, init_step
from common.config import PipelineConfig
from common.io_utils import sha1_text, write_json, write_jsonl
from common.logging_utils import log_duration
from step1_data_collection.collectors import (
    BaseCollector,
    HuggingFaceCollector,
    KMCBookTextCollector,
    KMCJsonCollector,
    LocalFileCollector,
    MedicalRelevanceFilter,
    WebCollector,
)

STEP_NAME = "step1_data_collection"


def build_collectors(cfg: PipelineConfig) -> list[tuple[BaseCollector, bool]]:
    """Return enabled collectors paired with their `apply_medical_filter` flag."""
    section = cfg.section("data_collection")
    corpus_root = cfg.path("corpus_root")
    collectors: list[tuple[BaseCollector, bool]] = []

    kmc = section.get("kmc_json", {})
    if kmc.get("enabled", False):
        collectors.append((
            KMCJsonCollector(
                files=[corpus_root / name for name in kmc.get("files", [])],
                directories=[corpus_root / name for name in kmc.get("directories", [])],
                instruction_fields=kmc.get("instruction_fields", ["instruction", "question"]),
                response_fields=kmc.get("response_fields", ["response", "answer"]),
                include_instruction=bool(kmc.get("include_instruction", False)),
                exclude_instruction_patterns=kmc.get("exclude_instruction_patterns") or [],
            ),
            bool(kmc.get("apply_medical_filter", False)),
        ))

    books = section.get("kmc_books", {})
    if books.get("enabled", False):
        collectors.append((
            KMCBookTextCollector(
                directories=[corpus_root / name for name in books.get("directories", [])],
                join_lines=bool(books.get("join_wrapped_lines", True)),
            ),
            bool(books.get("apply_medical_filter", False)),
        ))

    local = section.get("local_files", {})
    if local.get("enabled", False):
        collectors.append((
            LocalFileCollector(
                directories=[cfg.resolve(d) for d in local.get("directories", [])],
                extensions=local.get("extensions", [".txt"]),
                text_field=local.get("text_field", "text"),
                title_field=local.get("title_field", "title"),
            ),
            bool(local.get("apply_medical_filter", False)),
        ))

    web = section.get("web", {})
    if web.get("enabled", False):
        collectors.append((
            WebCollector(
                urls=web.get("urls", []),
                delay_seconds=float(web.get("request_delay_seconds", 1.5)),
                timeout_seconds=float(web.get("timeout_seconds", 20)),
                user_agent=str(web.get("user_agent", "KM-KENet-Research-Crawler/1.0")),
                respect_robots_txt=bool(web.get("respect_robots_txt", True)),
            ),
            bool(web.get("apply_medical_filter", True)),
        ))

    hf = section.get("huggingface", {})
    if hf.get("enabled", False):
        max_documents = hf.get("max_documents")
        collectors.append((
            HuggingFaceCollector(
                dataset=hf["dataset"],
                config_name=hf.get("config_name"),
                split=hf.get("split", "train"),
                text_field=hf.get("text_field", "text"),
                title_field=hf.get("title_field"),
                max_documents=int(max_documents) if max_documents is not None else None,
                streaming=bool(hf.get("streaming", True)),
            ),
            bool(hf.get("apply_medical_filter", True)),
        ))
    return collectors


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("Step 1 - collect the Kurdish Medical Corpus")
    args = parser.parse_args(argv)
    cfg, logger = init_step(args, STEP_NAME)
    section = cfg.section("data_collection")

    collectors = build_collectors(cfg)
    if not collectors:
        logger.error("No data source is enabled in 'data_collection'")
        return 1

    filter_cfg = section.get("medical_filter", {})
    relevance = MedicalRelevanceFilter(filter_cfg.get("keywords", []), filter_cfg.get("min_keyword_hits", 2))
    min_chars = int(section.get("min_document_chars", 40))
    collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    stats: Counter[str] = Counter()
    seen: set[str] = set()
    used_ids: set[str] = set()
    records: list[dict[str, Any]] = []

    with log_duration(logger, "Collecting documents"):
        for collector, apply_filter in collectors:
            for doc in collector.collect():
                stats[f"{collector.name}.seen"] += 1
                text = doc.text.strip()
                if len(text) < min_chars:
                    stats[f"{collector.name}.too_short"] += 1
                    continue
                if apply_filter and not relevance.is_relevant(text):
                    stats[f"{collector.name}.not_medical"] += 1
                    continue
                digest = sha1_text(" ".join(text.split()))
                if digest in seen:
                    stats[f"{collector.name}.duplicate"] += 1
                    continue
                seen.add(digest)
                stats[f"{collector.name}.kept"] += 1
                # Keep the KMC record id (CKB-MED-...) so results stay traceable to the dataset.
                doc_id = doc.external_id if doc.external_id and doc.external_id not in used_ids else None
                doc_id = doc_id or f"{doc.source_type}-{digest[:16]}"
                used_ids.add(doc_id)
                records.append({
                    "doc_id": doc_id,
                    "source_type": doc.source_type,
                    "source": doc.source,
                    "external_id": doc.external_id,
                    "title": doc.title,
                    "group": doc.group or doc.title or doc.source,
                    "url": doc.url,
                    "text": text,
                    "metadata": doc.metadata,
                    "collected_at": collected_at,
                })

    if not records:
        logger.error("No documents were collected; check 'paths.corpus_root' and the source settings")
        return 1

    output_path = cfg.path("interim_dir") / "corpus.jsonl"
    write_jsonl(output_path, records)
    summary = {
        "documents": len(records),
        "characters": sum(len(r["text"]) for r in records),
        "groups": len({r["group"] for r in records}),
        "per_source": dict(sorted(stats.items())),
    }
    write_json(cfg.path("interim_dir") / "collection_stats.json", summary)
    logger.info("Wrote %d documents (%d source groups) to %s", summary["documents"], summary["groups"], output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

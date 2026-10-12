"""Step 2 — Data Pre-processing.

Cleaning -> normalization -> sentence splitting -> script filtering ->
tokenization -> de-duplication. Writes:

* `processed_dir/documents.jsonl` — cleaned, normalised documents
* `processed_dir/sentences.jsonl` — one tokenised sentence per line

    python -m step2_preprocessing.main --config configs/config.yaml
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import Any, Iterator, Sequence

from common.cli import build_arg_parser, init_step
from common.io_utils import read_jsonl, require_file, sha1_text, write_json, write_jsonl
from common.logging_utils import log_duration
from step2_preprocessing.cleaner import CleaningConfig, KurdishScriptFilter, TextCleaner
from step2_preprocessing.text_tools import build_text_tools

STEP_NAME = "step2_preprocessing"


def chunk_tokens(tokens: Sequence[str], max_tokens: int) -> Iterator[list[str]]:
    """Split overly long sentences into consecutive windows of at most `max_tokens`."""
    for start in range(0, len(tokens), max_tokens):
        yield list(tokens[start : start + max_tokens])


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("Step 2 - clean, normalise and tokenise the corpus")
    args = parser.parse_args(argv)
    cfg, logger = init_step(args, STEP_NAME)
    section = cfg.section("preprocessing")

    corpus_path = require_file(cfg.path("interim_dir") / "corpus.jsonl", "Run step 1 first.")
    cleaner = TextCleaner(CleaningConfig.from_dict(section.get("cleaning")))
    script_filter = KurdishScriptFilter.from_dict(section.get("script_filter"))
    tools = build_text_tools(cfg)

    min_tokens = int(section.get("min_sentence_tokens", 4))
    max_tokens = int(section.get("max_sentence_tokens", 96))
    if max_tokens < min_tokens:
        raise ValueError("max_sentence_tokens must be >= min_sentence_tokens")
    deduplicate = bool(section.get("deduplicate_sentences", True))

    stats: Counter[str] = Counter()
    seen: set[str] = set()
    documents: list[dict[str, Any]] = []
    sentences: list[dict[str, Any]] = []

    with log_duration(logger, "Pre-processing documents"):
        for doc in read_jsonl(corpus_path):
            stats["documents.in"] += 1
            text = tools.normalizer.normalize(cleaner.clean(doc["text"]))
            if not text:
                stats["documents.empty_after_cleaning"] += 1
                continue

            kept_in_doc = 0
            for sentence in tools.tokenizer.split_sentences(text):
                stats["sentences.in"] += 1
                if not script_filter.accepts(sentence):
                    stats["sentences.rejected_script"] += 1
                    continue
                tokens = tools.tokenizer.tokenize(sentence)
                if len(tokens) < min_tokens:
                    stats["sentences.too_short"] += 1
                    continue

                chunks = list(chunk_tokens(tokens, max_tokens))
                if len(chunks) > 1:
                    stats["sentences.split_long"] += 1
                for chunk in chunks:
                    if len(chunk) < min_tokens:
                        continue
                    chunk_text = sentence if len(chunks) == 1 else " ".join(chunk)
                    if deduplicate:
                        key = sha1_text(" ".join(chunk))
                        if key in seen:
                            stats["sentences.duplicate"] += 1
                            continue
                        seen.add(key)
                    sentences.append({
                        "sentence_id": f"{doc['doc_id']}-s{kept_in_doc:05d}",
                        "doc_id": doc["doc_id"],
                        "group": doc.get("group") or doc["doc_id"],
                        "text": chunk_text,
                        "tokens": chunk,
                    })
                    kept_in_doc += 1

            if kept_in_doc:
                stats["documents.out"] += 1
                documents.append({
                    "doc_id": doc["doc_id"],
                    "group": doc.get("group") or doc["doc_id"],
                    "title": doc.get("title"),
                    "source_type": doc.get("source_type"),
                    "url": doc.get("url"),
                    "text": text,
                    "num_sentences": kept_in_doc,
                })

    if not sentences:
        logger.error("No sentences survived pre-processing; check the script filter settings")
        return 1

    processed_dir = cfg.path("processed_dir")
    write_jsonl(processed_dir / "documents.jsonl", documents)
    write_jsonl(processed_dir / "sentences.jsonl", sentences)
    summary = {
        **dict(sorted(stats.items())),
        "sentences.out": len(sentences),
        "tokens.out": sum(len(s["tokens"]) for s in sentences),
    }
    write_json(processed_dir / "preprocessing_stats.json", summary)
    logger.info(
        "Kept %d documents, %d sentences, %d tokens",
        len(documents), summary["sentences.out"], summary["tokens.out"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

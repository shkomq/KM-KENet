"""Step 6 — Cross-Lingual Query Translation and Retrieval.

Answers questions (e.g. in English) with evidence retrieved from the Kurdish
medical knowledge graph and corpus, using the Aya multilingual model for
translation and grounded answer generation.

    python -m step6_cross_lingual_retrieval.main --query "What are the treatments for Kurdish-specific health issues?"
    python -m step6_cross_lingual_retrieval.main --interactive
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from common.cli import build_arg_parser, init_step
from common.config import PipelineConfig
from common.io_utils import read_json, read_jsonl, require_file, write_json
from common.lexicon import load_glossary, load_lexicon
from common.logging_utils import log_duration
from common.utils import resolve_device
from step2_preprocessing.text_tools import build_text_tools, load_stopwords
from step6_cross_lingual_retrieval.answer import MultilingualAssistant
from step6_cross_lingual_retrieval.aya_client import build_chat_client
from step6_cross_lingual_retrieval.pipeline import CrossLingualQA, QAResult
from step6_cross_lingual_retrieval.retrievers import (
    BM25Retriever,
    DenseEncoder,
    DenseRetriever,
    KnowledgeGraphRetriever,
    Passage,
)
from step6_cross_lingual_retrieval.translator import LexiconTranslator

STEP_NAME = "step6_cross_lingual_retrieval"


def load_passages(path: Path, allowed_ids: set[str] | None) -> list[Passage]:
    return [
        Passage(record["sentence_id"], record["doc_id"], record["text"], tuple(record["tokens"]))
        for record in read_jsonl(path)
        if allowed_ids is None or record["sentence_id"] in allowed_ids
    ]


def build_pipeline(cfg: PipelineConfig, logger: logging.Logger) -> CrossLingualQA:
    section = cfg.section("retrieval")
    tools = build_text_tools(cfg)
    lexicon = load_lexicon(cfg.path("lexicon"))
    stopwords = load_stopwords(cfg.path("stopwords"), tools)

    encoder = None
    if section.get("use_dense_retrieval", True):
        device = resolve_device(cfg.device)
        try:
            encoder = DenseEncoder(
                str(section.get("embedding_model", "intfloat/multilingual-e5-base")),
                cfg.path("cache_dir"),
                device=str(device),
            )
        except Exception as exc:
            logger.warning("Dense retrieval disabled (%s); using BM25 and lexical KG matching only", exc)

    kg_payload = read_json(require_file(cfg.path("kg_dir") / "knowledge_graph.json", "Run step 5 first."))
    with log_duration(logger, "Indexing the knowledge graph"):
        kg = KnowledgeGraphRetriever.from_payload(kg_payload, tools, encoder)
    logger.info("Knowledge graph: %d nodes, %d edges", len(kg.nodes), len(kg.edges))

    source = str(section.get("passage_source", "kg_evidence"))
    if source not in ("kg_evidence", "all_sentences"):
        raise ValueError("retrieval.passage_source must be 'kg_evidence' or 'all_sentences'")
    sentences_path = require_file(cfg.path("processed_dir") / "sentences.jsonl", "Run step 2 first.")
    allowed = kg.evidence_sentence_ids if source == "kg_evidence" else None
    passages = load_passages(sentences_path, allowed)
    logger.info("Passage index: %d sentences (%s)", len(passages), source)

    with log_duration(logger, "Building passage indexes"):
        bm25 = BM25Retriever(passages, tools, stopwords)
        dense = DenseRetriever(passages, encoder) if encoder is not None else None

    translator = LexiconTranslator(lexicon, load_glossary(cfg.path("glossary")), tools)
    assistant = MultilingualAssistant(build_chat_client(section.get("aya", {}) or {}), translator)
    logger.info("Aya backend: %s", assistant.backend)

    return CrossLingualQA(
        assistant,
        passages,
        bm25,
        dense,
        kg,
        top_k_passages=int(section.get("top_k_passages", 8)),
        candidate_pool=int(section.get("candidate_pool", 100)),
        top_k_nodes=int(section.get("top_k_nodes", 12)),
        max_facts=int(section.get("max_facts", 25)),
        rrf_k=int(section.get("rrf_k", 60)),
    )


def print_result(result: QAResult) -> None:
    rule = "=" * 80
    print(rule)
    print(f"Question ({result.language}): {result.question}")
    print(f"English query : {result.query_en}")
    print(f"Kurdish query : {result.query_ckb}   [{result.translation_method}]")
    print(rule)
    print(result.answer)
    print(rule)
    if result.facts:
        print("Knowledge-graph facts:")
        for fact in result.facts:
            print(f"  {fact.render()}")
    print(rule, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("Step 6 - cross-lingual question answering with Aya")
    parser.add_argument("--query", action="append", default=[], help="Question to answer (repeatable)")
    parser.add_argument("--interactive", action="store_true", help="Ask questions in a loop")
    args = parser.parse_args(argv)
    cfg, logger = init_step(args, STEP_NAME)

    questions = list(args.query)
    if not questions and not args.interactive:
        questions = [str(cfg.section("retrieval").get("default_query"))]

    qa = build_pipeline(cfg, logger)
    answers_dir = cfg.path("answers_dir")
    results = []
    for question in questions:
        result = qa.answer(question)
        print_result(result)
        results.append(result.to_dict())

    if args.interactive:
        print("Interactive mode — type a question (empty line to quit).")
        while True:
            try:
                question = input("\n> ").strip()
            except EOFError:
                break
            if not question:
                break
            result = qa.answer(question)
            print_result(result)
            results.append(result.to_dict())

    if results:
        output = answers_dir / f"answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        write_json(output, results)
        logger.info("Saved %d answer(s) to %s", len(results), output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Step 5 — Knowledge Graph Construction.

Runs the trained KM-KENet model over the pre-processed corpus, optionally
merges the weakly labelled triples, canonicalises entities against the
lexicon and exports a structured medical knowledge graph.

    python -m step5_knowledge_graph.main --config configs/config.yaml
"""

from __future__ import annotations

import itertools
import sys

from common.cli import build_arg_parser, init_step
from common.io_utils import read_jsonl, require_file, write_json
from common.lexicon import load_lexicon
from common.logging_utils import log_duration
from common.utils import resolve_device
from step2_preprocessing.text_tools import build_text_tools
from step3_weak_supervision.labeling_functions.dictionary import LexiconMatcher
from step5_knowledge_graph.canonicalizer import EntityCanonicalizer
from step5_knowledge_graph.exporters import (
    export_cypher,
    export_graphml,
    export_json,
    export_triples_csv,
    export_turtle,
    export_visualization,
    graph_statistics,
)
from step5_knowledge_graph.extractor import extract_from_weak_labels, extract_with_model
from step5_knowledge_graph.graph_builder import KnowledgeGraphBuilder

STEP_NAME = "step5_knowledge_graph"


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("Step 5 - build the medical knowledge graph")
    args = parser.parse_args(argv)
    cfg, logger = init_step(args, STEP_NAME)
    section = cfg.section("knowledge_graph")

    tools = build_text_tools(cfg)
    matcher = LexiconMatcher(load_lexicon(cfg.path("lexicon")), tools)
    builder = KnowledgeGraphBuilder(
        EntityCanonicalizer(matcher, tools.stemmer),
        min_entity_confidence=float(section.get("min_entity_confidence", 0.5)),
        min_relation_confidence=float(section.get("min_relation_confidence", 0.5)),
        max_evidence_per_edge=int(section.get("max_evidence_per_edge", 10)),
        max_sentence_ids_per_node=int(section.get("max_sentence_ids_per_node", 200)),
    )
    sources: list[str] = []

    if section.get("use_model", True):
        from step4_model_training.predictor import KMKENetPredictor

        sentences_path = require_file(cfg.path("processed_dir") / "sentences.jsonl", "Run step 2 first.")
        limit = section.get("max_sentences")
        sentences = list(itertools.islice(read_jsonl(sentences_path), int(limit) if limit else None))
        device = resolve_device(cfg.device)
        predictor = KMKENetPredictor(cfg.path("model_dir"), device, batch_size=int(section.get("batch_size", 64)))
        with log_duration(logger, f"Extracting knowledge from {len(sentences)} sentences on {device}"):
            builder.add(extract_with_model(predictor, sentences))
        sources.append("km-kenet")
        del sentences, predictor

    if section.get("merge_weak_labels", True):
        weak_path = require_file(cfg.path("weak_dir") / "weak_labels_full.jsonl", "Run step 3 first.")
        with log_duration(logger, "Merging weakly labelled triples"):
            builder.add(extract_from_weak_labels(read_jsonl(weak_path)))
        sources.append("weak-supervision")

    if not sources:
        logger.error("Both knowledge_graph.use_model and merge_weak_labels are disabled")
        return 1

    graph = builder.build(min_triple_support=int(section.get("min_triple_support", 1)))
    stats = graph_statistics(graph)
    logger.info("Knowledge graph: %d nodes, %d edges", stats["nodes"], stats["edges"])
    logger.info("Relation types: %s", stats["relation_types"])
    if stats["edges"] == 0:
        logger.warning("The knowledge graph has no edges; consider lowering min_triple_support")

    kg_dir = cfg.path("kg_dir")
    exports = section.get("exports", {}) or {}
    metadata = {"name": "KM-KENet Kurdish Medical Knowledge Graph", "language": "ckb", "sources": sources}
    # The JSON export is always written: step 6 retrieves from it.
    export_json(graph, kg_dir / "knowledge_graph.json", metadata)
    if exports.get("graphml", True):
        export_graphml(graph, kg_dir / "knowledge_graph.graphml")
    if exports.get("csv", True):
        export_triples_csv(graph, kg_dir / "triples.csv")
    if exports.get("turtle", True):
        export_turtle(graph, kg_dir / "knowledge_graph.ttl")
    if exports.get("cypher", True):
        export_cypher(graph, kg_dir / "neo4j_import.cypher")
    if exports.get("visualization", True):
        export_visualization(graph, kg_dir / "knowledge_graph.png", int(section.get("visualization_max_nodes", 80)))
    write_json(kg_dir / "kg_stats.json", stats)
    logger.info("Exports written to %s", kg_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

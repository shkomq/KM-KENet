"""Run a SPARQL query against the exported knowledge graph (RDF/Turtle) with rdflib.

    python -m step5_knowledge_graph.sparql                         # first query in queries/example_queries.sparql
    python -m step5_knowledge_graph.sparql --query-file my.sparql
    python -m step5_knowledge_graph.sparql --query "SELECT ?s ?o WHERE { ?s <https://github.com/shkomq/SciKG-LLM/kmkg#TREATS> ?o } LIMIT 10"
"""

from __future__ import annotations

import sys
from pathlib import Path

from common.cli import build_arg_parser, init_step
from common.config import PROJECT_ROOT
from common.io_utils import require_file

STEP_NAME = "step5_sparql"
DEFAULT_QUERY_FILE = PROJECT_ROOT / "queries" / "example_queries.sparql"


def first_query(text: str) -> str:
    """The PREFIX lines plus the first uncommented SELECT/ASK/CONSTRUCT block of a .sparql file."""
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("Query the KM-KENet knowledge graph with SPARQL (rdflib)")
    parser.add_argument("--query", default=None, help="SPARQL query string")
    parser.add_argument("--query-file", type=Path, default=DEFAULT_QUERY_FILE)
    args = parser.parse_args(argv)
    cfg, logger = init_step(args, STEP_NAME)

    try:
        import rdflib
    except ImportError:
        logger.error("rdflib is not installed: pip install rdflib")
        return 1
    ttl = require_file(cfg.path("kg_dir") / "knowledge_graph.ttl", "Run step 5 first.")
    graph = rdflib.Graph()
    graph.parse(ttl, format="turtle")
    logger.info("Loaded %d RDF triples from %s", len(graph), ttl)

    query = args.query or first_query(args.query_file.read_text(encoding="utf-8"))
    rows = list(graph.query(query))
    for row in rows:
        print(" | ".join("" if value is None else str(value) for value in row))
    logger.info("%d result row(s)", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())

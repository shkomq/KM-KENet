"""Knowledge graph exporters: JSON, GraphML, CSV triples, RDF/Turtle (SPARQL), Neo4j Cypher and a PNG preview."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from common.io_utils import atomic_open, write_json

logger = logging.getLogger(__name__)

TYPE_COLORS = {
    "DISEASE": "#d62728",
    "SYMPTOM": "#ff7f0e",
    "DRUG": "#2ca02c",
    "DOSAGE": "#17becf",
    "PROCEDURE": "#9467bd",
    "BODY_PART": "#1f77b4",
}


def graph_statistics(graph: nx.MultiDiGraph) -> dict[str, Any]:
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_types": dict(Counter(data["type"] for _, data in graph.nodes(data=True)).most_common()),
        "relation_types": dict(Counter(key for _, _, key in graph.edges(keys=True)).most_common()),
        "lexicon_linked_nodes": sum(1 for _, data in graph.nodes(data=True) if data.get("lexicon_id")),
    }


def export_json(graph: nx.MultiDiGraph, path: Path, metadata: dict[str, Any]) -> None:
    payload = {
        "metadata": {
            **metadata,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "statistics": graph_statistics(graph),
        },
        "nodes": [{"id": node_id, **data} for node_id, data in graph.nodes(data=True)],
        "edges": [
            {"head": head, "tail": tail, **data}
            for head, tail, data in graph.edges(data=True)
        ],
    }
    write_json(path, payload)


def _scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict, set, tuple)):
        return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
    return value


def export_graphml(graph: nx.MultiDiGraph, path: Path) -> None:
    """GraphML only supports scalar attributes, so lists are JSON-encoded."""
    flat = nx.MultiDiGraph()
    for node_id, data in graph.nodes(data=True):
        flat.add_node(node_id, **{key: _scalar(value) for key, value in data.items()})
    for head, tail, key, data in graph.edges(keys=True, data=True):
        flat.add_edge(head, tail, key=key, **{k: _scalar(v) for k, v in data.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(flat, path, encoding="utf-8")


def export_triples_csv(graph: nx.MultiDiGraph, path: Path) -> None:
    columns = [
        "head_id", "head_type", "head_label_ckb", "head_label_en", "relation",
        "tail_id", "tail_type", "tail_label_ckb", "tail_label_en", "support", "confidence", "extractors",
    ]
    with atomic_open(path) as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for head, tail, data in sorted(graph.edges(data=True), key=lambda e: -e[2]["support"]):
            h, t = graph.nodes[head], graph.nodes[tail]
            writer.writerow([
                head, h["type"], h["label_ckb"], h["label_en"], data["relation"],
                tail, t["type"], t["label_ckb"], t["label_en"], data["support"], data["confidence"],
                "|".join(data["extractors"]),
            ])


def _cypher_string(value: Any) -> str:
    text = str(value).replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
    return f"'{text}'"


def _cypher_label(entity_type: str) -> str:
    return "".join(part.capitalize() for part in entity_type.split("_"))


def export_cypher(graph: nx.MultiDiGraph, path: Path) -> None:
    """Idempotent Neo4j import script (MERGE statements)."""
    with atomic_open(path) as handle:
        handle.write("// KM-KENet medical knowledge graph — Neo4j import script\n")
        handle.write("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;\n")
        for node_id, data in graph.nodes(data=True):
            handle.write(
                f"MERGE (n:Entity:{_cypher_label(data['type'])} {{id: {_cypher_string(node_id)}}}) "
                f"SET n.type = {_cypher_string(data['type'])}, "
                f"n.label_ckb = {_cypher_string(data['label_ckb'])}, "
                f"n.label_en = {_cypher_string(data['label_en'])}, "
                f"n.mention_count = {int(data['mention_count'])};\n"
            )
        for head, tail, data in graph.edges(data=True):
            handle.write(
                f"MATCH (h:Entity {{id: {_cypher_string(head)}}}), (t:Entity {{id: {_cypher_string(tail)}}}) "
                f"MERGE (h)-[r:{data['relation']}]->(t) "
                f"SET r.support = {int(data['support'])}, r.confidence = {float(data['confidence'])};\n"
            )


KMKG_NS = "https://github.com/shkomq/SciKG-LLM/kmkg#"


def _turtle_literal(value: Any, language: str | None = None) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")
    return f'"{text}"@{language}' if language else f'"{text}"'


def _turtle_local(node_id: str) -> str:
    """Safe local name for a node id (non-ASCII letters are allowed in Turtle prefixed names, `:` is not)."""
    import re

    return "e_" + re.sub(r"[^\w]", "_", node_id)


def export_turtle(graph: nx.MultiDiGraph, path: Path) -> None:
    """RDF/Turtle export so the graph can be loaded into any triple store and queried with SPARQL.

    Every triple is also reified as a `kmkg:Statement` carrying its support and confidence.
    """
    with atomic_open(path) as handle:
        handle.write(f"@prefix kmkg: <{KMKG_NS}> .\n")
        handle.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n")
        handle.write("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n")
        for node_id, data in graph.nodes(data=True):
            lines = [f"kmkg:{_turtle_local(node_id)} a kmkg:{_cypher_label(data['type'])}"]
            lines.append(f"rdfs:label {_turtle_literal(data['label_ckb'], 'ckb')}")
            if data.get("label_en"):
                lines.append(f"rdfs:label {_turtle_literal(data['label_en'], 'en')}")
            lines.append(f"kmkg:entityType {_turtle_literal(data['type'])}")
            lines.append(f'kmkg:mentionCount "{int(data["mention_count"])}"^^xsd:integer')
            handle.write(" ;\n    ".join(lines) + " .\n")
        handle.write("\n")
        for index, (head, tail, data) in enumerate(graph.edges(data=True)):
            h, t, relation = _turtle_local(head), _turtle_local(tail), data["relation"]
            handle.write(f"kmkg:{h} kmkg:{relation} kmkg:{t} .\n")
            handle.write(
                f"kmkg:stmt_{index} a kmkg:Statement ; kmkg:subject kmkg:{h} ; kmkg:predicate kmkg:{relation} ; "
                f'kmkg:object kmkg:{t} ; kmkg:support "{int(data["support"])}"^^xsd:integer ; '
                f'kmkg:confidence "{float(data["confidence"]):.4f}"^^xsd:decimal .\n'
            )


def export_visualization(graph: nx.MultiDiGraph, path: Path, max_nodes: int = 80) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping the visualization")
        return
    if graph.number_of_nodes() == 0:
        logger.warning("Empty graph; skipping the visualization")
        return

    degree = dict(graph.degree())
    top = sorted(degree, key=lambda node: degree[node], reverse=True)[:max_nodes]
    sub = nx.DiGraph(graph.subgraph(top))
    layout = nx.spring_layout(sub, seed=42, k=1.2 / max(1, len(sub)) ** 0.5)

    # Matplotlib cannot shape Arabic-script text, so English labels (or ids) are drawn.
    labels = {node: graph.nodes[node]["label_en"] or node.split(":")[-1] for node in sub}
    colors = [TYPE_COLORS.get(graph.nodes[node]["type"], "#7f7f7f") for node in sub]
    sizes = [200 + 40 * degree[node] for node in sub]

    figure, axis = plt.subplots(figsize=(16, 12))
    nx.draw_networkx_edges(sub, layout, ax=axis, alpha=0.3, arrows=True, arrowsize=8)
    nx.draw_networkx_nodes(sub, layout, ax=axis, node_color=colors, node_size=sizes, alpha=0.9)
    nx.draw_networkx_labels(sub, layout, labels=labels, ax=axis, font_size=7)
    for entity_type, color in TYPE_COLORS.items():
        axis.scatter([], [], c=color, label=entity_type)
    axis.legend(loc="upper left", frameon=False)
    axis.set_title(f"KM-KENet medical knowledge graph (top {len(sub)} nodes by degree)")
    axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)

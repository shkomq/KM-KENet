"""Aggregation of mentions into a semantically structured medical knowledge graph.

Nodes are canonical entities; edges are (head, relation, tail) triples with
support (number of distinct supporting sentences), mean confidence,
provenance (evidence sentences) and the extractors that produced them.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx

from step5_knowledge_graph.canonicalizer import CanonicalEntity, EntityCanonicalizer
from step5_knowledge_graph.extractor import EntityMention, RelationMention, SentenceExtraction


@dataclass
class _NodeRecord:
    entity: CanonicalEntity
    surface_forms: Counter[str] = field(default_factory=Counter)
    sentence_ids: dict[str, None] = field(default_factory=dict)  # ordered set
    confidence_sum: float = 0.0
    mentions: int = 0
    extractors: set[str] = field(default_factory=set)


@dataclass
class _EdgeRecord:
    sentence_ids: dict[str, None] = field(default_factory=dict)
    confidence_sum: float = 0.0
    mentions: int = 0
    evidence: list[dict[str, str]] = field(default_factory=list)
    extractors: set[str] = field(default_factory=set)


class KnowledgeGraphBuilder:
    def __init__(
        self,
        canonicalizer: EntityCanonicalizer,
        min_entity_confidence: float = 0.5,
        min_relation_confidence: float = 0.5,
        max_evidence_per_edge: int = 10,
        max_sentence_ids_per_node: int = 200,
    ) -> None:
        self.canonicalizer = canonicalizer
        self.min_entity_confidence = min_entity_confidence
        self.min_relation_confidence = min_relation_confidence
        self.max_evidence_per_edge = max_evidence_per_edge
        self.max_sentence_ids_per_node = max_sentence_ids_per_node
        self._nodes: dict[str, _NodeRecord] = {}
        self._edges: dict[tuple[str, str, str], _EdgeRecord] = {}

    def _add_entity(self, mention: EntityMention) -> str:
        entity = self.canonicalizer.canonicalize(mention.tokens, mention.type)
        record = self._nodes.get(entity.node_id)
        if record is None:
            record = self._nodes[entity.node_id] = _NodeRecord(entity)
        record.surface_forms[mention.text] += 1
        if len(record.sentence_ids) < self.max_sentence_ids_per_node:
            record.sentence_ids[mention.sentence_id] = None
        record.confidence_sum += mention.confidence
        record.mentions += 1
        record.extractors.add(mention.extractor)
        return entity.node_id

    def _node_id(self, mention: EntityMention) -> str:
        """Node id of an argument; registers the mention only if it was never seen as an entity."""
        node_id = self.canonicalizer.canonicalize(mention.tokens, mention.type).node_id
        return node_id if node_id in self._nodes else self._add_entity(mention)

    def _add_relation(self, mention: RelationMention) -> None:
        head_id = self._node_id(mention.head)
        tail_id = self._node_id(mention.tail)
        if head_id == tail_id:
            return
        record = self._edges.setdefault((head_id, mention.type, tail_id), _EdgeRecord())
        if mention.sentence_id not in record.sentence_ids and len(record.evidence) < self.max_evidence_per_edge:
            record.evidence.append({
                "sentence_id": mention.sentence_id,
                "doc_id": mention.doc_id,
                "text": mention.sentence_text,
            })
        record.sentence_ids[mention.sentence_id] = None
        record.confidence_sum += mention.confidence
        record.mentions += 1
        record.extractors.add(mention.extractor)

    def add(self, extractions: Iterable[SentenceExtraction]) -> None:
        for extraction in extractions:
            # Entities first, so relation arguments resolve to already-counted nodes.
            for entity in extraction.entities:
                if entity.confidence >= self.min_entity_confidence:
                    self._add_entity(entity)
            for relation in extraction.relations:
                if (
                    relation.confidence >= self.min_relation_confidence
                    and relation.head.confidence >= self.min_entity_confidence
                    and relation.tail.confidence >= self.min_entity_confidence
                ):
                    self._add_relation(relation)

    def build(self, min_triple_support: int = 1) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        kept_edges = {
            key: record for key, record in self._edges.items() if len(record.sentence_ids) >= min_triple_support
        }
        used_nodes = {head for head, _, _ in kept_edges} | {tail for _, _, tail in kept_edges}

        for node_id in used_nodes:
            record = self._nodes[node_id]
            entity = record.entity
            label_ckb = entity.label_ckb if entity.lexicon_id else record.surface_forms.most_common(1)[0][0]
            graph.add_node(
                node_id,
                type=entity.entity_type,
                label_ckb=label_ckb,
                label_en=entity.label_en or "",
                lexicon_id=entity.lexicon_id or "",
                aliases=[form for form, _ in record.surface_forms.most_common(20)],
                mention_count=record.mentions,
                confidence=round(record.confidence_sum / max(1, record.mentions), 4),
                sentence_ids=list(record.sentence_ids),
                extractors=sorted(record.extractors),
            )

        for (head_id, relation, tail_id), record in kept_edges.items():
            graph.add_edge(
                head_id,
                tail_id,
                key=relation,
                relation=relation,
                support=len(record.sentence_ids),
                mentions=record.mentions,
                confidence=round(record.confidence_sum / max(1, record.mentions), 4),
                evidence=record.evidence,
                extractors=sorted(record.extractors),
            )
        return graph

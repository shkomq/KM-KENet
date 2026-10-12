"""Retrievers over the Kurdish evidence: BM25, multilingual dense retrieval and the knowledge graph."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from common.io_utils import ensure_dir
from step2_preprocessing.text_tools import KurdishTextTools
from step2_preprocessing.tokenizer import is_number, is_punctuation

logger = logging.getLogger(__name__)

RELATION_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "TREATS": ("treat", "therap", "cure", "medic", "drug", "manage", "remed", "چارەسەر", "دەرمان"),
    "HAS_SYMPTOM": ("symptom", "sign", "manifest", "present", "نیشانە"),
    "DIAGNOSED_BY": ("diagnos", "test", "detect", "screen", "دەستنیشان", "پشکنین"),
    "AFFECTS": ("affect", "organ", "damage", "complicat", "کاریگەری", "زیان"),
    "HAS_DOSAGE": ("dose", "dosage", "how much", "mg", "ژەم", "بڕ"),
}


def detect_relation_intents(*queries: str) -> set[str]:
    """Relations the question asks about; empty set = no specific intent."""
    text = " ".join(queries).lower()
    return {relation for relation, keywords in RELATION_INTENT_KEYWORDS.items() if any(k in text for k in keywords)}


def reciprocal_rank_fusion(rankings: Iterable[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


@dataclass(frozen=True)
class Passage:
    sentence_id: str
    doc_id: str
    text: str
    tokens: tuple[str, ...]


class DenseEncoder:
    """Multilingual sentence embeddings (E5 prefixes are added automatically), with disk caching."""

    def __init__(self, model_name: str, cache_dir: Path, device: str | None = None, batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.cache_dir = ensure_dir(cache_dir)
        self.batch_size = batch_size
        self._is_e5 = "e5" in model_name.lower()

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        ).astype(np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([f"query: {t}" if self._is_e5 else t for t in texts])

    def encode_passages(self, texts: Sequence[str], cache_name: str) -> np.ndarray:
        prefixed = [f"passage: {t}" if self._is_e5 else t for t in texts]
        digest = hashlib.sha1(self.model_name.encode("utf-8"))
        for text in prefixed:
            digest.update(text.encode("utf-8"))
            digest.update(b"\x00")
        cache_path = self.cache_dir / f"{cache_name}_{digest.hexdigest()[:16]}.npy"
        if cache_path.is_file():
            embeddings = np.load(cache_path)
            if embeddings.shape[0] == len(prefixed):
                return embeddings
        embeddings = self._encode(prefixed) if prefixed else np.zeros((0, 1), dtype=np.float32)
        np.save(cache_path, embeddings)
        return embeddings


class BM25Retriever:
    def __init__(self, passages: Sequence[Passage], tools: KurdishTextTools, stopwords: frozenset[str]) -> None:
        from rank_bm25 import BM25Okapi

        self.tools = tools
        self.stopwords = stopwords
        corpus = [self._terms(passage.tokens) for passage in passages]
        self._bm25 = BM25Okapi([terms or ["<empty>"] for terms in corpus]) if corpus else None

    def _terms(self, tokens: Sequence[str]) -> list[str]:
        return [
            self.tools.stemmer.stem(token)
            for token in tokens
            if token not in self.stopwords and not is_punctuation(token) and not is_number(token)
        ]

    def search(self, query: str, top_n: int) -> list[int]:
        if self._bm25 is None:
            return []
        terms = self._terms(self.tools.tokenize(query))
        if not terms:
            return []
        scores = self._bm25.get_scores(terms)
        order = np.argsort(-scores)[:top_n]
        return [int(index) for index in order if scores[index] > 0]


class DenseRetriever:
    def __init__(self, passages: Sequence[Passage], encoder: DenseEncoder) -> None:
        self.encoder = encoder
        self.embeddings = encoder.encode_passages([p.text for p in passages], cache_name="passages")

    def search(self, query: str, top_n: int) -> list[int]:
        if self.embeddings.shape[0] == 0:
            return []
        scores = self.embeddings @ self.encoder.encode_queries([query])[0]
        top_n = min(top_n, scores.shape[0])
        order = np.argpartition(-scores, top_n - 1)[:top_n]
        return [int(index) for index in order[np.argsort(-scores[order])]]


@dataclass
class KGFact:
    fact_id: str
    relation: str
    head: dict[str, Any]
    tail: dict[str, Any]
    support: int
    confidence: float
    evidence: list[dict[str, str]]
    score: float = 0.0

    @staticmethod
    def _name(node: Mapping[str, Any]) -> str:
        english = node.get("label_en") or ""
        return f"{english} ({node['label_ckb']})" if english else node["label_ckb"]

    def render(self) -> str:
        return (
            f"[{self.fact_id}] {self._name(self.head)} --{self.relation}--> {self._name(self.tail)} "
            f"(support={self.support}, confidence={self.confidence:.2f})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.fact_id,
            "relation": self.relation,
            "head": {k: self.head.get(k) for k in ("id", "type", "label_ckb", "label_en")},
            "tail": {k: self.tail.get(k) for k in ("id", "type", "label_ckb", "label_en")},
            "support": self.support,
            "confidence": self.confidence,
            "score": round(self.score, 4),
            "evidence": self.evidence[:3],
        }


@dataclass
class KnowledgeGraphRetriever:
    nodes: dict[str, dict[str, Any]]
    edges: list[dict[str, Any]]
    tools: KurdishTextTools
    encoder: DenseEncoder | None = None
    _incident: dict[str, list[int]] = field(init=False, default_factory=dict)
    _sentence_nodes: dict[str, list[str]] = field(init=False, default_factory=dict)
    _node_stems: dict[str, list[tuple[str, ...]]] = field(init=False, default_factory=dict)
    _node_ids: list[str] = field(init=False, default_factory=list)
    _node_embeddings: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        for index, edge in enumerate(self.edges):
            self._incident.setdefault(edge["head"], []).append(index)
            self._incident.setdefault(edge["tail"], []).append(index)
        for node_id, node in self.nodes.items():
            for sentence_id in node.get("sentence_ids", []):
                self._sentence_nodes.setdefault(sentence_id, []).append(node_id)
        for edge in self.edges:
            for item in edge.get("evidence", []):
                linked = self._sentence_nodes.setdefault(item["sentence_id"], [])
                for node_id in (edge["head"], edge["tail"]):
                    if node_id not in linked:
                        linked.append(node_id)
        for node_id, node in self.nodes.items():
            forms = {node["label_ckb"], *node.get("aliases", [])}
            self._node_stems[node_id] = [
                stems for stems in (tuple(self.tools.stemmer.stem(t) for t in self.tools.tokenize(f)) for f in forms)
                if stems
            ]
        self._node_ids = list(self.nodes)
        if self.encoder is not None and self._node_ids:
            texts = [
                f"{self.nodes[n].get('label_en') or ''} {self.nodes[n]['label_ckb']}".strip() for n in self._node_ids
            ]
            self._node_embeddings = self.encoder.encode_passages(texts, cache_name="kg_nodes")

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], tools: KurdishTextTools, encoder: DenseEncoder | None
    ) -> "KnowledgeGraphRetriever":
        nodes = {node["id"]: node for node in payload.get("nodes", [])}
        edges = [edge for edge in payload.get("edges", []) if edge["head"] in nodes and edge["tail"] in nodes]
        return cls(nodes, edges, tools, encoder)

    @property
    def evidence_sentence_ids(self) -> set[str]:
        """Sentences that support at least one knowledge-graph edge."""
        return {item["sentence_id"] for edge in self.edges for item in edge.get("evidence", [])}

    def match_nodes(self, query_en: str, query_ckb: str, top_k: int) -> dict[str, float]:
        """Lexical matches (score 1.0) plus the top dense matches (cosine similarity)."""
        scores: dict[str, float] = {}
        lowered = query_en.lower()
        query_stems = {self.tools.stemmer.stem(t) for t in self.tools.tokenize(query_ckb)}
        for node_id, node in self.nodes.items():
            english = (node.get("label_en") or "").lower()
            if english and re.search(rf"\b{re.escape(english)}\b", lowered):
                scores[node_id] = 1.0
            elif any(set(stems) <= query_stems for stems in self._node_stems[node_id]):
                scores[node_id] = 1.0

        queries = [q for q in (query_en, query_ckb) if q.strip()]
        if queries and self.encoder is not None and self._node_embeddings is not None and self._node_embeddings.shape[0]:
            query_vectors = self.encoder.encode_queries(queries)
            similarity = (self._node_embeddings @ query_vectors.T).max(axis=1)
            for index in np.argsort(-similarity)[:top_k]:
                node_id = self._node_ids[int(index)]
                scores[node_id] = max(scores.get(node_id, 0.0), float(similarity[index]))
        return scores

    def nodes_in_sentences(self, sentence_ids: Iterable[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sentence_id in sentence_ids:
            for node_id in self._sentence_nodes.get(sentence_id, []):
                counts[node_id] = counts.get(node_id, 0) + 1
        return counts

    def facts(self, seeds: Mapping[str, float], relations: set[str], max_facts: int) -> list[KGFact]:
        best: dict[int, float] = {}
        for node_id, seed_score in seeds.items():
            for index in self._incident.get(node_id, []):
                edge = self.edges[index]
                if relations and edge["relation"] not in relations:
                    continue
                score = seed_score * float(edge.get("confidence", 1.0)) * (1.0 + math.log1p(int(edge["support"])))
                best[index] = max(best.get(index, 0.0), score)
        ranked = sorted(best.items(), key=lambda pair: pair[1], reverse=True)[:max_facts]
        return [
            KGFact(
                fact_id=f"F{rank}",
                relation=self.edges[index]["relation"],
                head={"id": self.edges[index]["head"], **self.nodes[self.edges[index]["head"]]},
                tail={"id": self.edges[index]["tail"], **self.nodes[self.edges[index]["tail"]]},
                support=int(self.edges[index]["support"]),
                confidence=float(self.edges[index].get("confidence", 1.0)),
                evidence=list(self.edges[index].get("evidence", [])),
                score=score,
            )
            for rank, (index, score) in enumerate(ranked, start=1)
        ]

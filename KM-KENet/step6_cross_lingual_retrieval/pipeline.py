"""Cross-lingual question answering over the Kurdish medical knowledge base.

question (any language)
  -> language detection -> Aya translation to English and Sorani
  -> passage retrieval: BM25 (Sorani query) + dense multilingual retrieval, fused with RRF
  -> KG retrieval: nodes matched by the query or mentioned in the top passages,
     expanded along the relations the question asks about
  -> Aya answer in English, grounded in KG facts and Kurdish passages
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from step6_cross_lingual_retrieval.answer import MultilingualAssistant
from step6_cross_lingual_retrieval.language import detect_language
from step6_cross_lingual_retrieval.retrievers import (
    BM25Retriever,
    DenseRetriever,
    KGFact,
    KnowledgeGraphRetriever,
    Passage,
    detect_relation_intents,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    question: str
    language: str
    query_en: str
    query_ckb: str
    translation_method: str
    relation_intents: list[str]
    facts: list[KGFact]
    passages: list[Passage]
    answer: str
    answer_method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "detected_language": self.language,
            "query_en": self.query_en,
            "query_ckb": self.query_ckb,
            "translation_method": self.translation_method,
            "relation_intents": self.relation_intents,
            "answer": self.answer,
            "answer_method": self.answer_method,
            "facts": [fact.to_dict() for fact in self.facts],
            "passages": [
                {"id": f"P{index}", "sentence_id": p.sentence_id, "doc_id": p.doc_id, "text": p.text}
                for index, p in enumerate(self.passages, start=1)
            ],
        }


class CrossLingualQA:
    def __init__(
        self,
        assistant: MultilingualAssistant,
        passages: Sequence[Passage],
        bm25: BM25Retriever,
        dense: DenseRetriever | None,
        kg: KnowledgeGraphRetriever,
        *,
        top_k_passages: int = 8,
        candidate_pool: int = 100,
        top_k_nodes: int = 12,
        max_facts: int = 25,
        rrf_k: int = 60,
    ) -> None:
        self.assistant = assistant
        self.passages = list(passages)
        self.bm25 = bm25
        self.dense = dense
        self.kg = kg
        self.top_k_passages = top_k_passages
        self.candidate_pool = candidate_pool
        self.top_k_nodes = top_k_nodes
        self.max_facts = max_facts
        self.rrf_k = rrf_k

    def _retrieve_passages(self, query_en: str, query_ckb: str) -> list[Passage]:
        rankings = [self.bm25.search(query_ckb, self.candidate_pool)]
        if self.dense is not None:
            rankings.append(self.dense.search(query_en, self.candidate_pool))
            if query_ckb != query_en:
                rankings.append(self.dense.search(query_ckb, self.candidate_pool))
        fused = reciprocal_rank_fusion([r for r in rankings if r], k=self.rrf_k)
        return [self.passages[index] for index, _ in fused[: self.top_k_passages]]

    def answer(self, question: str) -> QAResult:
        question = question.strip()
        if not question:
            raise ValueError("The question is empty")
        language = detect_language(question)
        query_en, method_en = self.assistant.translate(question, language, "en")
        query_ckb, method_ckb = self.assistant.translate(question, language, "ckb")
        translation_method = method_ckb if language == "en" else method_en
        logger.info("Language=%s | EN: %s | CKB: %s", language, query_en, query_ckb)

        intents = detect_relation_intents(question, query_en, query_ckb)
        passages = self._retrieve_passages(query_en, query_ckb)

        seeds = self.kg.match_nodes(query_en, query_ckb, self.top_k_nodes)
        for node_id, count in self.kg.nodes_in_sentences(p.sentence_id for p in passages).items():
            seeds[node_id] = max(seeds.get(node_id, 0.0), min(0.9, 0.5 + 0.1 * count))
        facts = self.kg.facts(seeds, intents, self.max_facts)
        logger.info("Intents=%s | seed nodes=%d | facts=%d | passages=%d",
                    sorted(intents) or "any", len(seeds), len(facts), len(passages))

        answer, answer_method = self.assistant.answer(query_en, facts, passages)
        return QAResult(
            question=question,
            language=language,
            query_en=query_en,
            query_ckb=query_ckb,
            translation_method=translation_method,
            relation_intents=sorted(intents),
            facts=facts,
            passages=passages,
            answer=answer,
            answer_method=answer_method,
        )

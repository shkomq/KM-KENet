"""Query translation and grounded answer generation with Aya, plus template fallbacks."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

from step6_cross_lingual_retrieval.aya_client import ChatClient
from step6_cross_lingual_retrieval.language import LANGUAGE_NAMES
from step6_cross_lingual_retrieval.retrievers import KGFact, Passage
from step6_cross_lingual_retrieval.translator import LexiconTranslator

logger = logging.getLogger(__name__)

TRANSLATION_SYSTEM_PROMPT = (
    "You are a professional medical translator. Translate the user's text faithfully, "
    "keeping medical terminology precise. Output only the translation, without notes or quotes."
)

ANSWER_SYSTEM_PROMPT = (
    "You are KM-KENet, a cross-lingual medical knowledge assistant. You answer questions in English "
    "using ONLY the evidence provided, which was extracted from Central Kurdish (Sorani) medical sources: "
    "structured facts from a medical knowledge graph and original Kurdish passages. Translate the Kurdish "
    "evidence as needed, cite facts as [F#] and passages as [P#], and say clearly when the evidence is "
    "insufficient. Do not invent treatments or dosages. End with a one-line reminder that this is not "
    "a substitute for professional medical advice."
)

_RELATION_PHRASES = {
    "TREATS": "Treatments",
    "HAS_SYMPTOM": "Symptoms",
    "AFFECTS": "Affected organs / body parts",
    "DIAGNOSED_BY": "Diagnostic tests and procedures",
    "HAS_DOSAGE": "Dosages",
}


def _name(node: dict) -> str:
    english = node.get("label_en") or ""
    return f"{english} ({node['label_ckb']})" if english else node["label_ckb"]


class TemplateAnswerGenerator:
    """Deterministic English answer built from KG facts (used when Aya is unavailable)."""

    def generate(self, question: str, facts: Sequence[KGFact], passages: Sequence[Passage]) -> str:
        if not facts and not passages:
            return "No supporting evidence was found in the Kurdish medical knowledge base for this question."

        lines = [f"Question: {question}", "Answer extracted from Kurdish medical sources:"]
        grouped: dict[str, dict[str, list[KGFact]]] = defaultdict(lambda: defaultdict(list))
        for fact in facts:
            # Group by the condition (TREATS tail) or by the head (other relations, e.g. the drug for HAS_DOSAGE).
            anchor = fact.tail if fact.relation == "TREATS" else fact.head
            grouped[fact.relation][_name(anchor)].append(fact)

        for relation, by_anchor in grouped.items():
            lines.append(f"\n{_RELATION_PHRASES.get(relation, relation)}:")
            for anchor, anchor_facts in by_anchor.items():
                others = [
                    f"{_name(f.head if relation == 'TREATS' else f.tail)} [{f.fact_id}]" for f in anchor_facts
                ]
                lines.append(f"- {anchor}: " + "; ".join(others))

        if passages:
            lines.append("\nSupporting Kurdish passages (original text):")
            for index, passage in enumerate(passages[:3], start=1):
                lines.append(f"[P{index}] {passage.text}")
        lines.append(
            "\nNote: generated without a language model from automatically extracted facts; "
            "this is not a substitute for professional medical advice."
        )
        return "\n".join(lines)


class MultilingualAssistant:
    """Uses Aya when available and falls back to lexicon translation / template answers."""

    def __init__(self, client: ChatClient | None, translator: LexiconTranslator) -> None:
        self.client = client
        self.translator = translator
        self.template = TemplateAnswerGenerator()

    @property
    def backend(self) -> str:
        return self.client.name if self.client else "lexicon-fallback"

    def translate(self, text: str, source: str, target: str) -> tuple[str, str]:
        """Return (translation, method)."""
        if source == target:
            return text, "identity"
        if self.client is not None:
            prompt = (
                f"Translate the following text from {LANGUAGE_NAMES.get(source, source)} "
                f"to {LANGUAGE_NAMES.get(target, target)}.\n\n{text}"
            )
            try:
                translation = self.client.chat(
                    [{"role": "system", "content": TRANSLATION_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                    max_new_tokens=256,
                ).strip().strip('"')
                if translation:
                    return translation, self.client.name
            except Exception as exc:
                logger.warning("Aya translation failed (%s); using the lexicon translator", exc)
        return self.translator.translate(text, source, target), "lexicon"

    def answer(self, question_en: str, facts: Sequence[KGFact], passages: Sequence[Passage]) -> tuple[str, str]:
        """Return (answer, method)."""
        if self.client is not None and (facts or passages):
            fact_block = "\n".join(fact.render() for fact in facts) or "(no facts found)"
            passage_block = "\n".join(
                f"[P{index}] {passage.text}" for index, passage in enumerate(passages, start=1)
            ) or "(no passages found)"
            prompt = (
                f"Question: {question_en}\n\n"
                f"Knowledge-graph facts:\n{fact_block}\n\n"
                f"Kurdish source passages:\n{passage_block}\n\n"
                "Answer the question in English based only on this evidence."
            )
            try:
                answer = self.client.chat(
                    [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                )
                if answer:
                    return answer, self.client.name
            except Exception as exc:
                logger.warning("Aya answer generation failed (%s); using the template answer", exc)
        return self.template.generate(question_en, facts, passages), "template"

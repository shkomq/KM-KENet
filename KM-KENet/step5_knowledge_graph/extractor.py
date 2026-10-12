"""Entity and relation mentions from the trained model and from the weak labels."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence

from tqdm import tqdm

if TYPE_CHECKING:  # torch is only needed when the trained model is used
    from step4_model_training.predictor import KMKENetPredictor

logger = logging.getLogger(__name__)

EXTRACTOR_MODEL = "km-kenet"
EXTRACTOR_WEAK = "weak-supervision"


@dataclass(frozen=True)
class EntityMention:
    sentence_id: str
    doc_id: str
    tokens: tuple[str, ...]
    type: str
    confidence: float
    extractor: str

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


@dataclass(frozen=True)
class RelationMention:
    sentence_id: str
    doc_id: str
    sentence_text: str
    head: EntityMention
    tail: EntityMention
    type: str
    confidence: float
    extractor: str


@dataclass
class SentenceExtraction:
    entities: list[EntityMention]
    relations: list[RelationMention]


def extract_with_model(
    predictor: "KMKENetPredictor",
    sentences: Sequence[dict[str, Any]],
    chunk_size: int = 2048,
) -> Iterator[SentenceExtraction]:
    """Run the model over sentences in chunks, yielding one extraction per sentence."""
    progress = tqdm(total=len(sentences), desc="KM-KENet inference", mininterval=10)
    for start in range(0, len(sentences), chunk_size):
        chunk = sentences[start : start + chunk_size]
        predictions = predictor.predict([sentence["tokens"] for sentence in chunk])
        for sentence, prediction in zip(chunk, predictions):
            tokens = sentence["tokens"]
            entities = [
                EntityMention(
                    sentence["sentence_id"], sentence["doc_id"], tuple(tokens[e.start : e.end]),
                    e.type, e.confidence, EXTRACTOR_MODEL,
                )
                for e in prediction.entities
            ]
            relations = [
                RelationMention(
                    sentence["sentence_id"], sentence["doc_id"], sentence["text"],
                    entities[r.head], entities[r.tail], r.type, r.confidence, EXTRACTOR_MODEL,
                )
                for r in prediction.relations
            ]
            yield SentenceExtraction(entities, relations)
        progress.update(len(chunk))
    progress.close()


def extract_from_weak_labels(records: Iterable[dict[str, Any]]) -> Iterator[SentenceExtraction]:
    for record in records:
        tokens = record["tokens"]
        entities = [
            EntityMention(
                record["sentence_id"], record["doc_id"], tuple(tokens[e["start"] : e["end"]]),
                e["type"], float(e.get("confidence", 1.0)), EXTRACTOR_WEAK,
            )
            for e in record["entities"]
        ]
        relations = [
            RelationMention(
                record["sentence_id"], record["doc_id"], record["text"],
                entities[r["head"]], entities[r["tail"]], r["type"],
                float(r.get("confidence", 1.0)), EXTRACTOR_WEAK,
            )
            for r in record.get("relations", [])
        ]
        yield SentenceExtraction(entities, relations)

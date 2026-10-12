"""Dataset, label space and batching for joint NER + relation extraction.

Word-level weak labels are aligned to XLM-R sub-words: the NER label sits on
the first sub-word of each word (other sub-words are ignored by the loss),
and every entity span is mapped to its sub-word range for relation pooling.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from common.schema import (
    ENTITY_TYPES,
    NO_RELATION,
    RELATION_TYPES,
    bio_labels,
    candidate_pairs,
)

IGNORE_INDEX = -100
REL_PAIR_COLUMNS = 7  # batch, head_start, head_end, head_type, tail_start, tail_end, tail_type


@dataclass
class LabelSpace:
    ner_labels: tuple[str, ...]
    relation_labels: tuple[str, ...]
    entity_types: tuple[str, ...]
    ner2id: dict[str, int] = field(init=False)
    rel2id: dict[str, int] = field(init=False)
    type2id: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        if self.relation_labels[0] != NO_RELATION:
            raise ValueError("NO_RELATION must be the first relation label")
        self.ner2id = {label: index for index, label in enumerate(self.ner_labels)}
        self.rel2id = {label: index for index, label in enumerate(self.relation_labels)}
        self.type2id = {label: index for index, label in enumerate(self.entity_types)}

    @classmethod
    def default(cls) -> "LabelSpace":
        return cls(tuple(bio_labels(ENTITY_TYPES)), tuple(RELATION_TYPES), tuple(ENTITY_TYPES))

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "ner_labels": list(self.ner_labels),
            "relation_labels": list(self.relation_labels),
            "entity_types": list(self.entity_types),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Sequence[str]]) -> "LabelSpace":
        return cls(tuple(data["ner_labels"]), tuple(data["relation_labels"]), tuple(data["entity_types"]))


def word_to_subword_maps(word_ids: Sequence[int | None]) -> tuple[dict[int, int], dict[int, int]]:
    """First and last sub-word index of every (non-truncated) word."""
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for index, word in enumerate(word_ids):
        if word is None:
            continue
        first.setdefault(word, index)
        last[word] = index
    return first, last


class KMKENetDataset(Dataset):
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        tokenizer: PreTrainedTokenizerFast,
        label_space: LabelSpace,
        max_length: int,
        *,
        use_confidence_weights: bool = True,
        max_negative_relations: int | None = None,
        no_relation_weight: float = 1.0,
        seed: int = 42,
    ) -> None:
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for word alignment")
        self.tokenizer = tokenizer
        self.label_space = label_space
        self.max_length = max_length
        self.use_confidence_weights = use_confidence_weights
        self.max_negative_relations = max_negative_relations
        self.no_relation_weight = no_relation_weight
        rng = random.Random(seed)
        self.features = [
            feature for feature in (self._encode(record, rng) for record in records) if feature is not None
        ]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]

    def _encode(self, record: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
        tokens = record["tokens"]
        if not tokens:
            return None
        encoding = self.tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=self.max_length)
        input_ids = encoding["input_ids"]
        first, last = word_to_subword_maps(encoding.word_ids())
        if not first:
            return None

        tags = record["ner_tags"]
        confidences = record.get("token_confidence") or [1.0] * len(tokens)
        ner_labels = [IGNORE_INDEX] * len(input_ids)
        ner_weights = [0.0] * len(input_ids)
        for word, sub_index in first.items():
            ner_labels[sub_index] = self.label_space.ner2id[tags[word]]
            ner_weights[sub_index] = float(confidences[word]) if self.use_confidence_weights else 1.0

        # Map entity word spans to sub-word spans (None if truncated away).
        sub_spans: list[tuple[int, int] | None] = []
        for entity in record["entities"]:
            start, end = entity["start"], entity["end"] - 1
            sub_spans.append((first[start], last[end] + 1) if start in first and end in last else None)

        gold = {
            (relation["head"], relation["tail"]): (relation["type"], float(relation.get("confidence", 1.0)))
            for relation in record.get("relations", [])
        }
        positives: list[tuple[list[int], int, float]] = []
        negatives: list[tuple[list[int], int, float]] = []
        entity_types = [entity["type"] for entity in record["entities"]]
        for head, tail, _ in candidate_pairs(entity_types):
            head_span, tail_span = sub_spans[head], sub_spans[tail]
            if head_span is None or tail_span is None:
                continue
            pair = [
                head_span[0], head_span[1], self.label_space.type2id[entity_types[head]],
                tail_span[0], tail_span[1], self.label_space.type2id[entity_types[tail]],
            ]
            if (head, tail) in gold:
                relation, confidence = gold[(head, tail)]
                weight = confidence if self.use_confidence_weights else 1.0
                positives.append((pair, self.label_space.rel2id[relation], weight))
            else:
                negatives.append((pair, self.label_space.rel2id[NO_RELATION], self.no_relation_weight))

        if self.max_negative_relations is not None and len(negatives) > self.max_negative_relations:
            negatives = rng.sample(negatives, self.max_negative_relations)
        relations = positives + negatives

        return {
            "input_ids": input_ids,
            "ner_labels": ner_labels,
            "ner_weights": ner_weights,
            "rel_pairs": [pair for pair, _, _ in relations],
            "rel_labels": [label for _, label, _ in relations],
            "rel_weights": [weight for _, _, weight in relations],
        }


class KMKENetCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(features)
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        ner_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
        ner_weights = torch.zeros((batch_size, max_len), dtype=torch.float)

        rel_rows: list[list[int]] = []
        rel_labels: list[int] = []
        rel_weights: list[float] = []
        for row, feature in enumerate(features):
            length = len(feature["input_ids"])
            input_ids[row, :length] = torch.tensor(feature["input_ids"], dtype=torch.long)
            attention_mask[row, :length] = 1
            ner_labels[row, :length] = torch.tensor(feature["ner_labels"], dtype=torch.long)
            ner_weights[row, :length] = torch.tensor(feature["ner_weights"], dtype=torch.float)
            for pair in feature["rel_pairs"]:
                rel_rows.append([row, *pair])
            rel_labels.extend(feature["rel_labels"])
            rel_weights.extend(feature["rel_weights"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ner_labels": ner_labels,
            "ner_weights": ner_weights,
            "rel_pairs": (
                torch.tensor(rel_rows, dtype=torch.long)
                if rel_rows
                else torch.zeros((0, REL_PAIR_COLUMNS), dtype=torch.long)
            ),
            "rel_labels": torch.tensor(rel_labels, dtype=torch.long),
            "rel_weights": torch.tensor(rel_weights, dtype=torch.float),
        }

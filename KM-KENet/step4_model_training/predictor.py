"""Inference wrapper: tokenised sentences -> entities and relations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoTokenizer

from common.io_utils import read_json
from common.schema import NO_RELATION, bio_to_spans, candidate_pairs
from step4_model_training.data import word_to_subword_maps
from step4_model_training.model import KMKENet


@dataclass(frozen=True)
class PredictedEntity:
    start: int
    end: int
    type: str
    text: str
    confidence: float


@dataclass(frozen=True)
class PredictedRelation:
    head: int  # index into the sentence's entity list
    tail: int
    type: str
    confidence: float


@dataclass(frozen=True)
class SentencePrediction:
    entities: list[PredictedEntity]
    relations: list[PredictedRelation]


class KMKENetPredictor:
    def __init__(self, model_dir: Path, device: torch.device, batch_size: int = 64) -> None:
        if not (model_dir / KMKENet.CONFIG_NAME).is_file():
            raise FileNotFoundError(f"No trained KM-KENet model in {model_dir}. Run step 4 first.")
        self.model = KMKENet.from_pretrained(model_dir).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        training_config = read_json(model_dir / "training_config.json") if (model_dir / "training_config.json").is_file() else {}
        self.max_length = int(training_config.get("max_length", 160))
        self.device = device
        self.batch_size = batch_size
        space = self.model.label_space
        self.ner_labels = space.ner_labels
        self.relation_labels = space.relation_labels
        self.type2id = space.type2id
        self.no_relation_id = space.rel2id[NO_RELATION]

    @torch.inference_mode()
    def predict(self, sentences: Sequence[Sequence[str]]) -> list[SentencePrediction]:
        results: list[SentencePrediction] = []
        for start in range(0, len(sentences), self.batch_size):
            results.extend(self._predict_batch([list(tokens) for tokens in sentences[start : start + self.batch_size]]))
        return results

    def _predict_batch(self, batch: list[list[str]]) -> list[SentencePrediction]:
        non_empty = [index for index, tokens in enumerate(batch) if tokens]
        outputs = [SentencePrediction([], []) for _ in batch]
        if not non_empty:
            return outputs

        encoding = self.tokenizer(
            [batch[i] for i in non_empty],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        hidden_states = self.model.encode(input_ids, attention_mask)
        probabilities = torch.softmax(self.model.ner_logits(hidden_states).float(), dim=-1).cpu()

        entities_per_row: list[list[PredictedEntity]] = []
        pair_rows: list[list[int]] = []
        pair_index: list[tuple[int, int, int]] = []  # (row, head, tail)
        for row, original in enumerate(non_empty):
            tokens = batch[original]
            first, last = word_to_subword_maps(encoding.word_ids(batch_index=row))
            tags: list[str] = []
            token_conf: list[float] = []
            for word in range(len(tokens)):
                if word in first:
                    probs = probabilities[row, first[word]]
                    label_id = int(probs.argmax())
                    tags.append(self.ner_labels[label_id])
                    token_conf.append(float(probs[label_id]))
                else:  # truncated word
                    tags.append("O")
                    token_conf.append(1.0)

            entities = [
                PredictedEntity(
                    start, end, entity_type, " ".join(tokens[start:end]),
                    sum(token_conf[start:end]) / (end - start),
                )
                for start, end, entity_type in bio_to_spans(tags)
            ]
            entities_per_row.append(entities)

            for head, tail, _ in candidate_pairs([entity.type for entity in entities]):
                h, t = entities[head], entities[tail]
                if h.start not in first or (h.end - 1) not in last or t.start not in first or (t.end - 1) not in last:
                    continue
                pair_rows.append([
                    row, first[h.start], last[h.end - 1] + 1, self.type2id[h.type],
                    first[t.start], last[t.end - 1] + 1, self.type2id[t.type],
                ])
                pair_index.append((row, head, tail))

        relations_per_row: list[list[PredictedRelation]] = [[] for _ in non_empty]
        if pair_rows:
            rel_pairs = torch.tensor(pair_rows, dtype=torch.long, device=self.device)
            rel_probs = torch.softmax(self.model.relation_logits(hidden_states, rel_pairs).float(), dim=-1).cpu()
            for (row, head, tail), probs in zip(pair_index, rel_probs):
                label_id = int(probs.argmax())
                if label_id != self.no_relation_id:
                    relations_per_row[row].append(
                        PredictedRelation(head, tail, self.relation_labels[label_id], float(probs[label_id]))
                    )

        for row, original in enumerate(non_empty):
            outputs[original] = SentencePrediction(entities_per_row[row], relations_per_row[row])
        return outputs

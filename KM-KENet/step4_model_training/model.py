"""KM-KENet: a multilingual transformer encoder with joint NER and relation heads.

* NER head      — token classification over BIO tags (first sub-word of each word).
* Relation head — for an (head, tail) entity pair: [CLS; h; t; h*t; type(h); type(t)]
                  -> MLP -> relation logits. Mean-pooled sub-word spans give h and t.
                  Logits of relations the schema forbids for the pair's types are masked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, PreTrainedTokenizerBase

from common.io_utils import read_json, write_json
from common.schema import NO_RELATION, RELATION_SCHEMA
from step4_model_training.data import IGNORE_INDEX, LabelSpace

_MASK_VALUE = -1e4  # safe in fp16


def build_relation_mask(label_space: LabelSpace) -> torch.Tensor:
    """allowed[head_type, tail_type, relation] — NO_RELATION is always allowed."""
    n_types, n_rel = len(label_space.entity_types), len(label_space.relation_labels)
    allowed = torch.zeros((n_types, n_types, n_rel), dtype=torch.bool)
    allowed[:, :, label_space.rel2id[NO_RELATION]] = True
    for relation, pairs in RELATION_SCHEMA.items():
        if relation not in label_space.rel2id:
            continue
        for head_type, tail_type in pairs:
            allowed[
                label_space.type2id[head_type], label_space.type2id[tail_type], label_space.rel2id[relation]
            ] = True
    return allowed


@dataclass
class KMKENetOutput:
    ner_logits: torch.Tensor
    rel_logits: torch.Tensor


class KMKENet(nn.Module):
    CONFIG_NAME = "kmkenet_config.json"
    HEADS_NAME = "heads.pt"
    ENCODER_DIR = "encoder"

    def __init__(
        self,
        encoder_name_or_path: str,
        label_space: LabelSpace,
        dropout: float = 0.1,
        relation_hidden_size: int = 512,
        type_embedding_dim: int = 32,
        base_encoder: str | None = None,
    ) -> None:
        super().__init__()
        self.label_space = label_space
        self.hparams: dict[str, Any] = {
            "base_encoder": base_encoder or encoder_name_or_path,
            "dropout": dropout,
            "relation_hidden_size": relation_hidden_size,
            "type_embedding_dim": type_embedding_dim,
        }
        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.ner_head = nn.Linear(hidden, len(label_space.ner_labels))
        self.type_embeddings = nn.Embedding(len(label_space.entity_types), type_embedding_dim)
        self.relation_head = nn.Sequential(
            nn.Linear(4 * hidden + 2 * type_embedding_dim, relation_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(relation_hidden_size, len(label_space.relation_labels)),
        )
        self.register_buffer("relation_mask", build_relation_mask(label_space), persistent=False)

    # ------------------------------------------------------------------ forward
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def ner_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ner_head(self.dropout(hidden_states))

    @staticmethod
    def _pool_spans(
        hidden_states: torch.Tensor, batch_index: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
    ) -> torch.Tensor:
        positions = torch.arange(hidden_states.size(1), device=hidden_states.device).unsqueeze(0)
        mask = (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1))
        mask = mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states[batch_index] * mask).sum(dim=1)
        return summed / mask.sum(dim=1).clamp_min(1.0)

    def relation_logits(self, hidden_states: torch.Tensor, rel_pairs: torch.Tensor) -> torch.Tensor:
        if rel_pairs.numel() == 0:
            return hidden_states.new_zeros((0, len(self.label_space.relation_labels)))
        batch_index = rel_pairs[:, 0]
        head = self._pool_spans(hidden_states, batch_index, rel_pairs[:, 1], rel_pairs[:, 2])
        tail = self._pool_spans(hidden_states, batch_index, rel_pairs[:, 4], rel_pairs[:, 5])
        head_type, tail_type = rel_pairs[:, 3], rel_pairs[:, 6]
        features = torch.cat(
            [
                hidden_states[batch_index, 0],
                head,
                tail,
                head * tail,
                self.type_embeddings(head_type).to(hidden_states.dtype),
                self.type_embeddings(tail_type).to(hidden_states.dtype),
            ],
            dim=-1,
        )
        logits = self.relation_head(self.dropout(features))
        allowed = self.relation_mask[head_type, tail_type]
        return logits.masked_fill(~allowed, _MASK_VALUE)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, rel_pairs: torch.Tensor
    ) -> KMKENetOutput:
        hidden_states = self.encode(input_ids, attention_mask)
        return KMKENetOutput(self.ner_logits(hidden_states), self.relation_logits(hidden_states, rel_pairs))

    # ------------------------------------------------------------- persistence
    def save_pretrained(self, output_dir: Path, tokenizer: PreTrainedTokenizerBase | None = None) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(output_dir / self.ENCODER_DIR)
        torch.save(
            {
                "ner_head": self.ner_head.state_dict(),
                "type_embeddings": self.type_embeddings.state_dict(),
                "relation_head": self.relation_head.state_dict(),
            },
            output_dir / self.HEADS_NAME,
        )
        write_json(output_dir / self.CONFIG_NAME, {"hparams": self.hparams, "label_space": self.label_space.to_dict()})
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_dir: Path) -> "KMKENet":
        config = read_json(model_dir / cls.CONFIG_NAME)
        hparams = config["hparams"]
        model = cls(
            str(model_dir / cls.ENCODER_DIR),
            LabelSpace.from_dict(config["label_space"]),
            dropout=float(hparams["dropout"]),
            relation_hidden_size=int(hparams["relation_hidden_size"]),
            type_embedding_dim=int(hparams["type_embedding_dim"]),
            base_encoder=hparams.get("base_encoder"),
        )
        state = torch.load(model_dir / cls.HEADS_NAME, map_location="cpu", weights_only=True)
        model.ner_head.load_state_dict(state["ner_head"])
        model.type_embeddings.load_state_dict(state["type_embeddings"])
        model.relation_head.load_state_dict(state["relation_head"])
        return model


def compute_multitask_loss(
    output: KMKENetOutput,
    batch: dict[str, torch.Tensor],
    relation_loss_weight: float = 1.0,
    ner_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint objective L = alpha * L_NER + beta * L_RE (alpha = ner_loss_weight, beta = relation_loss_weight).

    Both terms are cross-entropies weighted by the confidence of the weak labels.
    """
    logits = output.ner_logits.float()
    labels = batch["ner_labels"].view(-1)
    token_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels, ignore_index=IGNORE_INDEX, reduction="none"
    )
    valid = labels != IGNORE_INDEX
    weights = batch["ner_weights"].view(-1)[valid]
    ner_loss = (token_loss[valid] * weights).sum() / weights.sum().clamp_min(1e-6)

    rel_loss = logits.new_zeros(())
    if output.rel_logits.numel() > 0:
        rel_token_loss = F.cross_entropy(output.rel_logits.float(), batch["rel_labels"], reduction="none")
        rel_weights = batch["rel_weights"]
        rel_loss = (rel_token_loss * rel_weights).sum() / rel_weights.sum().clamp_min(1e-6)

    total = ner_loss_weight * ner_loss + relation_loss_weight * rel_loss
    return total, {"ner_loss": float(ner_loss.detach()), "relation_loss": float(rel_loss.detach())}

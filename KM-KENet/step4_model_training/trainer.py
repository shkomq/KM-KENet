"""Training loop with mixed precision, gradient accumulation, evaluation and early stopping."""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from common.io_utils import write_json
from step4_model_training.data import IGNORE_INDEX
from step4_model_training.metrics import ner_metrics, relation_metrics
from step4_model_training.model import KMKENet, compute_multitask_loss
from step4_model_training.paper_metrics import (
    bio_to_io_probabilities,
    classification_measures,
    summarize,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 16
    eval_batch_size: int = 64
    encoder_learning_rate: float = 3e-5
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = True
    ner_loss_weight: float = 1.0
    relation_loss_weight: float = 1.0
    no_relation_weight: float = 0.5
    use_confidence_weights: bool = True
    max_negative_relations: int | None = 8
    max_train_examples: int | None = None
    max_eval_examples: int | None = None
    early_stopping_patience: int = 2
    num_workers: int = 0
    log_every_steps: int = 50

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        converted: dict[str, Any] = {}
        for item in fields(cls):
            if item.name not in data:
                continue
            value = data[item.name]
            if value is None:
                converted[item.name] = None
            elif item.name in {"fp16", "use_confidence_weights"}:
                converted[item.name] = bool(value)
            elif isinstance(item.default, float):
                converted[item.name] = float(value)
            else:
                converted[item.name] = int(value)
        unknown = set(data) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown training options: {sorted(unknown)}")
        config = cls(**converted)
        if config.epochs < 1 or config.batch_size < 1 or config.gradient_accumulation_steps < 1:
            raise ValueError("epochs, batch_size and gradient_accumulation_steps must be >= 1")
        return config


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class Trainer:
    def __init__(
        self,
        model: KMKENet,
        tokenizer: PreTrainedTokenizerBase,
        train_loader: DataLoader,
        dev_loader: DataLoader | None,
        config: TrainingConfig,
        device: torch.device,
        output_dir: Path,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.use_amp = bool(config.fp16 and device.type == "cuda")

        self.optimizer = torch.optim.AdamW(self._parameter_groups())
        steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
        total_steps = max(1, steps_per_epoch * config.epochs)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, int(config.warmup_ratio * total_steps), total_steps
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        label_space = model.label_space
        self.ner_id2label = list(label_space.ner_labels)
        self.rel_id2label = list(label_space.relation_labels)
        self.no_relation_id = 0
        self.io_classes = ["O", *label_space.entity_types]
        self.last_predictions: dict[str, Any] = {}

    def _parameter_groups(self) -> list[dict[str, Any]]:
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        groups: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {
            (True, True): [], (True, False): [], (False, True): [], (False, False): [],
        }
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            is_encoder = name.startswith("encoder.")
            decays = not any(marker in name for marker in no_decay)
            groups[(is_encoder, decays)].append(parameter)
        cfg = self.config
        return [
            {
                "params": params,
                "lr": cfg.encoder_learning_rate if is_encoder else cfg.head_learning_rate,
                "weight_decay": cfg.weight_decay if decays else 0.0,
            }
            for (is_encoder, decays), params in groups.items()
            if params
        ]

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------ train
    def train(self) -> dict[str, Any]:
        best_score = -math.inf
        best_epoch = 0
        patience = 0
        history: list[dict[str, Any]] = []

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(epoch)
            metrics = self.evaluate(self.dev_loader) if self.dev_loader is not None else {}
            score = metrics.get("score", -train_loss)
            history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
            logger.info(
                "Epoch %d | train_loss=%.4f | dev NER F1=%.4f | dev RE F1=%.4f",
                epoch, train_loss, metrics.get("ner", {}).get("f1", float("nan")),
                metrics.get("relations", {}).get("f1", float("nan")),
            )

            if score > best_score:
                best_score, best_epoch, patience = score, epoch, 0
                self.model.save_pretrained(self.output_dir, self.tokenizer)
                logger.info("New best model (score=%.4f) saved to %s", score, self.output_dir)
            else:
                patience += 1
                if patience >= self.config.early_stopping_patience:
                    logger.info("Early stopping after epoch %d", epoch)
                    break

        summary = {"best_epoch": best_epoch, "best_score": best_score, "history": history}
        write_json(self.output_dir / "training_history.json", summary)
        return summary

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        cfg = self.config
        total_loss = 0.0
        steps = len(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}", mininterval=10)
        for step, batch in enumerate(progress, start=1):
            batch = move_to_device(batch, self.device)
            with self._autocast():
                output = self.model(batch["input_ids"], batch["attention_mask"], batch["rel_pairs"])
            loss, parts = compute_multitask_loss(output, batch, cfg.relation_loss_weight, cfg.ner_loss_weight)
            self.scaler.scale(loss / cfg.gradient_accumulation_steps).backward()

            if step % cfg.gradient_accumulation_steps == 0 or step == steps:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.detach())
            if step % max(1, cfg.log_every_steps) == 0:
                progress.set_postfix(loss=f"{total_loss / step:.4f}", ner=f"{parts['ner_loss']:.3f}",
                                     rel=f"{parts['relation_loss']:.3f}")
        return total_loss / max(1, steps)

    # --------------------------------------------------------------- evaluate
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        """Span-level NER (seqeval) and relation P/R/F1, plus the paper's measures
        (accuracy, precision, recall, specificity, F1, AUC). Raw predictions of the
        last call are kept in `self.last_predictions` (used for ROC curves)."""
        self.model.eval()
        gold_tags: list[list[str]] = []
        pred_tags: list[list[str]] = []
        ner_probs: list[np.ndarray] = []
        gold_rel: list[int] = []
        pred_rel: list[int] = []
        rel_probs: list[np.ndarray] = []
        for batch in tqdm(loader, desc="Evaluating", mininterval=10, leave=False):
            batch = move_to_device(batch, self.device)
            with self._autocast():
                output = self.model(batch["input_ids"], batch["attention_mask"], batch["rel_pairs"])
            probabilities = torch.softmax(output.ner_logits.float(), dim=-1).cpu()
            predictions = probabilities.argmax(dim=-1)
            labels = batch["ner_labels"].cpu()
            for row in range(labels.size(0)):
                mask = labels[row] != IGNORE_INDEX
                gold_tags.append([self.ner_id2label[i] for i in labels[row][mask].tolist()])
                pred_tags.append([self.ner_id2label[i] for i in predictions[row][mask].tolist()])
                ner_probs.append(probabilities[row][mask].numpy())
            if output.rel_logits.numel() > 0:
                rel_probability = torch.softmax(output.rel_logits.float(), dim=-1).cpu()
                gold_rel.extend(batch["rel_labels"].cpu().tolist())
                pred_rel.extend(rel_probability.argmax(dim=-1).tolist())
                rel_probs.append(rel_probability.numpy())

        ner = ner_metrics(gold_tags, pred_tags)
        relations = relation_metrics(gold_rel, pred_rel, self.no_relation_id, self.rel_id2label)
        score = (ner["f1"] + relations["f1"]) / 2 if relations["gold_relations"] else ner["f1"]

        # --- Paper measures (word-level NER on IO types, relation pairs) -------------
        io_index = {name: i for i, name in enumerate(self.io_classes)}
        to_io = lambda tag: io_index[tag.split("-", 1)[1] if "-" in tag else tag]  # noqa: E731
        io_gold = [to_io(tag) for tags in gold_tags for tag in tags]
        io_pred = [to_io(tag) for tags in pred_tags for tag in tags]
        io_prob = (
            bio_to_io_probabilities(np.concatenate(ner_probs), self.ner_id2label, self.io_classes)
            if ner_probs else np.zeros((0, len(self.io_classes)))
        )
        rel_prob = np.concatenate(rel_probs) if rel_probs else np.zeros((0, len(self.rel_id2label)))
        paper_ner = classification_measures(io_gold, io_pred, io_prob, self.io_classes, negative_class=0)
        paper_rel = classification_measures(gold_rel, pred_rel, rel_prob, self.rel_id2label, self.no_relation_id)
        paper = {"ner": paper_ner, "relations": paper_rel, "overall": summarize(paper_ner, paper_rel)}

        self.last_predictions = {
            "ner": (io_gold, io_prob), "relations": (gold_rel, rel_prob),
        }
        return {"ner": ner, "relations": relations, "paper_metrics": paper, "score": score}

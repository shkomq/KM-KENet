"""Step 4 — KM-KENet Model Training.

Fine-tunes a multilingual encoder (XLM-R) on the weakly labelled dataset for
joint Named Entity Recognition and Relation Extraction, keeps the best
checkpoint on the dev split and reports test metrics (span-level F1 and the
paper's accuracy / precision / recall / specificity / F1 / AUC).

    python -m step4_model_training.main --config configs/config.yaml
"""

from __future__ import annotations

import itertools
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from common.cli import build_arg_parser, init_step
from common.config import PipelineConfig
from common.io_utils import read_jsonl, require_file, write_json
from common.utils import resolve_device, set_seed
from step4_model_training.data import KMKENetCollator, KMKENetDataset, LabelSpace
from step4_model_training.model import KMKENet
from step4_model_training.reporting import write_report
from step4_model_training.trainer import Trainer, TrainingConfig

STEP_NAME = "step4_model_training"


def load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return list(itertools.islice(read_jsonl(path), limit))


@dataclass
class TrainingSetup:
    """Everything needed to build datasets, loaders and a model from the config."""

    cfg: PipelineConfig
    model_cfg: Mapping[str, Any]
    train_cfg: TrainingConfig
    tokenizer: PreTrainedTokenizerBase
    label_space: LabelSpace
    encoder_name: str
    max_length: int
    device: torch.device

    @classmethod
    def from_config(cls, cfg: PipelineConfig) -> "TrainingSetup":
        model_cfg = cfg.section("model")
        encoder_name = str(model_cfg.get("encoder", "FacebookAI/xlm-roberta-large"))
        return cls(
            cfg=cfg,
            model_cfg=model_cfg,
            train_cfg=TrainingConfig.from_dict(cfg.section("training")),
            tokenizer=AutoTokenizer.from_pretrained(encoder_name, use_fast=True),
            label_space=LabelSpace.default(),
            encoder_name=encoder_name,
            max_length=int(model_cfg.get("max_length", 256)),
            device=resolve_device(cfg.device),
        )

    def dataset(self, records: list[dict[str, Any]], training: bool) -> KMKENetDataset:
        tc = self.train_cfg
        return KMKENetDataset(
            records,
            self.tokenizer,
            self.label_space,
            self.max_length,
            use_confidence_weights=tc.use_confidence_weights,
            max_negative_relations=tc.max_negative_relations if training else None,
            no_relation_weight=tc.no_relation_weight if training else 1.0,
            seed=self.cfg.seed,
        )

    def loader(self, dataset: KMKENetDataset | None, training: bool) -> DataLoader | None:
        if dataset is None or len(dataset) == 0:
            return None
        tc = self.train_cfg
        return DataLoader(
            dataset,
            batch_size=tc.batch_size if training else tc.eval_batch_size,
            shuffle=training,
            collate_fn=KMKENetCollator(self.tokenizer.pad_token_id),
            num_workers=tc.num_workers,
            generator=torch.Generator().manual_seed(self.cfg.seed) if training else None,
            pin_memory=self.device.type == "cuda",
        )

    def new_model(self) -> KMKENet:
        return KMKENet(
            self.encoder_name,
            self.label_space,
            dropout=float(self.model_cfg.get("dropout", 0.1)),
            relation_hidden_size=int(self.model_cfg.get("relation_hidden_size", 512)),
            type_embedding_dim=int(self.model_cfg.get("type_embedding_dim", 32)),
        )

    def label_names(self) -> dict[str, list[str]]:
        return {
            "ner": ["O", *self.label_space.entity_types],
            "relations": list(self.label_space.relation_labels),
        }

    def train(
        self,
        train_records: list[dict[str, Any]],
        dev_records: list[dict[str, Any]],
        output_dir: Path,
        logger: logging.Logger,
        train_cfg: TrainingConfig | None = None,
    ) -> Trainer:
        """Train a fresh model; the best checkpoint (on dev) is saved in `output_dir`."""
        set_seed(self.cfg.seed)
        tc = train_cfg or self.train_cfg
        train_set = self.dataset(train_records, training=True)
        dev_set = self.dataset(dev_records, training=False) if dev_records else None
        logger.info("Examples: train=%d dev=%d", len(train_set), len(dev_set) if dev_set else 0)
        if len(train_set) == 0:
            raise ValueError("No training example survived encoding")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "training_config.json", {
            "encoder": self.encoder_name,
            "max_length": self.max_length,
            "training": tc.__dict__,
        })
        trainer = Trainer(
            self.new_model(), self.tokenizer, self.loader(train_set, True), self.loader(dev_set, False),
            tc, self.device, output_dir,
        )
        summary = trainer.train()
        logger.info("Best epoch: %d (score %.4f)", summary["best_epoch"], summary["best_score"])
        # Reload the best checkpoint for evaluation.
        trainer.model = KMKENet.from_pretrained(output_dir).to(self.device)
        return trainer


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("Step 4 - train KM-KENet (joint NER + RE on XLM-R)")
    args = parser.parse_args(argv)
    cfg, logger = init_step(args, STEP_NAME)
    setup = TrainingSetup.from_config(cfg)
    tc = setup.train_cfg
    logger.info("Device: %s | encoder: %s", setup.device, setup.encoder_name)

    weak_dir = cfg.path("weak_dir")
    model_dir = cfg.path("model_dir")
    train_records = load_records(require_file(weak_dir / "train.jsonl", "Run step 3 first."), tc.max_train_examples)
    dev_records = load_records(weak_dir / "dev.jsonl", tc.max_eval_examples)
    test_records = load_records(weak_dir / "test.jsonl", tc.max_eval_examples)
    if not train_records:
        logger.error("The training split is empty")
        return 1

    trainer = setup.train(train_records, dev_records, model_dir, logger)

    test_loader = setup.loader(setup.dataset(test_records, training=False), False) if test_records else None
    if test_loader is not None:
        metrics = trainer.evaluate(test_loader)
        evaluation = cfg.raw.get("evaluation", {}) or {}
        write_report(metrics, trainer.last_predictions, setup.label_names(), model_dir, "test",
                     roc_curves=bool(evaluation.get("roc_curve", True)))
        logger.info(
            "Test | NER P=%.4f R=%.4f F1=%.4f | RE P=%.4f R=%.4f F1=%.4f",
            metrics["ner"]["precision"], metrics["ner"]["recall"], metrics["ner"]["f1"],
            metrics["relations"]["precision"], metrics["relations"]["recall"], metrics["relations"]["f1"],
        )
    logger.info("Model saved in %s", model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

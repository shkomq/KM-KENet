"""Evaluate a trained KM-KENet model on the dev or test split.

Writes `<split>_metrics.json`, `<split>_results.md` (paper measures: accuracy,
precision, recall, specificity, F1-score, AUC) and ROC curve images to the
model directory.

    python -m step4_model_training.evaluate --config configs/config.yaml
    python -m step4_model_training.evaluate --split dev
"""

from __future__ import annotations

import sys

from common.cli import build_arg_parser, init_step
from common.io_utils import require_file
from step4_model_training.main import TrainingSetup, load_records
from step4_model_training.model import KMKENet
from step4_model_training.reporting import write_report
from step4_model_training.trainer import Trainer

STEP_NAME = "step4_evaluate"


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser("Evaluate a trained KM-KENet model (paper metrics)")
    parser.add_argument("--split", choices=("dev", "test"), default=None, help="Default: evaluation.split")
    args = parser.parse_args(argv)
    cfg, logger = init_step(args, STEP_NAME)
    evaluation = cfg.raw.get("evaluation", {}) or {}
    split = args.split or str(evaluation.get("split", "test"))

    model_dir = cfg.path("model_dir")
    require_file(model_dir / KMKENet.CONFIG_NAME, "Train the model first (step 4).")
    setup = TrainingSetup.from_config(cfg)
    records = load_records(
        require_file(cfg.path("weak_dir") / f"{split}.jsonl", "Run step 3 first."),
        setup.train_cfg.max_eval_examples,
    )
    loader = setup.loader(setup.dataset(records, training=False), False)
    if loader is None:
        logger.error("The %s split is empty", split)
        return 1

    model = KMKENet.from_pretrained(model_dir)
    trainer = Trainer(model, setup.tokenizer, loader, None, setup.train_cfg, setup.device, model_dir)
    metrics = trainer.evaluate(loader)
    write_report(metrics, trainer.last_predictions, setup.label_names(), model_dir, split,
                 roc_curves=bool(evaluation.get("roc_curve", True)))
    logger.info("Results written to %s", model_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

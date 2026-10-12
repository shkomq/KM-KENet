"""Command-line scaffolding shared by every step's `main.py`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common.config import DEFAULT_CONFIG_PATH, PipelineConfig, load_config
from common.logging_utils import setup_logging
from common.utils import set_seed


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config file")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. training.epochs=3 (repeatable)",
    )
    parser.add_argument("--log-level", default=None, help="Overrides project.log_level")
    return parser


def init_step(args: argparse.Namespace, step_name: str) -> tuple[PipelineConfig, logging.Logger]:
    cfg = load_config(args.config, args.override)
    level = args.log_level or cfg.raw.get("project", {}).get("log_level", "INFO")
    setup_logging(level, cfg.path("logs_dir") / f"{step_name}.log")
    set_seed(cfg.seed)
    logger = logging.getLogger(step_name)
    logger.info("Configuration: %s", args.config)
    for item in args.override:
        logger.info("Override: %s", item)
    return cfg, logger

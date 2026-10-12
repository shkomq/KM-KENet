"""Download every external asset KM-KENet needs.

Assets (names are read from configs/config.yaml):

* dataset    — kurdish_medical_corpus_kmc.json from the Hugging Face dataset
               `data_collection.kmc_json.hf_repo_id`, saved to `paths.corpus_root`
               (only the JSON is fetched, not the 1.3 GB of book PDFs)
* encoder    — `model.encoder` (XLM-R) used by step 4
* embedding  — `retrieval.embedding_model` (multilingual E5) used by step 6
* aya        — `retrieval.aya.hf_model` (Aya Expanse, ~16 GB, gated) used by step 6

Models go to the standard Hugging Face cache (`$HF_HOME`, default
~/.cache/huggingface), so `from_pretrained("<repo id>")` in the pipeline finds
them without further configuration and the pipeline can run offline afterwards
(`export HF_HUB_OFFLINE=1`).

    python -m setup.download_assets                     # everything enabled in `assets`
    python -m setup.download_assets --only dataset
    python -m setup.download_assets --skip aya
    HF_TOKEN=hf_xxx python -m setup.download_assets     # needed for gated / private repos
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from common.config import DEFAULT_CONFIG_PATH, PipelineConfig, load_config
from common.logging_utils import setup_logging

logger = logging.getLogger("setup.download_assets")

ASSET_KINDS = ("dataset", "encoder", "embedding", "aya")
REQUIRED_ASSETS = frozenset({"dataset", "encoder"})  # step 6's assets have fallbacks

# Files a transformers / sentence-transformers checkpoint needs (weights are added per repo).
MODEL_ALLOW_PATTERNS = ["*.json", "*.txt", "*.model", "tokenizer*", "*.tiktoken"]
MODEL_IGNORE_PATTERNS = [
    "onnx/*", "openvino/*", "coreml/*", "*.onnx", "*.h5", "*.msgpack", "*.ot", "*.tflite", "original/*",
]


@dataclass
class AssetResult:
    kind: str
    name: str
    ok: bool
    detail: str


def _resolve_token(cli_token: str | None) -> str | None:
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _matches(path: str, allow: list[str], ignore: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in allow) and not any(fnmatch.fnmatch(path, p) for p in ignore)


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _check_disk_space(target: Path, required: int) -> None:
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    if required and free < required * 1.1:
        raise RuntimeError(
            f"Not enough disk space in {target}: need ~{_format_size(required)}, have {_format_size(free)}"
        )


def download_dataset(cfg: PipelineConfig, token: str | None, force: bool) -> AssetResult:
    from huggingface_hub import snapshot_download

    section = cfg.section("data_collection")
    kmc = section.get("kmc_json", {}) or {}
    repo_id = str(kmc.get("hf_repo_id", "shkomq/Kurdish_Medical_Corpus_KMC"))
    corpus_root = cfg.path("corpus_root")

    files = [str(name) for name in kmc.get("files", [])]
    patterns = [*files, "README.md"]
    patterns += [f"{directory}/*.json" for directory in kmc.get("directories", []) or []]
    books = section.get("kmc_books", {}) or {}
    if books.get("enabled", False):
        patterns += [f"{directory}/*.txt" for directory in books.get("directories", []) or []]

    missing = [name for name in files if not (corpus_root / name).is_file()]
    if not missing and not force:
        return AssetResult("dataset", repo_id, True, f"already present in {corpus_root}")

    logger.info("Downloading %s (%s) into %s", repo_id, ", ".join(patterns), corpus_root)
    _check_disk_space(corpus_root, 300 * 1024**2)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(corpus_root),
        allow_patterns=patterns,
        token=token,
    )
    still_missing = [name for name in files if not (corpus_root / name).is_file()]
    if still_missing:
        return AssetResult("dataset", repo_id, False, f"files not found in the repository: {still_missing}")
    sizes = ", ".join(f"{name} ({_format_size((corpus_root / name).stat().st_size)})" for name in files)
    return AssetResult("dataset", repo_id, True, f"saved {sizes} to {corpus_root}")


def download_model(kind: str, repo_id: str, token: str | None, force: bool) -> AssetResult:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    if Path(repo_id).expanduser().exists():
        return AssetResult(kind, repo_id, True, "local path, nothing to download")

    api = HfApi(token=token)
    info = api.model_info(repo_id, files_metadata=True)
    files = {sibling.rfilename: (sibling.size or 0) for sibling in info.siblings or []}
    has_safetensors = any(name.endswith(".safetensors") for name in files)
    allow = [*MODEL_ALLOW_PATTERNS, "*.safetensors" if has_safetensors else "*.bin"]
    selected = {name: size for name, size in files.items() if _matches(name, allow, MODEL_IGNORE_PATTERNS)}
    total = sum(selected.values())

    logger.info("Downloading %s: %d files, %s", repo_id, len(selected), _format_size(total))
    _check_disk_space(Path(HF_HUB_CACHE), total)
    path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=allow,
        ignore_patterns=MODEL_IGNORE_PATTERNS,
        token=token,
        force_download=force,
    )
    return AssetResult(kind, repo_id, True, f"{_format_size(total)} cached at {path}")


def plan_assets(cfg: PipelineConfig, only: list[str], skip: list[str]) -> list[tuple[str, str]]:
    assets_cfg = cfg.raw.get("assets", {}) or {}
    flags = {
        "dataset": assets_cfg.get("download_dataset", True),
        "encoder": assets_cfg.get("download_encoder", True),
        "embedding": assets_cfg.get("download_embedding_model", True),
        "aya": assets_cfg.get("download_aya", True),
    }
    retrieval = cfg.section("retrieval")
    aya_cfg = retrieval.get("aya", {}) or {}
    names = {
        "dataset": str(cfg.section("data_collection").get("kmc_json", {}).get("hf_repo_id", "")),
        "encoder": str(cfg.section("model").get("encoder", "FacebookAI/xlm-roberta-large")),
        "embedding": str(retrieval.get("embedding_model", "intfloat/multilingual-e5-base")),
        "aya": str(aya_cfg.get("hf_model", "CohereLabs/aya-expanse-8b")),
    }
    if str(aya_cfg.get("backend", "hf")).lower() != "hf" and not only:
        flags["aya"] = False  # Aya is not loaded locally with the cohere / none backends
    if not retrieval.get("use_dense_retrieval", True) and not only:
        flags["embedding"] = False

    selected = only or [kind for kind in ASSET_KINDS if flags[kind]]
    return [(kind, names[kind]) for kind in ASSET_KINDS if kind in selected and kind not in skip]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the KMC dataset and the pre-trained models")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--only", action="append", choices=ASSET_KINDS, default=[], help="Download only these")
    parser.add_argument("--skip", action="append", choices=ASSET_KINDS, default=[], help="Skip these")
    parser.add_argument("--token", default=None, help="Hugging Face token (default: $HF_TOKEN)")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging(str(cfg.raw.get("project", {}).get("log_level", "INFO")), cfg.path("logs_dir") / "setup.log")
    token = _resolve_token(args.token)
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        logger.error("HF_HUB_OFFLINE=1 is set; unset it to download assets")
        return 1

    plan = plan_assets(cfg, args.only, args.skip)
    if not plan:
        logger.warning("Nothing to download")
        return 0

    results: list[AssetResult] = []
    for kind, name in plan:
        action: Callable[[], AssetResult] = (
            (lambda: download_dataset(cfg, token, args.force))
            if kind == "dataset"
            else (lambda kind=kind, name=name: download_model(kind, name, token, args.force))
        )
        try:
            results.append(action())
        except Exception as exc:  # network, auth, gated repo, disk space ...
            hint = ""
            text = str(exc).lower()
            if "gated" in text or "401" in text or "403" in text or "access" in text:
                hint = (f" — request access at https://huggingface.co/{name} and pass a token "
                        "(export HF_TOKEN=... or `huggingface-cli login`)")
            results.append(AssetResult(kind, name, False, f"{type(exc).__name__}: {exc}{hint}"))
        logger.info("[%s] %s: %s", "OK" if results[-1].ok else "FAILED", kind, results[-1].detail)

    print("\nDownload summary")
    print("-" * 80)
    for result in results:
        print(f"{'OK    ' if result.ok else 'FAILED'} {result.kind:<10} {result.name}\n       {result.detail}")
    print("-" * 80)

    failed_required = [r for r in results if not r.ok and r.kind in REQUIRED_ASSETS]
    failed_optional = [r for r in results if not r.ok and r.kind not in REQUIRED_ASSETS]
    if failed_optional:
        print("Optional assets failed: step 6 will fall back to BM25 / lexicon translation / template answers.")
    if failed_required:
        print("Required assets failed: steps 1-4 cannot run until they are downloaded.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

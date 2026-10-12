#!/usr/bin/env bash
# =============================================================================
# KM-KENet installer
#
#   1. checks Python (>= 3.10)
#   2. creates a virtual environment
#   3. installs PyTorch (CPU / Apple MPS / CUDA) and requirements.txt
#   4. verifies the installation (imports + available devices)
#   5. downloads the KMC dataset and the pre-trained models
#      (XLM-R, multilingual E5, Aya Expanse) via `python -m setup.download_assets`
#
# Examples
#   ./install.sh                              # .venv + default PyTorch wheel + all assets
#   ./install.sh --cuda cu121                 # PyTorch built for CUDA 12.1
#   ./install.sh --skip-aya                   # do not download the 16 GB Aya model
#   ./install.sh --with-4bit --with-cohere    # optional extras
#   HF_TOKEN=hf_xxx ./install.sh              # token for gated models (Aya)
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"
CUDA_TAG=""
DOWNLOAD=1
WITH_4BIT=0
WITH_COHERE=0
DOWNLOAD_ARGS=()

usage() {
  cat <<EOF
Usage: ./install.sh [options]

Options:
  --python BIN        Python interpreter used to create the venv (default: ${PYTHON_BIN})
  --venv DIR          Virtual environment directory (default: ${VENV_DIR}); "none" = current env
  --cuda TAG          Install PyTorch for a CUDA version, e.g. cu118, cu121, cu124
  --with-4bit         Also install bitsandbytes (4-bit Aya on CUDA)
  --with-cohere       Also install the Cohere SDK (retrieval.aya.backend: cohere)
  --skip-aya          Do not download the Aya model (~16 GB)
  --no-download       Install packages only; do not download the dataset/models
  --token TOKEN       Hugging Face token (default: \$HF_TOKEN)
  -h, --help          Show this help
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }
log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
require_value() { [[ $# -ge 2 && -n "$2" ]] || die "Option $1 requires a value"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)       require_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --venv)         require_value "$@"; VENV_DIR="$2"; shift 2 ;;
    --cuda)         require_value "$@"; CUDA_TAG="$2"; shift 2 ;;
    --with-4bit)    WITH_4BIT=1; shift ;;
    --with-cohere)  WITH_COHERE=1; shift ;;
    --skip-aya)     DOWNLOAD_ARGS+=("--skip" "aya"); shift ;;
    --no-download)  DOWNLOAD=0; shift ;;
    --token)        require_value "$@"; export HF_TOKEN="$2"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *)              usage >&2; die "Unknown option: $1" ;;
  esac
done

trap 'echo "Installation failed (line $LINENO)." >&2' ERR

# --- 1. Python ---------------------------------------------------------------
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python interpreter not found: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY' || die "Python 3.10 or newer is required"
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
log "Using $("$PYTHON_BIN" --version 2>&1)"

# --- 2. Virtual environment --------------------------------------------------
if [[ "$VENV_DIR" != "none" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating virtual environment in ${VENV_DIR}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  PYTHON_BIN="python"
fi

# --- 3. Packages -------------------------------------------------------------
log "Upgrading pip"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

if [[ -n "$CUDA_TAG" ]]; then
  log "Installing PyTorch for ${CUDA_TAG}"
  "$PYTHON_BIN" -m pip install "torch>=2.3" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
else
  log "Installing PyTorch (default wheel: CUDA on Linux, MPS on Apple Silicon, CPU elsewhere)"
  "$PYTHON_BIN" -m pip install "torch>=2.3"
fi

log "Installing requirements.txt"
"$PYTHON_BIN" -m pip install -r requirements.txt
if (( WITH_4BIT )); then
  log "Installing bitsandbytes"
  "$PYTHON_BIN" -m pip install "bitsandbytes>=0.43"
fi
if (( WITH_COHERE )); then
  log "Installing the Cohere SDK"
  "$PYTHON_BIN" -m pip install "cohere>=5.5"
fi

# --- 4. Verification ---------------------------------------------------------
log "Verifying the installation"
"$PYTHON_BIN" - <<'PY'
import importlib

modules = [
    "numpy", "yaml", "tqdm", "requests", "bs4", "datasets", "torch", "transformers", "tokenizers",
    "sentencepiece", "accelerate", "seqeval", "networkx", "matplotlib", "rank_bm25",
    "sentence_transformers", "huggingface_hub", "safetensors",
]
missing = []
for name in modules:
    try:
        module = importlib.import_module(name)
        print(f"  {name:<22} {getattr(module, '__version__', 'ok')}")
    except Exception as exc:
        missing.append(f"{name} ({exc})")

import torch

print(f"  CUDA available         {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
print(f"  MPS available          {getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}")
if missing:
    raise SystemExit("Missing packages: " + ", ".join(missing))
PY

# --- 5. Dataset and models ---------------------------------------------------
if (( DOWNLOAD )); then
  log "Downloading the dataset and pre-trained models"
  export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  "$PYTHON_BIN" -m setup.download_assets --config configs/config.yaml \
    ${DOWNLOAD_ARGS[@]+"${DOWNLOAD_ARGS[@]}"}
fi

log "Installation complete."
if [[ "$VENV_DIR" != "none" ]]; then
  echo "Activate the environment with:  source ${VENV_DIR}/bin/activate"
  echo "Run the pipeline with:           ./run.sh --venv ${VENV_DIR}"
else
  echo "Run the pipeline with:           ./run.sh"
fi

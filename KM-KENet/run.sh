#!/usr/bin/env bash
# =============================================================================
# KM-KENet — end-to-end pipeline runner
#
#   1. Data Collection            (step1_data_collection)
#   2. Data Pre-processing        (step2_preprocessing)
#   3. Weak Supervision Labeling  (step3_weak_supervision)
#   4. KM-KENet Model Training    (step4_model_training)
#   5. Knowledge Graph            (step5_knowledge_graph)
#   6. Cross-Lingual Retrieval    (step6_cross_lingual_retrieval)
#
# Examples
#   ./run.sh                                  # run all six steps
#   ./install.sh && ./run.sh --venv .venv     # full install (packages + dataset + models), then run
#   ./run.sh --install --download --venv .venv  # same, from run.sh
#   ./run.sh --from 4                         # resume from model training
#   ./run.sh --only 6 -q "How is thalassemia treated?"
#   ./run.sh --override training.epochs=2 --override retrieval.aya.backend=none
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="configs/config.yaml"
FROM_STEP=1
TO_STEP=6
INSTALL=0
DOWNLOAD=0
VENV_DIR=""
QUERY_ARGS=()
OVERRIDE_ARGS=()

STEP_MODULES=(
  ""
  "step1_data_collection"
  "step2_preprocessing"
  "step3_weak_supervision"
  "step4_model_training"
  "step5_knowledge_graph"
  "step6_cross_lingual_retrieval"
)
STEP_TITLES=(
  ""
  "Data Collection"
  "Data Pre-processing"
  "Weak Supervision Labeling"
  "KM-KENet Model Training"
  "Knowledge Graph Construction"
  "Cross-Lingual Query Translation and Retrieval"
)

usage() {
  cat <<EOF
Usage: ./run.sh [options]

Options:
  -c, --config FILE      Configuration file (default: ${CONFIG})
      --from N           First step to run, 1-6 (default: 1)
      --to N             Last step to run, 1-6 (default: 6)
      --only N           Run a single step
  -q, --query TEXT       Question for step 6 (repeatable; default: retrieval.default_query)
      --override K=V     Override a config value, e.g. training.epochs=3 (repeatable)
      --install          pip-install requirements.txt before running (see also ./install.sh)
      --download         Download the KMC dataset and pre-trained models before running
      --venv DIR         Create/activate a virtual environment in DIR
  -h, --help             Show this help

Environment:
  PYTHON_BIN             Python interpreter (default: python3)
  HF_TOKEN               Hugging Face token (gated models such as Aya)
  COHERE_API_KEY         Needed only with retrieval.aya.backend=cohere
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

require_value() {
  [[ $# -ge 2 && -n "$2" ]] || die "Option $1 requires a value"
}

is_step() { [[ "$1" =~ ^[1-6]$ ]]; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)  require_value "$@"; CONFIG="$2"; shift 2 ;;
    --from)       require_value "$@"; FROM_STEP="$2"; shift 2 ;;
    --to)         require_value "$@"; TO_STEP="$2"; shift 2 ;;
    --only)       require_value "$@"; FROM_STEP="$2"; TO_STEP="$2"; shift 2 ;;
    -q|--query)   require_value "$@"; QUERY_ARGS+=("--query" "$2"); shift 2 ;;
    --override)   require_value "$@"; OVERRIDE_ARGS+=("--override" "$2"); shift 2 ;;
    --install)    INSTALL=1; shift ;;
    --download)   DOWNLOAD=1; shift ;;
    --venv)       require_value "$@"; VENV_DIR="$2"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *)            usage >&2; die "Unknown option: $1" ;;
  esac
done

is_step "$FROM_STEP" || die "--from must be between 1 and 6 (got '$FROM_STEP')"
is_step "$TO_STEP"   || die "--to must be between 1 and 6 (got '$TO_STEP')"
(( FROM_STEP <= TO_STEP )) || die "--from ($FROM_STEP) must not be greater than --to ($TO_STEP)"
[[ -f "$CONFIG" ]] || die "Configuration file not found: $CONFIG"

# --- Logging -----------------------------------------------------------------
mkdir -p logs
LOG_FILE="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

CURRENT_STEP=""
on_error() {
  local code=$?
  log "FAILED${CURRENT_STEP:+ during step ${CURRENT_STEP}} (exit code ${code}). See ${LOG_FILE} and logs/."
  exit "$code"
}
trap on_error ERR

# --- Python environment ------------------------------------------------------
if [[ -n "$VENV_DIR" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating virtual environment in ${VENV_DIR}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  PYTHON_BIN="python"
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python interpreter not found: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY' || die "Python 3.10 or newer is required"
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY

if (( INSTALL )); then
  log "Installing Python dependencies"
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if (( DOWNLOAD )); then
  CURRENT_STEP="0 (asset download)"
  log "Downloading the KMC dataset and pre-trained models"
  "$PYTHON_BIN" -m setup.download_assets --config "$CONFIG"
fi

# --- Run the steps -----------------------------------------------------------
run_step() {
  local step="$1"
  shift
  local module="${STEP_MODULES[$step]}"
  CURRENT_STEP="${step} (${STEP_TITLES[$step]})"
  log "=================================================================="
  log "Step ${step}/6 — ${STEP_TITLES[$step]}"
  log "=================================================================="
  local started
  started=$(date +%s)
  "$PYTHON_BIN" -m "${module}.main" --config "$CONFIG" \
    ${OVERRIDE_ARGS[@]+"${OVERRIDE_ARGS[@]}"} "$@"
  log "Step ${step} finished in $(( $(date +%s) - started ))s"
}

log "KM-KENet pipeline: steps ${FROM_STEP} to ${TO_STEP} | config: ${CONFIG} | python: $("$PYTHON_BIN" --version 2>&1)"
PIPELINE_START=$(date +%s)

for (( step = FROM_STEP; step <= TO_STEP; step++ )); do
  if (( step == 6 )); then
    run_step "$step" ${QUERY_ARGS[@]+"${QUERY_ARGS[@]}"}
  else
    run_step "$step"
  fi
done

CURRENT_STEP=""
log "Pipeline completed in $(( $(date +%s) - PIPELINE_START ))s. Log: ${LOG_FILE}"

#!/usr/bin/env bash
# =============================================================================
# run2.sh - One Amazon sequential-recommendation agent with four parallel trials.
#
# Main Amazon experiment on four datasets:
#   Movies_and_TV / Industrial_and_Scientific / Electronics / CDs_and_Vinyl
# The default cold-start template trains one model per dataset.
# Each iteration proposes four hypotheses and executes them in parallel.
#
# VOLCENGINE_API_KEY must be provided through the environment.
# LangSmith tracing is optional and disabled when LANGSMITH_API_KEY is empty.
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_head()  { echo -e "\n${BOLD}${GREEN}$*${NC}"; }
die()       { log_error "$*"; exit 1; }

# ---- LangSmith tracing (optional; skipped if LANGSMITH_API_KEY is empty) ----
export LANGSMITH_ENDPOINT="${LANGSMITH_ENDPOINT:-https://api.smith.langchain.com}"
export LANGSMITH_API_KEY="${LANGSMITH_API_KEY:-}"
LANGSMITH_PROJECT_PREFIX="${LANGSMITH_PROJECT_PREFIX:-${LANGSMITH_PROJECT:-recharness}}"
if [[ -n "${LANGSMITH_API_KEY}" ]]; then
    export LANGCHAIN_TRACING_V2="${LANGCHAIN_TRACING_V2:-true}"
else
    export LANGCHAIN_TRACING_V2="false"
fi

BUDGET=43200
TRIAL_SECS=10800
BASIN_TRANSFER_RHO=0.25
RUN_ID=""
BASE_DATA_DIR="./input"
RAW_DATA_DIR=""
USE_PREPARED_SPLITS=false
COLD_START="sasrec_perdataset"
GPU_IDS_ARG=""
CPUS=64
USE_THOMPSON_SAMPLING=true

usage() {
    cat <<EOF
Usage: $0 --gpus '0,1,2,3' [options]

Options:
  --budget SECONDS        Total GPU/compute-time budget (default: ${BUDGET})
  --trial-secs SECONDS    Per-trial floor budget (default: ${TRIAL_SECS})
  --rho FLOAT             Basin jump posterior transfer coefficient (default: ${BASIN_TRANSFER_RHO})
  --run-id ID
  --data-dir DIR          Output root containing trainval/ and test/ (default: ${BASE_DATA_DIR})
  --raw-data-dir DIR      Local Amazon raw-data root used for preprocessing
  --prepared-data         Use existing split files; skip preprocessing
  --cold-start NAME       Cold-start template (default: ${COLD_START})
  --gpus '0,1,2,3'        Exactly 4 GPU ids for parallel trials (required)
  --cpus N                CPU cores available (default: ${CPUS})
  --routing-policy thompson|textual_gradient
  --no-thompson           Alias for --routing-policy textual_gradient
  --textual-gradient      Alias for --routing-policy textual_gradient
  --help|-h
EOF
}

set_routing_policy() {
    local policy
    policy="$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
    case "${policy}" in
        thompson|ts|on|true|1) USE_THOMPSON_SAMPLING=true ;;
        textual_gradient|textual|llm|llm_memory|memory|off|false|0) USE_THOMPSON_SAMPLING=false ;;
        *) die "Unknown routing policy: $1 (choose thompson or textual_gradient)" ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --budget)        BUDGET="$2";        shift 2 ;;
        --trial-secs)    TRIAL_SECS="$2";    shift 2 ;;
        --rho)           BASIN_TRANSFER_RHO="$2"; shift 2 ;;
        --run-id)        RUN_ID="$2";        shift 2 ;;
        --data-dir)      BASE_DATA_DIR="$2"; shift 2 ;;
        --raw-data-dir)  RAW_DATA_DIR="$2"; shift 2 ;;
        --prepared-data)  USE_PREPARED_SPLITS=true; shift ;;
        --cold-start)    COLD_START="$2";    shift 2 ;;
        --gpus)          GPU_IDS_ARG="$2";   shift 2 ;;
        --cpus)          CPUS="$2";          shift 2 ;;
        --routing-policy) set_routing_policy "$2"; shift 2 ;;
        --no-thompson|--textual-gradient) USE_THOMPSON_SAMPLING=false; shift ;;
        --help|-h)        usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

if [[ "${USE_THOMPSON_SAMPLING}" == true ]]; then
    ROUTING_POLICY="thompson"
    USE_THOMPSON_SAMPLING_PY="True"
else
    ROUTING_POLICY="textual_gradient"
    USE_THOMPSON_SAMPLING_PY="False"
fi

[[ -z "$RUN_ID" ]] && RUN_ID="exp2-${COLD_START}-$(date +%Y%m%d-%H%M%S)"

LANGSMITH_PROJECT="${LANGSMITH_PROJECT_PREFIX}-${COLD_START}"
export LANGSMITH_PROJECT
export LANGCHAIN_PROJECT="${LANGSMITH_PROJECT}"

TRAINVAL_DIR="${BASE_DATA_DIR}/trainval"
TEST_DIR="${BASE_DATA_DIR}/test"
LOGS_DIR="./logs/${RUN_ID}"
RESULTS_DIR="./results/${RUN_ID}"
WORKSPACE="./workspace/${RUN_ID}/${COLD_START}"

mkdir -p "${TRAINVAL_DIR}" "${TEST_DIR}" "${LOGS_DIR}" "${RESULTS_DIR}" "${WORKSPACE}"

DATASETS=("Movies_and_TV" "Industrial_and_Scientific" "Electronics" "CDs_and_Vinyl")
PYTHON=$(command -v python3 || command -v python || die "python3 was not found")
PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ══════════════════════════════════════════════════════════════════════
log_head "===== Step 1/4: Environment check ====="
# ══════════════════════════════════════════════════════════════════════

log_info "Python: $($PYTHON --version)"
log_info "Project root: ${PROJ_ROOT}"

# Check LLM credentials.
[[ -n "${VOLCENGINE_API_KEY:-${ARK_API_KEY:-}}" ]] \
    || die "VOLCENGINE_API_KEY is not set. Export it before running this script."

if ! $PYTHON -c "import gagc" 2>/dev/null; then
    log_info "The legacy-compatible gagc package is not installed; running uv pip install -e ."
    uv pip install -e "${PROJ_ROOT}" -q || die "uv pip install -e . failed"
fi
$PYTHON -c "import gagc" 2>/dev/null || die "The gagc compatibility package could not be imported"
$PYTHON -c "import torch" 2>/dev/null || die "PyTorch is not installed"
$PYTHON -c "import langsmith" 2>/dev/null || { uv pip install langsmith -q; }
$PYTHON -c "import langchain_openai" 2>/dev/null || { uv pip install langchain-openai -q; }

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
if [[ "$GPU_COUNT" -eq 0 ]]; then
    log_warn "No GPU was detected; running in CPU mode"
    GPU_IDS_PY="[]"
    GPU_IDS_DISPLAY="none"
else
    if [[ -z "$GPU_IDS_ARG" ]]; then
        die "--gpus is required, for example: --gpus '0,1,2,3'"
    fi
    # Normalize full-width Chinese comma: "4，5，6，7" -> "4,5,6,7"
    GPU_IDS_CSV="$(echo "$GPU_IDS_ARG" | sed 's/，/,/g')"
    [[ "$GPU_IDS_CSV" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
        die "Invalid --gpus value: ${GPU_IDS_ARG}; use a value such as '4,5,6,7'"

    IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS_CSV"
    [[ "${#GPU_ID_ARRAY[@]}" -eq 4 ]] || \
        die "run2.sh requires exactly four GPUs; received ${#GPU_ID_ARRAY[@]}: ${GPU_IDS_CSV}"
    for GPU_ID in "${GPU_ID_ARRAY[@]}"; do
        (( GPU_ID >= 0 && GPU_ID < GPU_COUNT )) || \
            die "GPU ${GPU_ID} is out of range; ${GPU_COUNT} GPUs were detected"
    done

    # Convert "4,5,6,7" -> "[4, 5, 6, 7]"
    GPU_IDS_PY="[$(echo "$GPU_IDS_CSV" | sed 's/,/, /g')]"
    GPU_IDS_DISPLAY="$GPU_IDS_CSV"
    log_ok "Detected ${GPU_COUNT} GPUs; using ${GPU_IDS_DISPLAY} for parallel trials"
fi

log_info "Run ID: ${RUN_ID}"
log_info "Mode: one agent with four parallel hypotheses per iteration"
log_info "LangSmith project: ${LANGSMITH_PROJECT}$([[ -z "${LANGSMITH_API_KEY}" ]] && echo ' (tracing disabled)')"
log_info "GPU     : ${GPU_IDS_DISPLAY}  |  Budget: ${BUDGET}s  |  TrialFloor: ${TRIAL_SECS}s  |  ColdStart: ${COLD_START}  |  Routing: ${ROUTING_POLICY}"
log_ok "Environment check passed"

# ══════════════════════════════════════════════════════════════════════
log_head "===== Step 2/4: Local data preparation ====="
# ══════════════════════════════════════════════════════════════════════

if [[ "$USE_PREPARED_SPLITS" == true ]]; then
    log_info "Using existing split files"
    for DS in "${DATASETS[@]}"; do
        [[ -f "${TRAINVAL_DIR}/${DS}_train.txt" ]] || \
            die "Missing split file: ${TRAINVAL_DIR}/${DS}_train.txt"
    done
    log_ok "Existing split files validated"
else
    PREPROCESS_SCRIPT="${PROJ_ROOT}/gagc/data_preprocess.py"
    [[ -f "${PREPROCESS_SCRIPT}" ]] || die "Could not find ${PREPROCESS_SCRIPT}"
    [[ -n "${RAW_DATA_DIR}" ]] || die "--raw-data-dir is required unless --prepared-data is used"
    $PYTHON -c "import pandas, tqdm" 2>/dev/null || {
        uv pip install pandas tqdm -q
    }
    for DS in "${DATASETS[@]}"; do
        if [[ -f "${TRAINVAL_DIR}/${DS}_train.txt" && \
              -f "${TRAINVAL_DIR}/${DS}_valid.txt" && \
              -f "${TEST_DIR}/${DS}_test.txt" ]]; then
            log_info "  ${DS} already exists; skipping"
            continue
        fi
        log_info "  Processing ${DS}"
        $PYTHON "${PREPROCESS_SCRIPT}" \
            --dataset  "${DS}" \
            --local-dir "${RAW_DATA_DIR}" \
            --data_dir "${TRAINVAL_DIR}" \
            --test_dir "${TEST_DIR}" \
            || die "Preprocessing failed for ${DS}"
    done

    log_info "Computing dataset_stats.json"
    $PYTHON "${PROJ_ROOT}/gagc/compute_dataset_stats.py" \
        --data_dir "${TRAINVAL_DIR}" \
        --test_dir "${TEST_DIR}" \
        --output_dir "${TRAINVAL_DIR}" \
        || die "compute_dataset_stats.py failed"

    log_ok "Data preparation completed"
fi

# ══════════════════════════════════════════════════════════════════════
log_head "===== Step 3/4: Start ${COLD_START} agent with four parallel trials ====="
# ══════════════════════════════════════════════════════════════════════

# CPU affinity: cores 0 through CPUS-1.
CPU_START=0
CPU_END=$((CPUS - 1))

AGENT_LOG="${LOGS_DIR}/agent.log"

AGENT_PY="$(mktemp /tmp/gagc_agent_XXXXXX.py)"

cat > "${AGENT_PY}" << PYEOF
import os, sys
sys.path.insert(0, "${PROJ_ROOT}")

os.environ["LANGSMITH_ENDPOINT"]   = "${LANGSMITH_ENDPOINT}"
os.environ["LANGSMITH_API_KEY"]    = "${LANGSMITH_API_KEY}"
os.environ["LANGSMITH_PROJECT"]    = "${LANGSMITH_PROJECT}"
os.environ["LANGCHAIN_PROJECT"]    = "${LANGSMITH_PROJECT}"
os.environ["LANGCHAIN_TRACING_V2"] = "${LANGCHAIN_TRACING_V2}"
os.environ["GAGC_BASIN_TRANSFER_RHO"] = "${BASIN_TRANSFER_RHO}"
os.environ["GAGC_TRIAL_SECS"] = "${TRIAL_SECS}"

os.environ["LANGCHAIN_RUN_NAME"]   = "${RUN_ID}/${COLD_START}"

# CPU affinity: 0 .. CPUS-1
try:
    os.sched_setaffinity(0, set(range(${CPU_START}, ${CPU_END} + 1)))
except (AttributeError, OSError):
    pass

from gagc.agent import create_gagc_agent

agent = create_gagc_agent(
    llm_provider       = "volcengine",
    model_id           = "glm-5.2",
    data_dir           = "${TRAINVAL_DIR}",
    test_dir           = "${TEST_DIR}",
    cold_start         = "${COLD_START}",
    workspace_root     = "${WORKSPACE}",
    logs_root          = "${LOGS_DIR}",
    global_budget_secs = ${BUDGET},
    num_gpus           = 4,
    num_cpus           = ${CPUS},
    gpu_ids            = ${GPU_IDS_PY},
    use_thompson_sampling = ${USE_THOMPSON_SAMPLING_PY},
    basin_transfer_rho = ${BASIN_TRANSFER_RHO},
)

_cold = "${COLD_START}"
_perdataset_hint = (
    " Note: train.py trains one model per dataset, stores checkpoints as "
    "working/{dataset}_model.pt, and predict.py routes by dataset_name."
    if _cold in {"perdataset", "sasrec_perdataset", "hstu_perdataset"} else ""
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": (
        "Optimize the sequential recommender to maximize HR@10 across the four Amazon datasets.\n"
        f"Template: {_cold}. Workspace: ${WORKSPACE}."
        + _perdataset_hint + "\n\n"
        "## Workflow\n"
        "Step 1 is mandatory: run the unmodified cold-start template once with execute_trial "
        "and record the baseline validation HR@10.\n"
        "From Step 2 onward, use propose_action_group followed by execute_trial_group to explore "
        "four optimization hypotheses in parallel and improve HR@10.\n"
        "Run all four hypotheses through execute_trial_group each iteration on GPUs ${GPU_IDS_DISPLAY}."
    )}]},
    config={"configurable": {"thread_id": "${RUN_ID}-${COLD_START}"}},
)

os.makedirs("${RESULTS_DIR}", exist_ok=True)
final_msg = (result.get("messages") or [None])[-1]
content = final_msg.content if final_msg else str(result)
with open("${RESULTS_DIR}/agent_output.txt", "w") as f:
    f.write(content)
print(content, flush=True)
PYEOF

log_info "Starting agent: CPUs=${CPU_START}-${CPU_END}, GPUs=${GPU_IDS_DISPLAY}, log=${AGENT_LOG}"
$PYTHON "${AGENT_PY}" > "${AGENT_LOG}" 2>&1
EXIT_CODE=$?
rm -f "${AGENT_PY}"

if [[ $EXIT_CODE -ne 0 ]]; then
    log_warn "Agent exited with status ${EXIT_CODE}; inspect ${AGENT_LOG}"
fi

# ══════════════════════════════════════════════════════════════════════
log_head "===== Step 4/4: Final evaluation ====="
# ══════════════════════════════════════════════════════════════════════

PREDICT_PY="${WORKSPACE}/predict.py"

if [[ ! -f "${PREDICT_PY}" ]]; then
    log_warn "predict.py was not found; skipping final evaluation"
else
    SCORE_PY="$(mktemp /tmp/gagc_score_XXXXXX.py)"
    cat > "${SCORE_PY}" << PYEOF
import json, sys, os
sys.path.insert(0, "${PROJ_ROOT}")
from gagc.benchmarks.amazon_reviews.harness import evaluate
from gagc.benchmarks.amazon_reviews.protocol import ProtocolViolation

try:
    result = evaluate(
        predict_script = "${PREDICT_PY}",
        data_dir       = "${TRAINVAL_DIR}",
        test_dir       = "${TEST_DIR}",
        mode           = "test",
        max_eval_users = 10000,
        verbose        = True,
    )
    out = {
        "primary_metric": result.primary_metric,
        "aggregate":      result.aggregate,
        "per_dataset": {
            ds: {"metrics": r.metrics, "n_users": r.n_users}
            for ds, r in result.per_dataset.items()
        },
    }
    score_path = os.path.join("${RESULTS_DIR}", "score.json")
    with open(score_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"HR@10 = {result.primary_metric:.4f}")
    print(f"Results saved to: {score_path}")
except (ProtocolViolation, Exception) as e:
    print(f"Evaluation failed: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

    $PYTHON "${SCORE_PY}" 2>&1 | tee "${LOGS_DIR}/score.log" || true
    rm -f "${SCORE_PY}"
fi

echo ""
log_ok "════════════════════════════════════════"
log_ok "  Run ID  : ${RUN_ID}"
log_ok "  Logs    : ${LOGS_DIR}/"
log_ok "  Results : ${RESULTS_DIR}/"
log_ok "════════════════════════════════════════"

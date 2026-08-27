#!/usr/bin/env bash
# =============================================================================
# gr.sh — KuaiRec watch-time GR agent
#
# Usage:
#   bash gr.sh [options]
#
# Options:
#   --train-data  Path to the training array (.npy), required
#   --test-data   Path to the evaluation array (.npy), required
#   --gpus        Comma-separated GPU IDs, required when GPUs are available
#   --budget      Total compute-time budget in seconds (default: 43200)
#   --trial-secs  Estimated cost per trial in seconds (default: 6400)
#   --cold-start  Cold-start template: gr | d2q | ks_d2q | tpm (default: gr)
#   --rho         Basin-jump posterior transfer coefficient (default: 0.25)
#   --cpus        Available CPU cores (default: 64)
#   --run-id      Experiment identifier (default: gr-YYYYMMDD-HHMMSS)
#   --help        Show this help text
#
# Example:
#   bash gr.sh \
#       --train-data ./input/train_data.npy \
#       --test-data  ./input/test_data.npy  \
#       --gpus 0,1,2,3 \
#       --budget 43200
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

# ── LangSmith ────────────────────────────────────────────────────────
export LANGSMITH_ENDPOINT="${LANGSMITH_ENDPOINT:-https://api.smith.langchain.com}"
export LANGSMITH_API_KEY="${LANGSMITH_API_KEY:-}"
LANGSMITH_PROJECT_PREFIX="${LANGSMITH_PROJECT_PREFIX:-${LANGSMITH_PROJECT:-agent}}"
export LANGCHAIN_TRACING_V2="${LANGCHAIN_TRACING_V2:-true}"

# Default parameters.
TRAIN_DATA=""
TEST_DATA=""
GPU_IDS_ARG=""
BUDGET=43200
TRIAL_SECS=6400
COLD_START="gr"
BASIN_TRANSFER_RHO=0.25
CPUS=64
RUN_ID=""

# Parse arguments.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-data) TRAIN_DATA="$2";  shift 2 ;;
        --test-data)  TEST_DATA="$2";   shift 2 ;;
        --gpus)       GPU_IDS_ARG="$2"; shift 2 ;;
        --budget)     BUDGET="$2";      shift 2 ;;
        --rho)        BASIN_TRANSFER_RHO="$2"; shift 2 ;;
        --trial-secs) TRIAL_SECS="$2";  shift 2 ;;
        --cold-start) COLD_START="$2";  shift 2 ;;
        --cpus)       CPUS="$2";        shift 2 ;;
        --run-id)     RUN_ID="$2";      shift 2 ;;
        --help)
            sed -n '6,20p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) die "Unknown argument: $1 (run bash gr.sh --help for usage)" ;;
    esac
done

[[ -z "$RUN_ID" ]] && RUN_ID="gr-$(date +%Y%m%d-%H%M%S)"

LANGSMITH_PROJECT="${LANGSMITH_PROJECT_PREFIX}-${COLD_START}"
export LANGSMITH_PROJECT
export LANGCHAIN_PROJECT="${LANGSMITH_PROJECT}"

# Output paths.
LOGS_DIR="./logs/${RUN_ID}"
RESULTS_DIR="./results/${RUN_ID}"
WORKSPACE="./workspace/${RUN_ID}/${COLD_START}"
AGENT_LOG="${LOGS_DIR}/agent.log"

mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}" "${WORKSPACE}"

PROJ_ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -x "${PROJ_ROOT}/.venv/bin/python3" ]]; then
    PYTHON="${PROJ_ROOT}/.venv/bin/python3"
else
    PYTHON=$(command -v python3 || command -v python || die "python3 was not found")
fi

# ══════════════════════════════════════════════════════════════════════
log_head "===== Step 1/3: Environment check ====="
# ══════════════════════════════════════════════════════════════════════

log_info "Python: $($PYTHON --version)"
log_info "Project root: ${PROJ_ROOT}"

# Check required inputs.
[[ -z "$TRAIN_DATA" ]] && die "--train-data is required, for example: --train-data ./input/train_data.npy"
[[ -z "$TEST_DATA"  ]] && die "--test-data is required, for example: --test-data ./input/test_data.npy"

# Convert inputs to absolute paths.
TRAIN_DATA="$(cd "$(dirname "$TRAIN_DATA")" && pwd)/$(basename "$TRAIN_DATA")"
TEST_DATA="$(cd "$(dirname "$TEST_DATA")" && pwd)/$(basename "$TEST_DATA")"

if [[ ! -f "$TRAIN_DATA" ]] || [[ ! -f "$TEST_DATA" ]]; then
    log_error "The data files do not exist. Prepare local KuaiRec data first:"
    log_error ""
    log_error "  python ${PROJ_ROOT}/prepare_gr_data.py --raw-dir /path/to/local/kuairec --output-dir $(dirname "${TRAIN_DATA}")"
    exit 1
fi

# Install the legacy-compatible package if needed.
if ! $PYTHON -c "import gagc" 2>/dev/null; then
    log_info "The gagc compatibility package is not installed; running uv pip install -e ."
    uv pip install -e "${PROJ_ROOT}" -q || die "uv pip install -e . failed"
fi
$PYTHON -c "import gagc" 2>/dev/null || die "The gagc compatibility package could not be imported"
$PYTHON -c "import torch" 2>/dev/null || die "PyTorch is not installed; run uv pip install torch"
$PYTHON -c "import langsmith" 2>/dev/null || { uv pip install langsmith -q; }
$PYTHON -c "import langchain_openai" 2>/dev/null || { uv pip install langchain-openai -q; }

# Detect GPUs.
GPU_COUNT=$( (nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true) | wc -l | tr -d ' ')
if [[ "$GPU_COUNT" -eq 0 ]]; then
    log_warn "No GPU was detected; running in CPU mode"
    GPU_IDS_PY="[]"
    NUM_GPUS=0
    GPU_IDS_DISPLAY="none"
else
    if [[ -z "$GPU_IDS_ARG" ]]; then
        die "--gpus is required, for example: --gpus '0,1,2,3'"
    fi
    GPU_IDS_PY="[$(echo "$GPU_IDS_ARG" | sed 's/,/, /g')]"
    NUM_GPUS=$(echo "$GPU_IDS_ARG" | tr ',' '\n' | wc -l | tr -d ' ')
    GPU_IDS_DISPLAY="$GPU_IDS_ARG"
    log_ok "Detected ${GPU_COUNT} GPUs; using ${GPU_IDS_DISPLAY} (${NUM_GPUS} parallel slots)"
fi

log_info "Run ID      : ${RUN_ID}"
log_info "Training data: ${TRAIN_DATA}"
log_info "Test data    : ${TEST_DATA}"
log_info "GPU       : ${GPU_IDS_DISPLAY}  |  CPUs: ${CPUS}  |  Budget: ${BUDGET}s  |  TrialSecs: ${TRIAL_SECS}s"
log_info "Cold start  : ${COLD_START}"
log_info "LangSmith : ${LANGSMITH_PROJECT}"
log_info "Workspace   : ${WORKSPACE}"
log_ok "Environment check passed"

# ══════════════════════════════════════════════════════════════════════
log_head "===== Step 2/3: Start GR agent ====="
# ══════════════════════════════════════════════════════════════════════

AGENT_PY="$(mktemp /tmp/gagc_gr_agent_XXXXXX.py)"

cat > "${AGENT_PY}" << PYEOF
import os, sys
sys.path.insert(0, "${PROJ_ROOT}")

os.environ["LANGSMITH_ENDPOINT"]   = "${LANGSMITH_ENDPOINT}"
os.environ["LANGSMITH_API_KEY"]    = "${LANGSMITH_API_KEY}"
os.environ["LANGSMITH_PROJECT"]    = "${LANGSMITH_PROJECT}"
os.environ["LANGCHAIN_PROJECT"]    = "${LANGSMITH_PROJECT}"
os.environ["LANGCHAIN_TRACING_V2"] = "${LANGCHAIN_TRACING_V2}"
os.environ["LANGCHAIN_RUN_NAME"]   = "${RUN_ID}/${COLD_START}"
os.environ["GAGC_TRIAL_SECS"]      = "${TRIAL_SECS}"
os.environ["GAGC_BASIN_TRANSFER_RHO"] = "${BASIN_TRANSFER_RHO}"

# Optional CPU affinity on Linux.
try:
    os.sched_setaffinity(0, set(range(0, ${CPUS})))
except (AttributeError, OSError):
    pass

from gagc.agent import create_gr_agent

agent = create_gr_agent(
    llm_provider       = "volcengine",
    model_id           = "glm-5-2-260617",
    train_data         = "${TRAIN_DATA}",
    test_data          = "${TEST_DATA}",
    cold_start         = "${COLD_START}",
    workspace_root     = "${WORKSPACE}",
    logs_root          = "${LOGS_DIR}",
    global_budget_secs = ${BUDGET},
    num_gpus           = ${NUM_GPUS} if ${NUM_GPUS} > 0 else 1,
    num_cpus           = ${CPUS},
    gpu_ids            = ${GPU_IDS_PY},
    basin_transfer_rho = ${BASIN_TRANSFER_RHO},
)

# UUID v7 thread_id (LangSmith's recommended format -- sorts by creation time in
# thread list views). Persisted under logs_root so re-running with the same
# --run-id resumes the same LangGraph thread instead of starting a fresh one.
from langsmith.uuid import uuid7
_thread_id_path = os.path.join("${LOGS_DIR}", "thread_id.txt")
if os.path.isfile(_thread_id_path):
    with open(_thread_id_path) as f:
        thread_id = f.read().strip()
else:
    thread_id = str(uuid7())
    with open(_thread_id_path, "w") as f:
        f.write(thread_id)

result = agent.invoke(
    {"messages": [{"role": "user", "content": (
        "Optimize the KuaiRec ${COLD_START} cold-start model to maximize watch-time xAUC "
        "while reducing watch-time MAE in seconds.\n"
        "Workspace: ${WORKSPACE}\n\n"
        "## Workflow\n"
        "Step 1 is mandatory: run the unmodified cold-start template once with execute_trial, "
        "record baseline WT xAUC and WT MAE, and monitor WR xAUC and WR MAE.\n"
        "From Step 2 onward, use propose_action_group followed by execute_trial_group to explore "
        "four optimization hypotheses in parallel and improve WT xAUC.\n"
        "Execute all four hypotheses in each iteration to use all available GPUs."
    )}]},
    config={
        "configurable": {"thread_id": thread_id},
        "run_name": "${RUN_ID}-${COLD_START}",
        "metadata": {"run_id": "${RUN_ID}", "cold_start": "${COLD_START}"},
    },
)

os.makedirs("${RESULTS_DIR}", exist_ok=True)
final_msg = (result.get("messages") or [None])[-1]
content = final_msg.content if final_msg else str(result)
with open("${RESULTS_DIR}/agent_output.txt", "w") as f:
    f.write(content)
print(content, flush=True)
PYEOF

log_info "Starting GR agent: GPUs=${GPU_IDS_DISPLAY}, CPUs=0-$((CPUS - 1)), log=${AGENT_LOG}"

$PYTHON "${AGENT_PY}" > "${AGENT_LOG}" 2>&1
EXIT_CODE=$?
rm -f "${AGENT_PY}"

if [[ $EXIT_CODE -ne 0 ]]; then
    log_warn "Agent exited with status ${EXIT_CODE}; inspect ${AGENT_LOG}"
else
    log_ok "Agent completed"
fi

# ══════════════════════════════════════════════════════════════════════
log_head "===== Step 3/3: Summarize results ====="
# ══════════════════════════════════════════════════════════════════════

SUMMARY_PY="$(mktemp /tmp/gagc_gr_summary_XXXXXX.py)"
cat > "${SUMMARY_PY}" << PYEOF
import json, os, glob, sys

logs_dir    = "${LOGS_DIR}"
results_dir = "${RESULTS_DIR}"

# Find the best xAUC and corresponding MAE across iteration logs.
iter_files = sorted(glob.glob(os.path.join(logs_dir, "iteration_*.json")))
best_xauc  = 0.0
best_mae   = float("inf")
best_iter  = -1

for fpath in iter_files:
    try:
        with open(fpath) as f:
            data = json.load(f)
        xauc = data.get("best_xauc") or data.get("best_test_score") or 0.0
        mae  = data.get("best_mae")  or data.get("test_metrics", {}).get("MAE") or float("inf")
        idx  = int(os.path.basename(fpath).replace("iteration_", "").replace(".json", ""))
        if xauc > best_xauc:
            best_xauc = xauc
            best_mae  = mae
            best_iter = idx
    except Exception:
        continue

summary = {
    "run_id":     "${RUN_ID}",
    "best_xauc":  best_xauc,
    "best_mae":   best_mae,
    "best_iter":  best_iter,
    "num_iters":  len(iter_files),
    "train_data": "${TRAIN_DATA}",
    "test_data":  "${TEST_DATA}",
    "cold_start": "${COLD_START}",
}

summary_path = os.path.join(results_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n{'='*50}")
if best_iter >= 0:
    print(f"  Best result (iteration {best_iter})")
    print(f"  xAUC = {best_xauc:.4f}")
    print(f"  MAE  = {best_mae:.4f}")
else:
    print("  No iteration logs were found; inspect the agent log")
print(f"{'='*50}")
print(f"\nResults saved to: {summary_path}")
PYEOF

$PYTHON "${SUMMARY_PY}" 2>&1 | tee "${LOGS_DIR}/summary.log" || true
rm -f "${SUMMARY_PY}"

echo ""
log_ok "════════════════════════════════════════"
log_ok "  Run ID    : ${RUN_ID}"
log_ok "  Logs      : ${LOGS_DIR}/"
log_ok "  Results   : ${RESULTS_DIR}/"
log_ok "  Agent log : ${AGENT_LOG}"
log_ok "  LangSmith : ${LANGSMITH_ENDPOINT}/projects/${LANGSMITH_PROJECT}"
log_ok "════════════════════════════════════════"

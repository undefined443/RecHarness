#!/usr/bin/env bash
# =============================================================================
# diversity.sh — diversity_v3 (DPP multi-window search diversity) agent
#
# Usage:
#   bash diversity.sh [options]
#
# Options:
#   --sample-path  Path to the request-log dataset (Parquet or ORC, file or
#                   directory), required
#   --vec-path     Path to the matching goods-vector dataset (Parquet or ORC),
#                   required
#   --budget       Total compute-time budget in seconds (default: 43200)
#   --trial-secs   Estimated cost per trial in seconds (default: 300)
#   --cold-start   Cold-start template: diversity_dpp (default: diversity_dpp)
#   --cpus         Available CPU cores (default: 8)
#   --run-id       Experiment identifier (default: diversity-YYYYMMDD-HHMMSS)
#   --help         Show this help text
#
# This benchmark is CPU-only (no GPU concept); trial parallelism is scheduled
# by --cpus, not --gpus.
#
# Example:
#   bash diversity.sh \
#       --sample-path ./input/diversity_v3/sample_500.parquet \
#       --vec-path    ./input/diversity_v3/vec_filtered.parquet \
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
SAMPLE_PATH=""
VEC_PATH=""
BUDGET=43200
TRIAL_SECS=300
COLD_START="diversity_dpp"
CPUS=8
RUN_ID=""

# Parse arguments.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample-path) SAMPLE_PATH="$2"; shift 2 ;;
        --vec-path)    VEC_PATH="$2";    shift 2 ;;
        --budget)      BUDGET="$2";      shift 2 ;;
        --trial-secs)  TRIAL_SECS="$2";  shift 2 ;;
        --cold-start)  COLD_START="$2";  shift 2 ;;
        --cpus)        CPUS="$2";        shift 2 ;;
        --run-id)      RUN_ID="$2";      shift 2 ;;
        --help)
            sed -n '6,20p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) die "Unknown argument: $1 (run bash diversity.sh --help for usage)" ;;
    esac
done

[[ -z "$RUN_ID" ]] && RUN_ID="diversity-$(date +%Y%m%d-%H%M%S)"

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

[[ -n "${OPENAI_API_KEY:-}" ]] \
    || die "OPENAI_API_KEY is not set. Export it before running this script."

# Check required inputs.
[[ -z "$SAMPLE_PATH" ]] && die "--sample-path is required, for example: --sample-path ./input/diversity_v3/sample_500.parquet"
[[ -z "$VEC_PATH"    ]] && die "--vec-path is required, for example: --vec-path ./input/diversity_v3/vec_filtered.parquet"

# Convert inputs to absolute paths (may be a file or a directory of ORC parts).
_abs_path() {
    if [[ -d "$1" ]]; then
        (cd "$1" && pwd)
    else
        echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    fi
}
SAMPLE_PATH="$(_abs_path "$SAMPLE_PATH")"
VEC_PATH="$(_abs_path "$VEC_PATH")"

if [[ ! -e "$SAMPLE_PATH" ]] || [[ ! -e "$VEC_PATH" ]]; then
    log_error "Missing dataset path(s):"
    log_error "  sample: ${SAMPLE_PATH}"
    log_error "  vec:    ${VEC_PATH}"
    log_error "See gagc/benchmarks/diversity_v3/vendor's DATA_DESCRIPTION.md for how to"
    log_error "build the local dev sample from the raw request-log / vector datasets."
    exit 1
fi

# Install the legacy-compatible package if needed.
if ! $PYTHON -c "import gagc" 2>/dev/null; then
    log_info "The gagc compatibility package is not installed; running uv pip install -e ."
    uv pip install -e "${PROJ_ROOT}" -q || die "uv pip install -e . failed"
fi
$PYTHON -c "import gagc" 2>/dev/null || die "The gagc compatibility package could not be imported"
$PYTHON -c "import pyarrow" 2>/dev/null || die "pyarrow is not installed; run uv sync --extra diversity"
$PYTHON -c "import yaml" 2>/dev/null || die "pyyaml is not installed; run uv sync --extra diversity"
$PYTHON -c "import langsmith" 2>/dev/null || { uv pip install langsmith -q; }
$PYTHON -c "import langchain_openai" 2>/dev/null || { uv pip install langchain-openai -q; }

log_info "Run ID        : ${RUN_ID}"
log_info "Sample data   : ${SAMPLE_PATH}"
log_info "Vector data   : ${VEC_PATH}"
log_info "CPUs: ${CPUS}  |  Budget: ${BUDGET}s  |  TrialSecs: ${TRIAL_SECS}s"
log_info "Cold start  : ${COLD_START}"
log_info "LangSmith : ${LANGSMITH_PROJECT}"
log_info "Workspace   : ${WORKSPACE}"
log_ok "Environment check passed"

# ══════════════════════════════════════════════════════════════════════
log_head "===== Step 2/3: Start diversity_v3 agent ====="
# ══════════════════════════════════════════════════════════════════════

AGENT_PY="$(mktemp /tmp/gagc_diversity_agent_XXXXXX.py)"

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

# Optional CPU affinity on Linux.
try:
    os.sched_setaffinity(0, set(range(0, ${CPUS})))
except (AttributeError, OSError):
    pass

from gagc.agent import create_diversity_agent

agent = create_diversity_agent(
    llm_provider  = "openai",
    model_id      = "glm_52_fp8",
    sample_path   = "${SAMPLE_PATH}",
    vec_path      = "${VEC_PATH}",
    cold_start    = "${COLD_START}",
    workspace_root = "${WORKSPACE}",
    logs_root      = "${LOGS_DIR}",
    global_budget_secs = ${BUDGET},
    num_gpus  = 1,
    num_cpus  = ${CPUS},
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
        "Maximize combined_pass_rate_mean on diversity_v3 (cold-start template: ${COLD_START}).\n"
        "Workspace: ${WORKSPACE}\n\n"
        "## Workflow\n"
        "Step 1 is mandatory: run the unmodified cold-start config.yaml once with execute_trial, "
        "record the baseline standard 1-4 metrics.\n"
        "From Step 2 onward, use propose_action_group followed by execute_trial_group to explore "
        "up to 4 DPP hyperparameter mutations in parallel and raise combined_pass_rate_mean without "
        "regressing Cat3Diversity or RankValue. Read _contingency_table and _decide_keep_reason "
        "from val_metrics every round -- VecSim and RankValue pull in opposite directions, so use "
        "the wasted/rv_only/bottleneck diagnostics to pick the next arm, not just the raw score delta.\n"
        "When the search budget (global_budget in /thompson_state/state.json) is exhausted, call "
        "evaluate_final_incumbent(script_path) exactly once on the frozen incumbent config.yaml, "
        "then report the returned standard 1-4 breakdown. This final evaluation is for reporting "
        "only -- never feed test results back into search."
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

log_info "Starting diversity_v3 agent: CPUs=0-$((CPUS - 1)), log=${AGENT_LOG}"

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

SUMMARY_PY="$(mktemp /tmp/gagc_diversity_summary_XXXXXX.py)"
cat > "${SUMMARY_PY}" << PYEOF
import json, os, glob

logs_dir    = "${LOGS_DIR}"
results_dir = "${RESULTS_DIR}"

# Best val_score seen during search, from the score_history recorded on each iteration.
iter_files = sorted(glob.glob(os.path.join(logs_dir, "iteration_*.json")))
best_val_score = 0.0
best_iter = -1
for fpath in iter_files:
    try:
        with open(fpath) as f:
            data = json.load(f)
        history = data.get("state_update", {}).get("score_history_after") or []
        idx = int(os.path.basename(fpath).replace("iteration_", "").replace(".json", ""))
        for score in history:
            if score is not None and score > best_val_score:
                best_val_score = score
                best_iter = idx
    except Exception:
        continue

# Held-out final metrics from evaluate_final_incumbent, if the agent called it.
final_eval_path = os.path.join(logs_dir, "final_eval", "latest.json")
final_metrics = None
if os.path.isfile(final_eval_path):
    try:
        with open(final_eval_path) as f:
            final_eval = json.load(f)
        final_metrics = (final_eval.get("test_metrics") or {}).get("DiversityV3")
    except Exception:
        pass

summary = {
    "run_id":           "${RUN_ID}",
    "best_val_score":   best_val_score,
    "best_iter":        best_iter,
    "final_metrics":    final_metrics,
    "num_iters":        len(iter_files),
    "sample_path":      "${SAMPLE_PATH}",
    "vec_path":         "${VEC_PATH}",
    "cold_start":       "${COLD_START}",
}

summary_path = os.path.join(results_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n{'='*50}")
if best_iter >= 0:
    print(f"  Best result (iteration {best_iter})")
    print(f"  combined_pass_rate = {best_val_score:.4f}")
else:
    print("  No iteration logs were found; inspect the agent log")
if final_metrics is not None:
    print(f"  final combined_pass_rate_mean = {final_metrics.get('combined_pass_rate_mean', 'n/a')}")
else:
    print("  No final_eval/latest.json found -- evaluate_final_incumbent was not called")
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

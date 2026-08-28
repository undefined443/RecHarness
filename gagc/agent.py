from __future__ import annotations

"""RecHarness main orchestrator agent.

Usage (company OpenAI-compatible gateway, GLM-5.2 — default):
    from gagc.agent import create_gagc_agent

    agent = create_gagc_agent(
        data_dir        = "./input/trainval",
        cold_start      = "sasrec",   # popular | sasrec | gru4rec | bert4rec
        workspace_root  = "./workspace",
        global_budget_secs = 43200.0,
        num_gpus=8, num_cpus=128,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Optimize the recommender to maximize HR@10."}]},
        config={"configurable": {"thread_id": "run-001"}},
    )

Usage (Claude, override):
    agent = create_gagc_agent(
        llm_provider="anthropic",
        model_id="claude-sonnet-4-6",
        ...
    )

Usage (VolcEngine Ark directly, override):
    agent = create_gagc_agent(
        llm_provider="volcengine",
        model_id="glm-5-2-260617",
        ...
    )
"""

import json
import os
import shutil
import sqlite3
import sys
from typing import Any

import gagc.tools as _tools_module
from gagc.prompts import (
    DIVERSITY_ORCHESTRATOR_PROMPT,
    GAGC_ORCHESTRATOR_PROMPT,
    GR_ORCHESTRATOR_PROMPT,
    SPOOKY_ORCHESTRATOR_PROMPT,
)
from gagc.templates import get_template_dir
from gagc.tools import (
    ServerConfig,
    _active_arms,
    default_experiment_skill_for_policy,
    evaluate_final_incumbent,
    execute_trial,
    execute_trial_group,
    promote_winner,
    propose_action_group,
    update_thompson_state,
)

# OpenAI-compatible: https://ark.cn-beijing.volces.com/api/v3
# Using OpenAI-compatible since langchain_openai is more reliable for tool-use.
_VOLCENGINE_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
_VOLCENGINE_MODEL    = "glm-5-2-260617"

# Default provider: the company's own OpenAI-compatible gateway. Configured via
# the standard OPENAI_API_KEY / OPENAI_BASE_URL env vars (langchain_openai.ChatOpenAI
# reads these natively), not hardcoded here.
_OPENAI_GATEWAY_MODEL = "glm_52_fp8"


def _resolve_volcengine_api_key(api_key: str | None) -> str:
    """Resolve the VolcEngine API key from the argument or environment.

    The key is never hardcoded in source. Provide it via the ``api_key``
    argument or the ``VOLCENGINE_API_KEY`` / ``ARK_API_KEY`` environment
    variable (see ``.env.example``).
    """
    key = api_key or os.environ.get("VOLCENGINE_API_KEY") or os.environ.get("ARK_API_KEY")
    if not key:
        raise RuntimeError(
            "VolcEngine API key not configured. Set VOLCENGINE_API_KEY (see "
            ".env.example) or pass api_key=... to create_gagc_agent."
        )
    return key


TEXTUAL_GRADIENT_ROUTING_PROMPT = """

# Textual-gradient routing ablation

Runtime `routing_policy` is `textual_gradient`, so Thompson Sampling is disabled for arm routing.
Before calling `propose_action_group`, choose the next arms yourself from `/thompson_state/state.json`
using `skill_notes`, `experiment_skill`, `recent_text_gradients`, `failure_memory`,
`rejected_dims_buffer`, score history, and diagnostics. Pass them as `textual_selected_arms_json`,
for example:
`{ "arms": [{"arm": "tune_lr", "delta": -1, "reason": "recent loss oscillation suggests lower LR"}, {"arm": "tune_dropout", "delta": 1}] }`.
Do not use α/β values to rank arms in this mode; they are retained only for logging compatibility.
"""


def _build_model(llm_provider: str, model_id: str, api_key: str | None, base_url: str | None):
    """Construct the LangChain chat model for the given provider."""
    # GLM-5.2 (the default model, on both the OpenAI-compatible gateway and VolcEngine)
    # is a reasoning model whose reasoning_content can consume most of a small max_tokens
    # budget, truncating the actual tool-call/answer content to empty and silently ending
    # the agent loop mid-round. Verified via a local smoke test: 4096 reproduced this
    # after 1-3 rounds; 16000 did not.
    max_tokens = int(os.environ.get("GAGC_LLM_MAX_TOKENS", "16000"))
    if llm_provider == "openai":
        # Company's own OpenAI-compatible gateway. api_key/base_url fall back to the
        # standard OPENAI_API_KEY / OPENAI_BASE_URL env vars (ChatOpenAI's own defaults)
        # when not passed explicitly -- nothing gateway-specific is hardcoded here.
        # Thinking is disabled per team experience: no measurable capability loss on this
        # gateway's GLM-5.2 deployment, and it avoids burning max_tokens on reasoning_content.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_id,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    elif llm_provider == "volcengine":
        # VolcEngine Ark is OpenAI-compatible — use ChatOpenAI, not ChatAnthropic.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_id,
            api_key=_resolve_volcengine_api_key(api_key),
            base_url=base_url or os.environ.get("VOLCENGINE_BASE_URL") or _VOLCENGINE_BASE_URL,
            max_tokens=max_tokens,
        )
    elif llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_id,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            max_tokens=max_tokens,
        )
    else:
        # Generic: pass a "provider:model" string directly to deepagents
        return f"{llm_provider}:{model_id}"


def _enable_claude_backend_by_default() -> None:
    raw = os.environ.get("GAGC_ENABLE_CLAUDE_BACKEND")
    if raw is None:
        os.environ["GAGC_ENABLE_CLAUDE_BACKEND"] = "1"
        return
    if raw.strip().lower() in {"0", "false", "no", "off", "none"}:
        return
    os.environ["GAGC_ENABLE_CLAUDE_BACKEND"] = raw


def _init_workspace(workspace_root: str, cold_start: str) -> None:
    """Copy cold-start template files into workspace_root if best.py is absent."""
    best_py = os.path.join(workspace_root, "best.py")
    if os.path.exists(best_py):
        return  # already initialised — don't overwrite

    template_dir = get_template_dir(cold_start)
    os.makedirs(workspace_root, exist_ok=True)
    for fname in os.listdir(template_dir):
        src = os.path.join(template_dir, fname)
        dst = os.path.join(workspace_root, fname)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    # The entry point RecHarness evolves is always best.py
    train_src = os.path.join(workspace_root, "train.py")
    if os.path.exists(train_src) and not os.path.exists(best_py):
        shutil.copy2(train_src, best_py)

    # Initialize git repo so agent can use git diff to generate patches
    import subprocess
    try:
        subprocess.run(
            ["git", "init"],
            cwd=workspace_root,
            capture_output=True,
            check=True,
            timeout=5,
        )
        subprocess.run(
            ["git", "add", "."],
            cwd=workspace_root,
            capture_output=True,
            check=True,
            timeout=5,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial cold-start template"],
            cwd=workspace_root,
            capture_output=True,
            check=True,
            timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # Git init failed — agent will have to generate diffs another way
        pass


class _PersistentStoreProxy:
    """Mirror the Thompson state store to disk without changing StoreBackend use."""

    def __init__(self, inner: Any, path: str):
        self._inner = inner
        self._path = os.path.abspath(path)
        self._load()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                value = json.load(f)
            if not isinstance(value, dict):
                return
            self._inner.put(("gagc", "thompson"), "/state.json", value)
        except Exception as exc:  # pragma: no cover - best-effort resume aid
            print(f"[gagc] warning: failed to load Thompson state {self._path}: {exc}", file=sys.stderr)

    def _persist(self, namespace: Any, key: Any, value: Any) -> None:
        if tuple(namespace) != ("gagc", "thompson") or key != "/state.json":
            return
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            tmp_path = f"{self._path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._path)
        except Exception as exc:  # pragma: no cover - best-effort resume aid
            print(f"[gagc] warning: failed to persist Thompson state {self._path}: {exc}", file=sys.stderr)

    def put(self, namespace: Any, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        result = self._inner.put(namespace, key, value, *args, **kwargs)
        self._persist(namespace, key, value)
        return result

    def get(self, namespace: Any, key: str, *args: Any, **kwargs: Any) -> Any:
        return self._inner.get(namespace, key, *args, **kwargs)


def _checkpoint_path(logs_root: str, checkpoint_path: str | None) -> str | None:
    raw = checkpoint_path or os.environ.get("GAGC_LANGGRAPH_CHECKPOINT_PATH") or os.environ.get("GAGC_CHECKPOINT_PATH")
    if raw and raw.strip().lower() in {"", "0", "false", "none", "memory", "off"}:
        return None
    return os.path.abspath(raw or os.path.join(logs_root, "langgraph_checkpoints.sqlite"))


def _build_checkpointer(logs_root: str, checkpoint_path: str | None, enable_persistent_checkpoint: bool):
    from langgraph.checkpoint.memory import MemorySaver

    if not enable_persistent_checkpoint:
        return MemorySaver()

    path = _checkpoint_path(logs_root, checkpoint_path)
    if not path:
        return MemorySaver()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception as exc:
        print(
            "[gagc] warning: langgraph-checkpoint-sqlite is unavailable; "
            f"falling back to in-memory checkpoints ({exc})",
            file=sys.stderr,
        )
        return MemorySaver()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    if hasattr(saver, "setup"):
        saver.setup()
    return saver


def _build_store(logs_root: str, thompson_state_path: str | None, enable_persistent_checkpoint: bool):
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    if not enable_persistent_checkpoint:
        return store
    path = thompson_state_path or os.environ.get("GAGC_THOMPSON_STATE_PATH")
    if path and path.strip().lower() in {"", "0", "false", "none", "memory", "off"}:
        return store
    return _PersistentStoreProxy(store, path or os.path.join(logs_root, "thompson_state.json"))


def _seed_thompson_state(store: Any, default_state: Any) -> None:
    try:
        item = store.get(("gagc", "thompson"), "/state.json")
        if item and item.value.get("content"):
            return
    except Exception:
        pass
    import datetime
    store.put(("gagc", "thompson"), "/state.json", {
        "content": default_state.to_json(),
        "modified_at": datetime.datetime.now(datetime.UTC).isoformat(),
    })


def create_gagc_agent(
    # ── LLM ─────────────────────────────────────────────────────────
    llm_provider: str = "openai",
    model_id: str = _OPENAI_GATEWAY_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    # ── Task ────────────────────────────────────────────────────────
    data_dir: str = "./input/trainval",
    test_dir: str | None = None,
    cold_start: str = "sasrec",
    # ── Infrastructure ──────────────────────────────────────────────
    workspace_root: str = "./workspace",
    logs_root: str = "./logs",
    global_budget_secs: float = 43200.0,
    num_gpus: int = 8,
    num_cpus: int = 128,
    gpu_ids: list[int] | None = None,
    use_thompson_sampling: bool = True,
    basin_transfer_rho: float | None = None,
    enable_persistent_checkpoint: bool = True,
    checkpoint_path: str | None = None,
    thompson_state_path: str | None = None,
):
    """Build and return the RecHarness orchestrator as a compiled deep agent.

    Args:
        llm_provider: 'openai' (default, company gateway) | 'volcengine' | 'anthropic' | any
            deepagents provider string.
        model_id: Model name. Default 'glm_52_fp8' for the 'openai' gateway provider.
        api_key: API key. For 'openai', read from OPENAI_API_KEY env var if not passed.
            For 'volcengine', read from VOLCENGINE_API_KEY env var if not passed.
        base_url: Override base URL. For 'openai', read from OPENAI_BASE_URL env var if
            not passed. For 'volcengine', the Ark URL is bundled as default.
        data_dir: Directory with *_train.txt and *_valid.txt files.
        test_dir: Directory with *_test.txt files (defaults to data_dir).
        cold_start: Template name — 'popular' | 'sasrec' | 'gru4rec' | 'bert4rec'.
        workspace_root: Root for training scripts.
        logs_root: Directory for per-iteration JSON logs.
        global_budget_secs: Total GPU/compute-time budget in seconds (default 43,200 s).
        num_gpus: GPU devices available (default 8, A800 server).
        num_cpus: CPU cores available (default 128, A800 server).
        use_thompson_sampling: If False, run the textual-gradient ablation:
            the LLM chooses arms from global experiment memory and Thompson
            posterior samples are not used for routing.
        basin_transfer_rho: Optional one-time posterior transfer coefficient
            used when a successful basin jump creates a new local Thompson basin.
        enable_persistent_checkpoint: Persist LangGraph checkpoints and Thompson
            state under logs_root so the same thread_id can resume after restart.
        checkpoint_path: Optional SQLite checkpoint DB path. Defaults to
            logs_root/langgraph_checkpoints.sqlite. Set to "memory" to disable.
        thompson_state_path: Optional Thompson state JSON path. Defaults to
            logs_root/thompson_state.json. Set to "memory" to disable.
    """
    _enable_claude_backend_by_default()

    # --- hardware config ---
    _tools_module._SERVER_CONFIG = ServerConfig(
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        cpus_per_trial=num_cpus // num_gpus,
        gpu_ids=gpu_ids or [],
    )
    _tools_module._DATA_DIR = data_dir
    _tools_module._TEST_DIR = test_dir or data_dir
    _tools_module._LOGS_ROOT = os.path.abspath(logs_root)
    _tools_module._WORKSPACE_ROOT = os.path.abspath(workspace_root)
    _tools_module._ROUTING_POLICY = "thompson" if use_thompson_sampling else "textual_gradient"
    os.environ["GAGC_COLD_START"] = cold_start
    if basin_transfer_rho is not None:
        _tools_module.BASIN_TRANSFER_RHO = min(1.0, max(0.0, float(basin_transfer_rho)))

    os.makedirs(logs_root, exist_ok=True)

    # --- cold start ---
    _init_workspace(workspace_root, cold_start)

    # absolute paths
    abs_workspace = os.path.abspath(workspace_root)
    abs_data_dir  = os.path.abspath(data_dir)
    abs_test_dir  = os.path.abspath(test_dir or data_dir)
    abs_logs_root = os.path.abspath(logs_root)

    system_prompt = (
        GAGC_ORCHESTRATOR_PROMPT
        + f"\n\n# Runtime context\n"
        f"## Virtual paths (use these with read_file / write_file / ls)\n"
        f"- /workspace/best.py          — training script\n"
        f"- /workspace/predict.py       — inference script\n"
        f"- /thompson_state/state.json  — Thompson Sampling + SkillOpt memory state\n"
        f"- /logs/iteration_<N>.json    — per-iteration logs\n"
        f"\n## Real paths (use ONLY for trial-tool script_path and git diff)\n"
        f"- trial-tool script_path      : {abs_workspace}/best.py\n"
        f"- git diff workspace path     : {abs_workspace}\n"
        f"  (use: bash(\"cd {abs_workspace} && git diff best.py\") to generate code_diff)\n"
        f"\n## Other\n"
        f"- data_dir       : {abs_data_dir}\n"
        f"- test_dir       : {abs_test_dir}\n"
        f"- global_budget  : {global_budget_secs}s\n"
        f"- num_gpus       : {num_gpus}\n"
        f"- gpu_ids        : {gpu_ids or list(range(num_gpus))}\n"
        f"- cold_start     : {cold_start}\n"
        f"\nRULE: read_file/write_file/ls MUST use the virtual paths above (starting with\n"
        f"/workspace/, /thompson_state/, /logs/). NEVER pass a real absolute path like\n"
        f"{abs_workspace}/... to read_file — it will not be found.\n"
        f"The ONLY exceptions are trial-tool script_path and bash commands for git diff.\n"
    )
    if os.environ.get("GAGC_FULL_RANKING", "").strip().lower() in {"1", "true", "yes", "on"}:
        system_prompt += (
            "\n## Runtime evaluation override\n"
            "This run uses full-item ranking over all non-history items. "
            "Treat Recall@10 as the primary metric, and report Recall@5/10 and NDCG@5/10. "
            "In this leave-one-out setting, internal HR@K hits are exposed as Recall@K aliases.\n"
        )
    if not use_thompson_sampling:
        system_prompt += TEXTUAL_GRADIENT_ROUTING_PROMPT

    # Patch deepagents summarization to prevent empty message list reaching the LLM.
    #
    # Root cause: after summarization, _apply_event_to_messages may return only
    # [summary_msg] when cutoff_idx==len(messages). Then if _truncate_args further
    # reduces messages, or _find_safe_cutoff_point returns idx==len(messages) (when
    # the entire tail is ToolMessages with no matching AIMessage), the effective
    # message list passed to handler can be [] — the LLM API returns empty generations
    # → .generations[0][0] raises IndexError.
    #
    # Fix: patch _get_effective_messages in _DeepAgentsSummarizationMiddleware to
    # guarantee the result is never empty (fall back to the raw request messages).
    from deepagents.middleware.summarization import (
        _DeepAgentsSummarizationMiddleware as _DSM,
    )
    _original_get_effective = _DSM._get_effective_messages
    def _safe_get_effective_messages(self, request):  # type: ignore[misc]
        result = _original_get_effective(self, request)
        if not result:
            # Fallback: return raw messages; at minimum one message must exist.
            return list(request.messages) or [request.system_message] if request.system_message else list(request.messages)
        return result
    _DSM._get_effective_messages = _safe_get_effective_messages  # type: ignore[assignment]

    # Also patch _find_safe_cutoff_point so it never returns len(messages),
    # which would cause _partition_messages to produce an empty preserved slice.
    from langchain.agents.middleware.summarization import SummarizationMiddleware as _SM
    _original_cutoff = _SM._find_safe_cutoff_point
    @staticmethod  # type: ignore[misc]
    def _safe_cutoff_point(messages, cutoff_index):
        result = _original_cutoff(messages, cutoff_index)
        return min(result, max(0, len(messages) - 1))
    _SM._find_safe_cutoff_point = _safe_cutoff_point  # type: ignore[assignment]

    from deepagents import create_deep_agent
    from deepagents.backends import (
        CompositeBackend,
        FilesystemBackend,
        LocalShellBackend,
        StateBackend,
        StoreBackend,
    )
    model = _build_model(llm_provider, model_id, api_key, base_url)

    store = _build_store(abs_logs_root, thompson_state_path, enable_persistent_checkpoint)

    # Inject store into tools module so update_thompson_state can persist state
    # directly without requiring the LLM to call write_file.
    _tools_module._STORE = store

    # Pre-seed ThompsonState so read_file("/thompson_state/state.json") succeeds on iteration 1.
    from gagc.state import ArmState, ThompsonState
    default_state = ThompsonState(
        arms={arm: ArmState(name=arm) for arm in _active_arms()},
        global_budget=global_budget_secs,
        score_history=[],
        rejected_dims_buffer=[],
        skill_notes="Cold start: no prior knowledge.",
        pending_queue=[],
        recent_text_gradients=[],
        failure_memory=[],
        experiment_memory=[],
        experiment_skill=default_experiment_skill_for_policy(),
    )
    _seed_thompson_state(store, default_state)

    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/":    LocalShellBackend(root_dir=abs_workspace, virtual_mode=True),
            "/thompson_state/":  StoreBackend(namespace=lambda _rt: ("gagc", "thompson")),
            "/logs/":         FilesystemBackend(root_dir=abs_logs_root, virtual_mode=True),
        },
    )

    agent = create_deep_agent(
        model=model,
        tools=[
            propose_action_group,
            execute_trial_group,
            execute_trial,
            promote_winner,
            update_thompson_state,
            evaluate_final_incumbent,
        ],
        backend=backend,
        store=store,
        checkpointer=_build_checkpointer(abs_logs_root, checkpoint_path, enable_persistent_checkpoint),
        system_prompt=system_prompt,
    )

    return agent


def create_gr_agent(
    # ── LLM ─────────────────────────────────────────────────────────
    llm_provider: str = "openai",
    model_id: str = _OPENAI_GATEWAY_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    # ── Task ────────────────────────────────────────────────────────
    train_data: str = "./input/train_data.npy",
    val_data: str | None = None,
    test_data: str = "./input/test_data.npy",
    cold_start: str = "gr",
    # ── Infrastructure ──────────────────────────────────────────────
    workspace_root: str = "./workspace",
    logs_root: str = "./logs",
    global_budget_secs: float = 43200.0,
    num_gpus: int = 8,
    num_cpus: int = 128,
    gpu_ids: list[int] | None = None,
    use_thompson_sampling: bool = True,
    basin_transfer_rho: float | None = None,
    enable_persistent_checkpoint: bool = True,
    checkpoint_path: str | None = None,
    thompson_state_path: str | None = None,
):
    """Build and return the RecHarness GR orchestrator for KuaiRec watch-time prediction.

    Args:
        llm_provider: 'openai' (default, company gateway) | 'volcengine' | 'anthropic'.
        model_id: Model name.
        train_data: Path to train .npy file.
        val_data: Path to validation .npy file used during search. Defaults to
            GAGC_GR_VAL_DATA, then test_data for backward compatibility.
        test_data: Path to held-out test .npy file used only by final evaluation.
        cold_start: KuaiRec template name: 'gr' | 'd2q' | 'ks_d2q' | 'tpm'.
        workspace_root: Root for training scripts.
        logs_root: Directory for per-iteration JSON logs.
        global_budget_secs: Total GPU/compute-time budget.
        num_gpus: GPU devices available.
        num_cpus: CPU cores available.
        gpu_ids: Explicit GPU IDs to use (defaults to range(num_gpus)).
        use_thompson_sampling: If False, run the textual-gradient ablation:
            the LLM chooses arms from global experiment memory and Thompson
            posterior samples are not used for routing.
        basin_transfer_rho: Optional one-time posterior transfer coefficient
            used when a successful basin jump creates a new local Thompson basin.
        enable_persistent_checkpoint: Persist LangGraph checkpoints and Thompson
            state under logs_root so the same thread_id can resume after restart.
        checkpoint_path: Optional SQLite checkpoint DB path. Defaults to
            logs_root/langgraph_checkpoints.sqlite. Set to "memory" to disable.
        thompson_state_path: Optional Thompson state JSON path. Defaults to
            logs_root/thompson_state.json. Set to "memory" to disable.

    Usage:
        from gagc.agent import create_gr_agent
        agent = create_gr_agent(
            train_data="./input/train_data.npy",
            test_data="./input/test_data.npy",
            workspace_root="./workspace_gr",
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Optimize the GR model to maximize xAUC and minimize MAE."}]},
            config={"configurable": {"thread_id": "gr-run-001"}},
        )
    """
    # Switch benchmark mode to kuairec BEFORE init_workspace / pre-seed
    _tools_module._BENCHMARK_MODE = "kuairec"
    _tools_module._GR_TRAIN_DATA = os.path.abspath(train_data)
    _tools_module._GR_VAL_DATA = os.path.abspath(val_data or os.environ.get("GAGC_GR_VAL_DATA") or test_data)
    _tools_module._GR_TEST_DATA = os.path.abspath(test_data)
    _tools_module._ROUTING_POLICY = "thompson" if use_thompson_sampling else "textual_gradient"
    os.environ["GAGC_COLD_START"] = cold_start
    if basin_transfer_rho is not None:
        _tools_module.BASIN_TRANSFER_RHO = min(1.0, max(0.0, float(basin_transfer_rho)))

    _tools_module._SERVER_CONFIG = ServerConfig(
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        cpus_per_trial=num_cpus // max(num_gpus, 1),
        gpu_ids=gpu_ids or [],
    )
    _tools_module._LOGS_ROOT = os.path.abspath(logs_root)
    _tools_module._WORKSPACE_ROOT = os.path.abspath(workspace_root)

    os.makedirs(logs_root, exist_ok=True)
    _init_workspace(workspace_root, cold_start)

    abs_workspace = os.path.abspath(workspace_root)
    abs_logs_root = os.path.abspath(logs_root)

    system_prompt = (
        GR_ORCHESTRATOR_PROMPT
        + f"\n\n# Runtime context\n"
        f"## Virtual paths\n"
        f"- /workspace/best.py          — KuaiRec training script (RecHarness evolves this)\n"
        f"- /thompson_state/state.json  — Thompson Sampling + SkillOpt memory state\n"
        f"- /logs/iteration_<N>.json    — per-iteration logs\n"
        f"\n## Real paths\n"
        f"- trial-tool script_path      : {abs_workspace}/best.py\n"
        f"- git diff workspace path     : {abs_workspace}\n"
        f"  (use: bash(\"cd {abs_workspace} && git diff best.py\") to generate code_diff)\n"
        f"\n## Data\n"
        f"- GR_TRAIN_DATA : {_tools_module._GR_TRAIN_DATA}\n"
        f"- GR_VAL_DATA   : {_tools_module._GR_VAL_DATA} (used during search)\n"
        f"- GR_TEST_DATA  : {_tools_module._GR_TEST_DATA} (held out until evaluate_final_incumbent)\n"
        f"- cold_start    : {cold_start}\n"
        f"\n## Budget\n"
        f"- global_budget : {global_budget_secs}s\n"
        f"- num_gpus      : {num_gpus}\n"
        f"- gpu_ids       : {gpu_ids or list(range(num_gpus))}\n"
        f"\nRULE: read_file/write_file/ls MUST use virtual paths (/workspace/, /thompson_state/, /logs/).\n"
        f"NEVER pass a real absolute path to read_file. Use the real trial-tool script_path above for execute_trial, execute_trial_group, promote_winner, and evaluate_final_incumbent.\n"
    )
    if not use_thompson_sampling:
        system_prompt += TEXTUAL_GRADIENT_ROUTING_PROMPT

    from deepagents.middleware.summarization import (
        _DeepAgentsSummarizationMiddleware as _DSM,
    )
    _original_get_effective = _DSM._get_effective_messages
    def _safe_get_effective_messages(self, request):  # type: ignore[misc]
        result = _original_get_effective(self, request)
        if not result:
            return list(request.messages) or [request.system_message] if request.system_message else list(request.messages)
        return result
    _DSM._get_effective_messages = _safe_get_effective_messages  # type: ignore[assignment]

    from langchain.agents.middleware.summarization import SummarizationMiddleware as _SM
    _original_cutoff = _SM._find_safe_cutoff_point
    @staticmethod  # type: ignore[misc]
    def _safe_cutoff_point(messages, cutoff_index):
        result = _original_cutoff(messages, cutoff_index)
        return min(result, max(0, len(messages) - 1))
    _SM._find_safe_cutoff_point = _safe_cutoff_point  # type: ignore[assignment]

    from deepagents import create_deep_agent
    from deepagents.backends import (
        CompositeBackend,
        FilesystemBackend,
        LocalShellBackend,
        StateBackend,
        StoreBackend,
    )
    model = _build_model(llm_provider, model_id, api_key, base_url)
    store = _build_store(abs_logs_root, thompson_state_path, enable_persistent_checkpoint)
    _tools_module._STORE = store

    from gagc.state import ArmState, ThompsonState
    from gagc.tools import _active_arms
    default_state = ThompsonState(
        arms={arm: ArmState(name=arm) for arm in _active_arms()},
        global_budget=global_budget_secs,
        score_history=[],
        rejected_dims_buffer=[],
        skill_notes="Cold start: no prior knowledge.",
        pending_queue=[],
        experiment_skill=default_experiment_skill_for_policy(),
    )
    _seed_thompson_state(store, default_state)

    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/":    LocalShellBackend(root_dir=abs_workspace, virtual_mode=True),
            "/thompson_state/":  StoreBackend(namespace=lambda _rt: ("gagc", "thompson")),
            "/logs/":         FilesystemBackend(root_dir=abs_logs_root, virtual_mode=True),
        },
    )

    agent = create_deep_agent(
        model=model,
        tools=[
            propose_action_group,
            execute_trial_group,
            execute_trial,
            promote_winner,
            update_thompson_state,
            evaluate_final_incumbent,
        ],
        backend=backend,
        store=store,
        checkpointer=_build_checkpointer(abs_logs_root, checkpoint_path, enable_persistent_checkpoint),
        system_prompt=system_prompt,
    )

    return agent


def create_spooky_agent(
    # ── LLM ─────────────────────────────────────────────────────────
    llm_provider: str = "openai",
    model_id: str = _OPENAI_GATEWAY_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    # ── Task ────────────────────────────────────────────────────────
    train_data: str = "./input/spooky_author/train.csv",
    val_data: str | None = None,
    test_data: str = "./input/spooky_author/test.csv",
    private_test_data: str | None = None,
    cold_start: str = "spooky_mlp",
    # ── Infrastructure ──────────────────────────────────────────────
    workspace_root: str = "./workspace",
    logs_root: str = "./logs",
    global_budget_secs: float = 43200.0,
    num_gpus: int = 8,
    num_cpus: int = 128,
    gpu_ids: list[int] | None = None,
    use_thompson_sampling: bool = True,
    basin_transfer_rho: float | None = None,
    enable_persistent_checkpoint: bool = True,
    checkpoint_path: str | None = None,
    thompson_state_path: str | None = None,
):
    """Build and return the RecHarness orchestrator for MLE-Bench Lite
    spooky-author-identification (3-class text classification).

    Args:
        llm_provider: 'openai' (default, company gateway) | 'volcengine' | 'anthropic'.
        model_id: Model name.
        train_data: Path to train.csv (produced by prepare_spooky_data.py).
        val_data: Path to val.csv used during search. Defaults to
            GAGC_SPOOKY_VAL_DATA, then a val.csv sibling of train_data.
        test_data: Path to test.csv (unlabeled) used only by final evaluation.
        private_test_data: Path to private_test.csv (answer key) used only by
            final evaluation. Defaults to GAGC_SPOOKY_PRIVATE_TEST, then a
            private_test.csv sibling of train_data.
        cold_start: Template name: 'spooky_mlp' (TF-IDF + MLP baseline).
        workspace_root: Root for training scripts.
        logs_root: Directory for per-iteration JSON logs.
        global_budget_secs: Total compute-time budget.
        num_gpus: GPU devices available (not required for this benchmark; used
            only for trial-slot scheduling).
        num_cpus: CPU cores available.
        gpu_ids: Explicit GPU IDs to use (defaults to range(num_gpus)).
        use_thompson_sampling: If False, run the textual-gradient ablation.
        basin_transfer_rho: Optional one-time posterior transfer coefficient
            used when a successful basin jump creates a new local Thompson basin.
        enable_persistent_checkpoint: Persist LangGraph checkpoints and Thompson
            state under logs_root so the same thread_id can resume after restart.
        checkpoint_path: Optional SQLite checkpoint DB path. Defaults to
            logs_root/langgraph_checkpoints.sqlite. Set to "memory" to disable.
        thompson_state_path: Optional Thompson state JSON path. Defaults to
            logs_root/thompson_state.json. Set to "memory" to disable.

    Usage:
        from gagc.agent import create_spooky_agent
        agent = create_spooky_agent(
            train_data="./input/spooky_author/train.csv",
            test_data="./input/spooky_author/test.csv",
            workspace_root="./workspace_spooky",
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Minimize log loss on spooky-author-identification."}]},
            config={"configurable": {"thread_id": "spooky-run-001"}},
        )
    """
    # Switch benchmark mode to spooky_author BEFORE init_workspace / pre-seed
    _tools_module._BENCHMARK_MODE = "spooky_author"
    _tools_module._SPOOKY_TRAIN_DATA = os.path.abspath(train_data)
    default_val = os.path.join(os.path.dirname(os.path.abspath(train_data)), "val.csv")
    _tools_module._SPOOKY_VAL_DATA = os.path.abspath(
        val_data or os.environ.get("GAGC_SPOOKY_VAL_DATA") or default_val
    )
    _tools_module._SPOOKY_TEST_DATA = os.path.abspath(test_data)
    default_private_test = os.path.join(
        os.path.dirname(os.path.abspath(train_data)), "private_test.csv"
    )
    _tools_module._SPOOKY_PRIVATE_TEST = os.path.abspath(
        private_test_data or os.environ.get("GAGC_SPOOKY_PRIVATE_TEST") or default_private_test
    )
    _tools_module._ROUTING_POLICY = "thompson" if use_thompson_sampling else "textual_gradient"
    os.environ["GAGC_COLD_START"] = cold_start
    if basin_transfer_rho is not None:
        _tools_module.BASIN_TRANSFER_RHO = min(1.0, max(0.0, float(basin_transfer_rho)))

    _tools_module._SERVER_CONFIG = ServerConfig(
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        cpus_per_trial=num_cpus // max(num_gpus, 1),
        gpu_ids=gpu_ids or [],
    )
    _tools_module._LOGS_ROOT = os.path.abspath(logs_root)
    _tools_module._WORKSPACE_ROOT = os.path.abspath(workspace_root)

    os.makedirs(logs_root, exist_ok=True)
    _init_workspace(workspace_root, cold_start)

    abs_workspace = os.path.abspath(workspace_root)
    abs_logs_root = os.path.abspath(logs_root)

    system_prompt = (
        SPOOKY_ORCHESTRATOR_PROMPT
        + f"\n\n# Runtime context\n"
        f"## Virtual paths\n"
        f"- /workspace/best.py          — spooky-author-identification training script (RecHarness evolves this)\n"
        f"- /thompson_state/state.json  — Thompson Sampling + SkillOpt memory state\n"
        f"- /logs/iteration_<N>.json    — per-iteration logs\n"
        f"\n## Real paths\n"
        f"- trial-tool script_path      : {abs_workspace}/best.py\n"
        f"- git diff workspace path     : {abs_workspace}\n"
        f"  (use: bash(\"cd {abs_workspace} && git diff best.py\") to generate code_diff)\n"
        f"\n## Data\n"
        f"- SPOOKY_TRAIN_DATA  : {_tools_module._SPOOKY_TRAIN_DATA}\n"
        f"- SPOOKY_VAL_DATA    : {_tools_module._SPOOKY_VAL_DATA} (used during search)\n"
        f"- test data          : {_tools_module._SPOOKY_TEST_DATA} (held out until evaluate_final_incumbent)\n"
        f"- private answers    : {_tools_module._SPOOKY_PRIVATE_TEST} (held out until evaluate_final_incumbent)\n"
        f"- cold_start         : {cold_start}\n"
        f"\n## Budget\n"
        f"- global_budget : {global_budget_secs}s\n"
        f"- num_gpus      : {num_gpus} (not required for this CPU-friendly benchmark)\n"
        f"- gpu_ids       : {gpu_ids or list(range(num_gpus))}\n"
        f"\nRULE: read_file/write_file/ls MUST use virtual paths (/workspace/, /thompson_state/, /logs/).\n"
        f"NEVER pass a real absolute path to read_file. Use the real trial-tool script_path above for execute_trial, execute_trial_group, promote_winner, and evaluate_final_incumbent.\n"
    )
    if not use_thompson_sampling:
        system_prompt += TEXTUAL_GRADIENT_ROUTING_PROMPT

    from deepagents import create_deep_agent
    from deepagents.backends import (
        CompositeBackend,
        FilesystemBackend,
        LocalShellBackend,
        StateBackend,
        StoreBackend,
    )
    model = _build_model(llm_provider, model_id, api_key, base_url)
    store = _build_store(abs_logs_root, thompson_state_path, enable_persistent_checkpoint)
    _tools_module._STORE = store

    from gagc.state import ArmState, ThompsonState
    from gagc.tools import _active_arms
    default_state = ThompsonState(
        arms={arm: ArmState(name=arm) for arm in _active_arms()},
        global_budget=global_budget_secs,
        score_history=[],
        rejected_dims_buffer=[],
        skill_notes="Cold start: no prior knowledge.",
        pending_queue=[],
        experiment_skill=default_experiment_skill_for_policy(),
    )
    _seed_thompson_state(store, default_state)

    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/":    LocalShellBackend(root_dir=abs_workspace, virtual_mode=True),
            "/thompson_state/":  StoreBackend(namespace=lambda _rt: ("gagc", "thompson")),
            "/logs/":         FilesystemBackend(root_dir=abs_logs_root, virtual_mode=True),
        },
    )

    agent = create_deep_agent(
        model=model,
        tools=[
            propose_action_group,
            execute_trial_group,
            execute_trial,
            promote_winner,
            update_thompson_state,
            evaluate_final_incumbent,
        ],
        backend=backend,
        store=store,
        checkpointer=_build_checkpointer(abs_logs_root, checkpoint_path, enable_persistent_checkpoint),
        system_prompt=system_prompt,
    )

    return agent


def create_diversity_agent(
    # ── LLM ─────────────────────────────────────────────────────────
    llm_provider: str = "openai",
    model_id: str = _OPENAI_GATEWAY_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    # ── Task ────────────────────────────────────────────────────────
    sample_path: str = "./input/diversity_v3/sample_500.parquet",
    vec_path: str = "./input/diversity_v3/vec_filtered.parquet",
    cold_start: str = "diversity_dpp",
    # ── Infrastructure ──────────────────────────────────────────────
    workspace_root: str = "./workspace",
    logs_root: str = "./logs",
    global_budget_secs: float = 43200.0,
    num_gpus: int = 1,
    num_cpus: int = 8,
    gpu_ids: list[int] | None = None,
    use_thompson_sampling: bool = True,
    enable_persistent_checkpoint: bool = True,
    checkpoint_path: str | None = None,
    thompson_state_path: str | None = None,
):
    """Build and return the RecHarness orchestrator for diversity_v3 (DPP
    multi-window search diversity re-ranking).

    Args:
        llm_provider: 'openai' (default, company gateway) | 'volcengine' | 'anthropic'.
        model_id: Model name.
        sample_path: Absolute or relative path to the request-log dataset (Parquet or
            ORC, directory or file) -- see gagc/benchmarks/diversity_v3/vendor's
            DATA_DESCRIPTION.md. Written into the cold-start config.yaml's data.sample_path
            so every trial reads the same shared, un-copied dataset.
        vec_path: Path to the matching goods-vector dataset (Parquet or ORC).
        cold_start: Template name: 'diversity_dpp' (fixed DPP algorithm + tunable config.yaml).
        workspace_root: Root for the incumbent train.py + config.yaml.
        logs_root: Directory for per-iteration JSON logs.
        global_budget_secs: Total compute-time budget.
        num_gpus: Not used by this CPU-only benchmark; kept only for trial-slot scheduling
            parity with the other agent factories. Default 1 (no real parallelism benefit
            beyond CPU core count for this benchmark).
        num_cpus: CPU cores available.
        gpu_ids: Explicit slot IDs to use (defaults to range(num_gpus)).
        use_thompson_sampling: If False, run the textual-gradient ablation.
        enable_persistent_checkpoint: Persist LangGraph checkpoints and Thompson
            state under logs_root so the same thread_id can resume after restart.
        checkpoint_path: Optional SQLite checkpoint DB path. Defaults to
            logs_root/langgraph_checkpoints.sqlite. Set to "memory" to disable.
        thompson_state_path: Optional Thompson state JSON path. Defaults to
            logs_root/thompson_state.json. Set to "memory" to disable.

    Usage:
        from gagc.agent import create_diversity_agent
        agent = create_diversity_agent(
            sample_path="./input/diversity_v3/sample_500.parquet",
            vec_path="./input/diversity_v3/vec_filtered.parquet",
            workspace_root="./workspace_diversity",
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Maximize combined_pass_rate on diversity_v3."}]},
            config={"configurable": {"thread_id": "diversity-run-001"}},
        )
    """
    _tools_module._BENCHMARK_MODE = "diversity_v3"
    _tools_module._DIVERSITY_SAMPLE_PATH = os.path.abspath(sample_path)
    _tools_module._DIVERSITY_VEC_PATH = os.path.abspath(vec_path)
    _tools_module._ROUTING_POLICY = "thompson" if use_thompson_sampling else "textual_gradient"
    os.environ["GAGC_COLD_START"] = cold_start

    _tools_module._SERVER_CONFIG = ServerConfig(
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        cpus_per_trial=num_cpus // max(num_gpus, 1),
        gpu_ids=gpu_ids or [],
    )
    _tools_module._LOGS_ROOT = os.path.abspath(logs_root)
    _tools_module._WORKSPACE_ROOT = os.path.abspath(workspace_root)

    os.makedirs(logs_root, exist_ok=True)
    _init_workspace(workspace_root, cold_start)

    abs_workspace = os.path.abspath(workspace_root)
    abs_logs_root = os.path.abspath(logs_root)

    # Point the cold-start config.yaml's data section at the real dataset -- the
    # template ships with the source task's own placeholder paths.
    import yaml
    config_path = os.path.join(abs_workspace, "config.yaml")
    with open(config_path, encoding="utf-8") as f:
        cold_start_config = yaml.safe_load(f)
    cold_start_config["data"] = {
        "sample_path": _tools_module._DIVERSITY_SAMPLE_PATH,
        "vec_path": _tools_module._DIVERSITY_VEC_PATH,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cold_start_config, f, sort_keys=False)

    system_prompt = (
        DIVERSITY_ORCHESTRATOR_PROMPT
        + f"\n\n# Runtime context\n"
        f"## Virtual paths\n"
        f"- /workspace/config.yaml      — mutable scatter hyperparameters (RecHarness evolves this)\n"
        f"- /workspace/train.py         — fixed DPP algorithm, NEVER mutate this\n"
        f"- /thompson_state/state.json  — Thompson Sampling + SkillOpt memory state\n"
        f"- /logs/iteration_<N>.json    — per-iteration logs\n"
        f"\n## Real paths\n"
        f"- trial-tool script_path      : {abs_workspace}/config.yaml\n"
        f"- git diff workspace path     : {abs_workspace}\n"
        f"  (use: bash(\"cd {abs_workspace} && git diff config.yaml\") to generate code_diff)\n"
        f"\n## Data\n"
        f"- sample_path : {_tools_module._DIVERSITY_SAMPLE_PATH}\n"
        f"- vec_path    : {_tools_module._DIVERSITY_VEC_PATH}\n"
        f"- cold_start  : {cold_start}\n"
        f"\n## Budget\n"
        f"- global_budget : {global_budget_secs}s\n"
        f"- num_cpus      : {num_cpus}\n"
        f"\nRULE: read_file/write_file/ls MUST use virtual paths (/workspace/, /thompson_state/, /logs/).\n"
        f"NEVER pass a real absolute path to read_file. Use the real trial-tool script_path above for execute_trial, execute_trial_group, promote_winner, and evaluate_final_incumbent.\n"
    )
    if not use_thompson_sampling:
        system_prompt += TEXTUAL_GRADIENT_ROUTING_PROMPT

    from deepagents import create_deep_agent
    from deepagents.backends import (
        CompositeBackend,
        FilesystemBackend,
        LocalShellBackend,
        StateBackend,
        StoreBackend,
    )
    model = _build_model(llm_provider, model_id, api_key, base_url)
    store = _build_store(abs_logs_root, thompson_state_path, enable_persistent_checkpoint)
    _tools_module._STORE = store

    from gagc.state import ArmState, ThompsonState
    from gagc.tools import _active_arms
    default_state = ThompsonState(
        arms={arm: ArmState(name=arm) for arm in _active_arms()},
        global_budget=global_budget_secs,
        score_history=[],
        rejected_dims_buffer=[],
        skill_notes="Cold start: no prior knowledge.",
        pending_queue=[],
        experiment_skill=default_experiment_skill_for_policy(),
    )
    _seed_thompson_state(store, default_state)

    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/":    LocalShellBackend(root_dir=abs_workspace, virtual_mode=True),
            "/thompson_state/":  StoreBackend(namespace=lambda _rt: ("gagc", "thompson")),
            "/logs/":         FilesystemBackend(root_dir=abs_logs_root, virtual_mode=True),
        },
    )

    agent = create_deep_agent(
        model=model,
        tools=[
            propose_action_group,
            execute_trial_group,
            execute_trial,
            promote_winner,
            update_thompson_state,
            evaluate_final_incumbent,
        ],
        backend=backend,
        store=store,
        checkpointer=_build_checkpointer(abs_logs_root, checkpoint_path, enable_persistent_checkpoint),
        system_prompt=system_prompt,
    )

    return agent

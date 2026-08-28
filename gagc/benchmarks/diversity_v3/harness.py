"""diversity_v3 evaluation harness.

Runs the vendored `prepare.py` (frozen evaluation pipeline) as a subprocess
inside a trial's isolated workspace, then feeds its output through the
vendored `decision.py` for the keep/revert call. `prepare.py` resolves its
own `config.yaml` via `Path(__file__).resolve().parent` -- that is the
directory of the *script file being executed*, independent of the process's
cwd. So a copy of `prepare.py` must physically exist in `workspace_dir` and
be the one invoked (not the vendored original), or every trial would read
the vendored copy's own config.yaml instead of the trial's mutated one.

`config.yaml`'s `data.sample_path` / `data.vec_path` are expected to be
absolute paths (see `gagc/agent.py`'s diversity agent factory) pointing at a
shared, pre-built local dataset -- never copied per-trial.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_HERE, "vendor")
_VENDOR_PREPARE_PY = os.path.join(_VENDOR_DIR, "prepare.py")


@dataclass
class EvalResult:
    """Outcome of one diversity_v3 evaluation run."""

    ok: bool
    metrics: dict = field(default_factory=dict)  # flattened, e.g. "combined_pass_rate_mean"
    primary_metric: float = 0.0
    num_requests: int = 0
    num_errors: int = 0
    contingency_table: str = ""
    error_message: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


def _decision_module():
    """Import the vendored decision.py by file path (it isn't a package member)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gagc_diversity_v3_vendor_decision", os.path.join(_VENDOR_DIR, "decision.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(
    workspace_dir: str,
    mode: str = "scatter",
    num_workers: int = 0,
    timeout_secs: float = 3600.0,
) -> EvalResult:
    """Run prepare.py in `workspace_dir` (must contain config.yaml + train.py) and
    return flattened metrics. Does not apply keep/revert -- see `decide_keep`.
    """
    local_prepare_py = os.path.join(workspace_dir, "prepare.py")
    if not os.path.isfile(local_prepare_py):
        shutil.copy2(_VENDOR_PREPARE_PY, local_prepare_py)

    output_path = os.path.join(workspace_dir, "eval_results.json")
    cmd = [
        sys.executable, local_prepare_py,
        "--output", "eval_results.json",
        "--workers", str(num_workers),
    ]
    if mode == "baseline":
        cmd.append("--verify-baseline")

    try:
        proc = subprocess.run(
            cmd, cwd=workspace_dir, capture_output=True, timeout=timeout_secs, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return EvalResult(
            ok=False, error_message="timeout",
            stdout_tail=(exc.stdout or b"").decode(errors="replace")[-2000:],
            stderr_tail=(exc.stderr or b"").decode(errors="replace")[-2000:],
        )

    stdout = proc.stdout.decode(errors="replace")
    stderr = proc.stderr.decode(errors="replace")
    if proc.returncode != 0 or not os.path.isfile(output_path):
        return EvalResult(
            ok=False,
            error_message=stderr[-2000:] if stderr else f"prepare.py exited {proc.returncode}",
            stdout_tail=stdout[-2000:], stderr_tail=stderr[-2000:],
        )

    with open(output_path, encoding="utf-8") as f:
        report = json.load(f)

    decision = _decision_module()
    metrics = decision.extract_metrics(report)
    return EvalResult(
        ok=True,
        metrics=metrics,
        primary_metric=decision.get_primary_metric_value(metrics),
        num_requests=report.get("num_requests", 0),
        num_errors=report.get("num_errors", 0),
        contingency_table=report.get("contingency_table", ""),
        stdout_tail=stdout[-2000:], stderr_tail=stderr[-2000:],
    )


def decide_keep(new_metrics: dict, branch_best: dict | None, state: dict | None = None) -> tuple[bool, str]:
    """Keep/revert call, delegated to the vendored decision.py (see its docstring
    for the 4-gate tolerance logic: mean baseline floor, VecSim non-worsening,
    pass-rate non-collapse, combined_pass_rate non-collapse)."""
    return _decision_module().decide_keep(new_metrics, branch_best, state or {})

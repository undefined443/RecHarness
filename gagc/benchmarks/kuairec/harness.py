"""KuaiRec Watch Time Prediction — Benchmark Harness.

Entry point for RecHarness to evaluate a GR model on KuaiRec.

Usage (agent code)
------------------
    from gagc.benchmarks.kuairec.harness import evaluate

    result = evaluate(
        train_script = './working/best.py',
        train_data   = './input/train_data.npy',
        test_data    = './input/test_data.npy',
    )
    print(f"xAUC: {result.primary_metric:.4f}  MAE: {result.mae:.4f}")
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from gagc.benchmarks.kuairec.grader import compute_metrics
from gagc.benchmarks.kuairec.task import ALL_METRICS, PRIMARY_METRIC, TASK


@dataclass
class BenchmarkResult:
    """Result from evaluating one GR model run."""

    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def primary_metric(self) -> float:
        return self.metrics.get(PRIMARY_METRIC, 0.0)

    @property
    def xauc(self) -> float:
        return self.metrics.get("xAUC", 0.0)

    @property
    def mae(self) -> float:
        return self.metrics.get("MAE", float("inf"))

    def summary_table(self) -> str:
        lines = [
            f"{'Metric':<12} {'Value':>10}",
            "-" * 24,
            f"{'xAUC':<12} {self.metrics.get('xAUC', 0):.4f}",
            f"{'MAE':<12} {self.metrics.get('MAE', 0):.4f}",
        ]
        return "\n".join(lines)


def evaluate(
    train_script: str,
    train_data: str,
    test_data: str,
    verbose: bool = True,
) -> BenchmarkResult:
    """Evaluate a GR model by running train_script and reading the reported metrics.

    The train_script (best.py) must print metrics in a parseable format:
        XAUC=<float>
        MAE=<float>
    after training completes.

    Parameters
    ----------
    train_script:
        Path to best.py (the GR training script evolved by RecHarness).
    train_data:
        Path to train .npy file (KuaiRec features).
    test_data:
        Path to test .npy file (KuaiRec features).
    verbose:
        Print progress.

    Returns
    -------
    BenchmarkResult
    """
    if verbose:
        print(f"[benchmark] Running GR training + evaluation ...", flush=True)

    env = os.environ.copy()
    env["GR_TRAIN_DATA"] = os.path.abspath(train_data)
    env["GR_TEST_DATA"] = os.path.abspath(test_data)

    result = subprocess.run(
        [sys.executable, train_script],
        capture_output=True,
        env=env,
    )

    stdout = result.stdout.decode(errors="replace")
    stderr = result.stderr.decode(errors="replace")

    metrics = _parse_metrics(stdout)

    if verbose and metrics:
        print(f"  xAUC={metrics.get('xAUC', 0):.4f}  MAE={metrics.get('MAE', 0):.4f}")

    return BenchmarkResult(metrics=metrics)


def _parse_metrics(stdout: str) -> dict[str, float]:
    """Parse XAUC=<float> and MAE=<float> lines from stdout."""
    import re
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        m = re.search(r"XAUC[=:\s]+([0-9.]+)", line, re.IGNORECASE)
        if m:
            metrics["xAUC"] = float(m.group(1))
        m = re.search(r"\bMAE[=:\s]+([0-9.]+)", line, re.IGNORECASE)
        if m:
            metrics["MAE"] = float(m.group(1))
    return metrics

"""diversity_v3 — e-commerce search diversity re-ranking benchmark.

Frozen evaluation logic (`vendor/prepare.py`, `vendor/decision.py`) is
vendored verbatim from the source task package, not reimplemented.

Quick-start:
    from gagc.benchmarks.diversity_v3.harness import evaluate, decide_keep
    from gagc.benchmarks.diversity_v3.task import TASK
"""
from gagc.benchmarks.diversity_v3.harness import EvalResult, decide_keep, evaluate
from gagc.benchmarks.diversity_v3.task import TASK

__all__ = ["TASK", "EvalResult", "decide_keep", "evaluate"]

"""DPP Multi-Window Diversity (diversity_v3) — Task Specification.

An e-commerce search re-ranking task: given ~1000 ranked candidates, select
and order 10 products to balance ranking value against diversity, using a
fixed DPP (Determinantal Point Process) multi-window algorithm. Unlike
RecHarness's other benchmarks, there is no training step and the algorithm
implementation itself (`vendor/train.py`, `vendor/prepare.py`) is frozen —
the only mutable surface is the `scatter:` hyperparameter section of
`config.yaml`. See `vendor/decision.py` for the authoritative keep/revert
gating logic, reused as-is rather than reimplemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ── Primary metric for search / promotion ────────────────────────────
# The fraction of requests where VecSim, RankValue_top4, and RankValue_bottom6
# ALL pass simultaneously (see vendor/decision.py's ONLINE_BASELINE / decide_keep).
PRIMARY_METRIC: Final[str] = "combined_pass_rate_mean"
METRIC_MAXIMIZE: Final[bool] = True

# Pass-rate thresholds a branch must clear (vendor/decision.py, same values).
VECSIM_PASS_RATE_THRESHOLD: Final[float] = 0.60
RANKVALUE_PASS_RATE_THRESHOLD: Final[float] = 0.60
COMBINED_PASS_RATE_THRESHOLD: Final[float] = 0.60

# Pit-position exposure weights (double-column layout, same pair per row).
EXPOSURE_PROBS: Final[list[float]] = [
    1.0, 1.0, 0.418337, 0.418337, 0.167033,
    0.167033, 0.140998, 0.140998, 0.122709, 0.122709,
]


@dataclass(frozen=True)
class TaskSpec:
    """Immutable description of the diversity_v3 task."""

    name: str = "diversity_v3"
    description: str = (
        "E-commerce search diversity re-ranking: select and order 10 products "
        "from ~1000 ranked candidates using a fixed DPP multi-window algorithm. "
        "Only rank (score), fst_rank (fst_score), and vector embeddings may be "
        "used as algorithm input -- category (cat1/cat3) is evaluation-only. "
        "The algorithm implementation is frozen; only config.yaml's scatter "
        "hyperparameters are tunable."
    )
    primary_metric: str = PRIMARY_METRIC
    metric_maximize: bool = METRIC_MAXIMIZE

    # Cold-start template contract: the mutable artifact is config.yaml, not
    # a .py file. train.py is copied alongside it but never mutated.
    mutable_config_name: str = "config.yaml"
    fixed_algorithm_name: str = "train.py"


TASK = TaskSpec()

"""KuaiRec Watch Time Prediction — Task Specification.

Generative Regression (GR) task: predict watch time as a sequence generation problem.
Primary metrics: xAUC (maximize) and MAE (minimize).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# ── Primary metrics ───────────────────────────────────────────────────
PRIMARY_METRIC: Final[str] = "xAUC"
METRIC_MAXIMIZE: Final[bool] = True
ALL_METRICS: Final[list[str]] = ["xAUC", "MAE"]

# ── Evaluation constants ──────────────────────────────────────────────
EVAL_SEED: Final[int] = 2024


@dataclass(frozen=True)
class TaskSpec:
    """Immutable description of the KuaiRec Watch Time Prediction task."""

    name: str = "kuairec-watch-time"
    description: str = (
        "Watch time prediction on KuaiRec dataset using Generative Regression (GR). "
        "The model reformulates continuous watch time regression as sequence generation "
        "over a dynamic vocabulary built from watch-ratio quantiles. "
        "Maximize xAUC (ranking quality) and minimize MAE (absolute error)."
    )
    primary_metric: str = PRIMARY_METRIC
    metric_maximize: bool = METRIC_MAXIMIZE
    all_metrics: tuple[str, ...] = tuple(ALL_METRICS)
    eval_seed: int = EVAL_SEED

    # Data layout expected by the harness
    train_data_key: str = "train_data"    # env var / path to train .npy file
    test_data_key: str = "test_data"      # env var / path to test .npy file

    # Model output contract
    predict_fn_name: str = "evaluate_model"


TASK = TaskSpec()

"""Metric computation for spooky-author-identification.

local_log_loss(): used during search — fast, no dependency on grading.py.
official_log_loss(): used only for the final held-out evaluation, via the
vendored grading.grade() (see grading.py).
to_val_score(): converts log_loss (lower is better) into val_score (higher
is better), the convention every RecHarness benchmark and gagc.tools relies on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sklearn_log_loss

from gagc.benchmarks.spooky_author import grading
from gagc.benchmarks.spooky_author.task import CLASSES, RANDOM_BASELINE_LOG_LOSS


def local_log_loss(probs: np.ndarray, labels: list[str]) -> float:
    """Multi-class log loss of `probs` (n, 3) against string author labels."""
    y_true = [CLASSES.index(a) for a in labels]
    # Renormalize: float32 softmax rows can sum to e.g. 0.999999, which
    # otherwise trips sklearn's strict "probabilities must sum to one" check.
    normalized = probs / probs.sum(axis=1, keepdims=True)
    return float(sklearn_log_loss(y_true, normalized, labels=list(range(len(CLASSES)))))


def to_val_score(log_loss_value: float) -> float:
    """Map log_loss (lower is better) to val_score (higher is better).

    val_score = max(0, 1 - log_loss / RANDOM_BASELINE_LOG_LOSS):
      - log_loss == 0 (perfect)                -> val_score == 1.0
      - log_loss == RANDOM_BASELINE_LOG_LOSS    -> val_score == 0.0
      - log_loss worse than random guessing     -> clamped to 0.0

    This keeps val_score on the same "higher is better, floor at 0" scale
    used by every other benchmark, and strictly above the crash/OOM/timeout
    penalty scores (_CRASH_SCORE=0.0, _OOM_SCORE=-0.5, _TIMEOUT_SCORE=-1.0
    in gagc/tools.py) for any model that beats random guessing.
    """
    return max(0.0, 1.0 - log_loss_value / RANDOM_BASELINE_LOG_LOSS)


def official_log_loss(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """Official MLE-Bench Lite grading — only for the final test-set evaluation."""
    return grading.grade(submission, answers)

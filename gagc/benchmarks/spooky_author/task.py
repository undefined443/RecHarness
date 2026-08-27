"""Spooky Author Identification — Task Specification.

MLE-Bench Lite competition `spooky-author-identification`: 3-class text
classification. Predict the author (EAP/HPL/MWS) of a horror-fiction text
passage. This module defines the frozen task spec — the single source of
truth for what any MLE agent is expected to do.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

CLASSES: Final[list[str]] = ["EAP", "HPL", "MWS"]

# ── Primary metric for search / promotion ────────────────────────────
PRIMARY_METRIC: Final[str] = "log_loss"
METRIC_MAXIMIZE: Final[bool] = False
ALL_METRICS: Final[list[str]] = ["log_loss", "accuracy"]

# ── Split protocol (FROZEN — agents must not change) ─────────────────
# Layer 1: raw Kaggle train.csv -> public_train / private_test, matching
# mlebench's own mlebench/competitions/spooky-author-identification/prepare.py.
RAW_SPLIT_TEST_SIZE: Final[float] = 0.1
RAW_SPLIT_SEED: Final[int] = 0

# Layer 2: RecHarness's own search-time validation split, carved out of
# public_train. Trial code only ever reads train.csv/val.csv.
VAL_SPLIT_TEST_SIZE: Final[float] = 0.2
VAL_SPLIT_SEED: Final[int] = 42

# Random-guessing baseline log loss for 3 (roughly balanced) classes.
RANDOM_BASELINE_LOG_LOSS: Final[float] = math.log(3)

# Official Kaggle leaderboard medal thresholds (log_loss, lower is better).
# Final-report only — never used for search/promotion, mirrors how Amazon's
# secondary ranking metrics are final-report-only.
GOLD_THRESHOLD: Final[float] = 0.16506
SILVER_THRESHOLD: Final[float] = 0.26996
BRONZE_THRESHOLD: Final[float] = 0.29381
MEDIAN_THRESHOLD: Final[float] = 0.41879


@dataclass(frozen=True)
class TaskSpec:
    """Immutable description of the spooky-author-identification task."""

    name: str = "spooky-author-identification"
    description: str = (
        "3-class text classification (MLE-Bench Lite): identify the author "
        "(EAP=Edgar Allan Poe, HPL=H.P. Lovecraft, MWS=Mary Wollstonecraft "
        "Shelley) of a horror-fiction text passage. Minimize multi-class log loss."
    )
    classes: tuple[str, ...] = tuple(CLASSES)
    primary_metric: str = PRIMARY_METRIC
    metric_maximize: bool = METRIC_MAXIMIZE
    all_metrics: tuple[str, ...] = tuple(ALL_METRICS)

    # Data layout expected by the harness (paths supplied by the caller).
    train_data_key: str = "train_file"
    val_data_key: str = "val_file"
    test_data_key: str = "test_file"
    private_test_key: str = "private_test_file"

    # Predict function contract.
    predict_fn_name: str = "predict"
    predict_signature: str = (
        "def predict(texts: list[str]) -> np.ndarray  # shape (n, 3), columns "
        "ordered EAP, HPL, MWS, rows are probabilities summing to 1"
    )


TASK = TaskSpec()

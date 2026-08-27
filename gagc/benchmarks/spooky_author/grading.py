"""Official MLE-Bench Lite grading for `spooky-author-identification`.

Reimplemented from openai/mle-bench (MIT License) — same validation rules
(submission shape, probabilities summing to 1, in [0, 1]) and the same metric
(sklearn multi-class log loss). Rewritten as self-contained code here, rather
than depending on the `mlebench` package, to avoid its unrelated heavy
dependencies (docker, tensorflow, pycocotools, openai, ...) declared for the
full 75-competition / agent-sandboxing use case.

Source:
  https://github.com/openai/mle-bench/blob/main/mlebench/competitions/spooky-author-identification/grade.py
  https://github.com/openai/mle-bench/blob/main/mlebench/competitions/utils.py
  https://github.com/openai/mle-bench/blob/main/mlebench/grade_helpers.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from gagc.benchmarks.spooky_author.task import CLASSES

_TOLERANCE = 1e-6


class InvalidSubmissionError(Exception):
    """Raised when a submission cannot be graded."""


def one_hot_dfs_to_log_loss_inputs(
    submission_one_hot: pd.DataFrame,
    answers_one_hot: pd.DataFrame,
    id_column: str = "id",
) -> dict:
    """Align one-hot submission/answers frames by id and return log_loss() kwargs."""
    required_cols = set(answers_one_hot.columns) - {id_column}
    submission_cols = set(submission_one_hot.columns)

    if not submission_cols.issuperset(required_cols):
        raise InvalidSubmissionError(
            "The submission DataFrame is missing some columns required by the "
            f"`answers` DataFrame. Missing columns: {required_cols - submission_cols}."
        )
    if id_column not in submission_one_hot.columns:
        raise InvalidSubmissionError(f"Submission is missing id column '{id_column}'.")

    class_cols = [c for c in answers_one_hot.columns if c != id_column]
    submission_sorted = submission_one_hot.sort_values(by=id_column).reset_index(drop=True)
    answers_sorted = answers_one_hot.sort_values(by=id_column).reset_index(drop=True)

    y_true = answers_sorted[class_cols].values.argmax(axis=1)
    y_pred = submission_sorted[class_cols].values
    return {"y_true": y_true, "y_pred": y_pred, "labels": list(range(len(class_cols)))}


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    """Validate submission shape/probabilities, then build log_loss() kwargs."""
    if submission.shape != (len(answers), len(CLASSES) + 1):
        raise InvalidSubmissionError(
            f"Submission shape {submission.shape} does not match answers shape {answers.shape}."
        )
    if not np.all(np.isclose(submission.iloc[:, 1:].sum(axis=1), 1, atol=_TOLERANCE)):
        raise InvalidSubmissionError("Each row in submission should sum to one, as probabilities.")
    if not ((submission.iloc[:, 1:] >= 0) & (submission.iloc[:, 1:] <= 1)).all().all():
        raise InvalidSubmissionError(
            "All probabilities in submission DataFrame must be between 0 and 1."
        )

    return one_hot_dfs_to_log_loss_inputs(submission, answers, id_column="id")


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """Official MLE-Bench Lite grading: multi-class log loss (lower is better)."""
    log_loss_inputs = prepare_for_metric(submission, answers)
    # Positional call: sklearn renamed the y_pred kwarg to y_proba in 1.9, but
    # the positional signature is stable across the >=1.3 range we support.
    return float(log_loss(
        log_loss_inputs["y_true"], log_loss_inputs["y_pred"], labels=log_loss_inputs["labels"],
    ))

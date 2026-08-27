"""Unit tests for the spooky-author-identification benchmark — no real data required."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gagc.benchmarks.spooky_author.grader import local_log_loss, to_val_score
from gagc.benchmarks.spooky_author.grading import InvalidSubmissionError, grade
from gagc.benchmarks.spooky_author.protocol import (
    ProtocolViolation,
    check_probabilities,
    load_predict_fn,
)
from gagc.benchmarks.spooky_author.task import CLASSES, RANDOM_BASELINE_LOG_LOSS

# ================================================================== #
# grading.grade                                                       #
# ================================================================== #

def _one_hot(labels: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(0, index=range(len(labels)), columns=["id"] + CLASSES)
    df["id"] = range(len(labels))
    for i, label in enumerate(labels):
        df.loc[i, label] = 1
    return df


def test_grade_near_perfect_submission_is_near_zero():
    answers = _one_hot(["EAP", "HPL", "MWS"])
    submission = pd.DataFrame({
        "id": [0, 1, 2],
        "EAP": [0.98, 0.01, 0.01],
        "HPL": [0.01, 0.98, 0.01],
        "MWS": [0.01, 0.01, 0.98],
    })
    assert grade(submission, answers) < 0.1


def test_grade_uniform_submission_equals_random_baseline():
    answers = _one_hot(["EAP", "HPL", "MWS"])
    submission = pd.DataFrame({
        "id": [0, 1, 2],
        "EAP": [1 / 3] * 3, "HPL": [1 / 3] * 3, "MWS": [1 / 3] * 3,
    })
    assert grade(submission, answers) == pytest.approx(math.log(3), rel=1e-4)


def test_grade_wrong_shape_raises():
    answers = _one_hot(["EAP", "HPL"])
    submission = pd.DataFrame({"id": [0, 1, 2], "EAP": [0.5] * 3, "HPL": [0.5] * 3})
    with pytest.raises(InvalidSubmissionError, match="shape"):
        grade(submission, answers)


def test_grade_rows_not_summing_to_one_raises():
    answers = _one_hot(["EAP", "HPL"])
    submission = pd.DataFrame({
        "id": [0, 1], "EAP": [0.5, 0.5], "HPL": [0.5, 0.4], "MWS": [0.0, 0.0],
    })
    with pytest.raises(InvalidSubmissionError, match="sum to one"):
        grade(submission, answers)


def test_grade_out_of_range_probability_raises():
    answers = _one_hot(["EAP", "HPL"])
    submission = pd.DataFrame({
        "id": [0, 1], "EAP": [1.5, -0.5], "HPL": [-0.5, 1.5], "MWS": [0.0, 0.0],
    })
    with pytest.raises(InvalidSubmissionError, match="between 0 and 1"):
        grade(submission, answers)


# ================================================================== #
# grader.local_log_loss / to_val_score                                #
# ================================================================== #

def test_local_log_loss_matches_grade():
    probs = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    labels = ["EAP", "HPL", "MWS"]
    assert local_log_loss(probs, labels) == pytest.approx(0.10536, rel=1e-3)


def test_to_val_score_perfect_is_one():
    assert to_val_score(0.0) == pytest.approx(1.0)


def test_to_val_score_random_baseline_is_zero():
    assert to_val_score(RANDOM_BASELINE_LOG_LOSS) == pytest.approx(0.0, abs=1e-9)


def test_to_val_score_worse_than_random_clamped_at_zero():
    assert to_val_score(2 * RANDOM_BASELINE_LOG_LOSS) == 0.0


def test_to_val_score_never_negative():
    assert to_val_score(100.0) == 0.0


# ================================================================== #
# protocol.check_probabilities                                        #
# ================================================================== #

def test_check_probabilities_valid():
    probs = np.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])
    result = check_probabilities(probs, num_texts=2)
    assert result.shape == (2, 3)


def test_check_probabilities_wrong_shape_raises():
    with pytest.raises(ProtocolViolation, match="expected"):
        check_probabilities(np.zeros((3, 2)), num_texts=3)


def test_check_probabilities_not_summing_to_one_raises():
    with pytest.raises(ProtocolViolation, match="summing to 1"):
        check_probabilities(np.array([[0.5, 0.5, 0.5]]), num_texts=1)


def test_check_probabilities_out_of_range_raises():
    with pytest.raises(ProtocolViolation, match=r"\[0, 1\]"):
        check_probabilities(np.array([[1.5, -0.5, 0.0]]), num_texts=1)


def test_check_probabilities_not_array_like_raises():
    with pytest.raises(ProtocolViolation, match="not array-like"):
        check_probabilities("not an array", num_texts=1)


# ================================================================== #
# protocol.load_predict_fn                                            #
# ================================================================== #

def test_load_predict_fn_valid(tmp_path):
    script = tmp_path / "predict.py"
    script.write_text(
        "import numpy as np\n"
        "def predict(texts):\n"
        "    return np.full((len(texts), 3), 1/3)\n"
    )
    fn = load_predict_fn(str(script))
    result = fn(["a", "b"])
    assert result.shape == (2, 3)


def test_load_predict_fn_missing_file():
    with pytest.raises(FileNotFoundError):
        load_predict_fn("/nonexistent/predict.py")


def test_load_predict_fn_no_predict_symbol(tmp_path):
    script = tmp_path / "predict.py"
    script.write_text("# no predict function here\n")
    with pytest.raises(ProtocolViolation, match="callable named 'predict'"):
        load_predict_fn(str(script))

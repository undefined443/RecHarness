"""Unit tests for grader.py and protocol.py — no real data required."""
from __future__ import annotations

import math
import os
import sys
import tempfile

import pytest

from gagc.benchmarks.amazon_reviews.grader import (
    MetricAccumulator,
    aggregate_dataset_metrics,
    rank_positive,
)
from gagc.benchmarks.amazon_reviews.protocol import (
    ProtocolViolation,
    check_scores,
    load_predict_fn,
)


# ================================================================== #
# rank_positive                                                       #
# ================================================================== #

def test_rank_positive_is_best():
    scores = [1.0, 0.5, 0.3, 0.2]
    assert rank_positive(scores) == 0


def test_rank_positive_is_last():
    scores = [0.1, 0.9, 0.8, 0.7]
    assert rank_positive(scores) == 3


def test_rank_positive_tie_lower_index_wins():
    # index 0 and index 1 both have score 1.0 → index 0 should win (rank 0)
    scores = [1.0, 1.0, 0.5]
    rank = rank_positive(scores)
    assert rank == 0


# ================================================================== #
# MetricAccumulator                                                   #
# ================================================================== #

def test_accumulator_empty_raises():
    acc = MetricAccumulator()
    with pytest.raises(RuntimeError):
        acc.result()


def test_accumulator_rank0_perfect_scores():
    acc = MetricAccumulator()
    acc.update(0)  # rank 0 → in top-1, so in top-10 and top-20
    result = acc.result()
    assert result["HR@10"] == 1.0
    assert result["HR@20"] == 1.0
    assert result["NDCG@10"] == pytest.approx(1.0 / math.log2(2), rel=1e-4)
    assert result["NDCG@20"] == pytest.approx(1.0 / math.log2(2), rel=1e-4)


def test_accumulator_rank10_in_top20_not_top10():
    acc = MetricAccumulator()
    acc.update(10)  # rank 10 = 11th position → NOT in top-10, IS in top-20
    result = acc.result()
    assert result["HR@10"] == 0.0
    assert result["HR@20"] == 1.0
    assert result["NDCG@10"] == 0.0
    assert result["NDCG@20"] == pytest.approx(1.0 / math.log2(12), abs=5e-4)


def test_accumulator_rank20_misses_everything():
    acc = MetricAccumulator()
    acc.update(20)
    result = acc.result()
    assert result["HR@10"] == 0.0
    assert result["HR@20"] == 0.0
    assert result["NDCG@10"] == 0.0
    assert result["NDCG@20"] == 0.0


def test_accumulator_averages():
    acc = MetricAccumulator()
    acc.update(0)   # rank 0 → HR@10=1
    acc.update(20)  # rank 20 → HR@10=0
    result = acc.result()
    assert result["HR@10"] == pytest.approx(0.5, abs=1e-4)
    assert acc.count == 2


# ================================================================== #
# aggregate_dataset_metrics                                           #
# ================================================================== #

def test_aggregate_two_datasets():
    per_ds = {
        "A": {"HR@10": 0.4, "HR@20": 0.6, "NDCG@10": 0.2, "NDCG@20": 0.3},
        "B": {"HR@10": 0.6, "HR@20": 0.8, "NDCG@10": 0.4, "NDCG@20": 0.5},
    }
    agg = aggregate_dataset_metrics(per_ds)
    assert agg["HR@10"] == pytest.approx(0.5, abs=1e-4)
    assert "A/HR@10" in agg
    assert agg["A/HR@10"] == pytest.approx(0.4, abs=1e-4)


# ================================================================== #
# check_scores                                                        #
# ================================================================== #

def test_check_scores_valid():
    candidates = [1, 2, 3]
    scores = check_scores([0.9, 0.5, 0.1], candidates)
    assert scores == pytest.approx([0.9, 0.5, 0.1])


def test_check_scores_wrong_length():
    candidates = [1, 2, 3]
    with pytest.raises(ProtocolViolation, match="returned 2 scores"):
        check_scores([0.9, 0.5], candidates)


def test_check_scores_not_iterable():
    with pytest.raises(ProtocolViolation, match="not iterable"):
        check_scores(42.0, [1, 2, 3])  # type: ignore[arg-type]


def test_check_scores_non_numeric():
    with pytest.raises(ProtocolViolation, match="numeric"):
        check_scores(["a", "b", "c"], [1, 2, 3])


def test_check_scores_accepts_numpy():
    import numpy as np
    candidates = list(range(100))
    scores = np.random.rand(100)
    result = check_scores(scores, candidates)
    assert len(result) == 100


# ================================================================== #
# load_predict_fn                                                     #
# ================================================================== #

def test_load_predict_fn_valid(tmp_path):
    script = tmp_path / "predict.py"
    script.write_text(
        "def predict(user_id, history, candidates):\n"
        "    return [float(i) for i in range(len(candidates))]\n"
    )
    fn = load_predict_fn(str(script))
    result = fn(1, [10, 20], [5, 6, 7])
    assert result == [0.0, 1.0, 2.0]


def test_load_predict_fn_missing_file():
    with pytest.raises(FileNotFoundError):
        load_predict_fn("/nonexistent/predict.py")


def test_load_predict_fn_no_predict_symbol(tmp_path):
    script = tmp_path / "predict.py"
    script.write_text("# no predict function here\n")
    with pytest.raises(ProtocolViolation, match="callable named 'predict'"):
        load_predict_fn(str(script))

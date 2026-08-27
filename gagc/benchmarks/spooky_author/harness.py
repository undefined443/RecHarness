"""Spooky Author Identification — Benchmark Harness.

Entry point for RecHarness to evaluate a trial's predict.py.

Usage (agent code)
------------------
    from gagc.benchmarks.spooky_author.harness import evaluate

    result = evaluate(
        predict_script="./working/predict.py",
        mode="val",
        val_file="./input/spooky_author/val.csv",
    )
    print(f"log_loss={result.metrics['log_loss']:.4f}  val_score={result.val_score:.4f}")
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from gagc.benchmarks.spooky_author import grader
from gagc.benchmarks.spooky_author.protocol import check_probabilities, load_predict_fn
from gagc.benchmarks.spooky_author.task import CLASSES, PRIMARY_METRIC


@dataclass
class BenchmarkResult:
    """Result from evaluating one spooky-author-identification predict.py run."""

    metrics: dict[str, float] = field(default_factory=dict)
    val_score: float = 0.0
    """Higher is better — see grader.to_val_score() for the log_loss mapping."""

    @property
    def primary_metric(self) -> float:
        return self.metrics.get(PRIMARY_METRIC, float("inf"))

    def summary_table(self) -> str:
        lines = [
            f"{'Metric':<12} {'Value':>10}",
            "-" * 24,
        ]
        for name, value in self.metrics.items():
            lines.append(f"{name:<12} {value:>10.4f}")
        return "\n".join(lines)


def evaluate(
    predict_script: str,
    mode: str = "val",
    val_file: str | None = None,
    test_file: str | None = None,
    private_test_file: str | None = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """Evaluate a trial's predict.py.

    Parameters
    ----------
    predict_script:
        Path to predict.py exposing predict(texts) -> np.ndarray (n, 3).
    mode:
        "val": score against the local val.csv split — used during search.
        "test": score against the held-out test.csv / private_test.csv via
            the official grading logic — used only by the final-incumbent
            evaluation, never during search.
    """
    predict_fn = load_predict_fn(predict_script)

    if mode == "val":
        if not val_file:
            raise ValueError("val_file is required for mode='val'")
        df = pd.read_csv(val_file)
        texts = df["text"].tolist()
        labels = df["author"].tolist()

        probs = check_probabilities(predict_fn(texts), len(texts), context="val")
        log_loss_value = grader.local_log_loss(probs, labels)
        true_idx = [CLASSES.index(a) for a in labels]
        accuracy = float((probs.argmax(axis=1) == true_idx).mean())

        metrics = {"log_loss": round(log_loss_value, 6), "accuracy": round(accuracy, 4)}
        val_score = grader.to_val_score(log_loss_value)
        if verbose:
            print(f"[spooky_author] val log_loss={log_loss_value:.4f} accuracy={accuracy:.4f}")
        return BenchmarkResult(metrics=metrics, val_score=val_score)

    if mode == "test":
        if not test_file or not private_test_file:
            raise ValueError("test_file and private_test_file are required for mode='test'")
        test_df = pd.read_csv(test_file)
        texts = test_df["text"].tolist()

        probs = check_probabilities(predict_fn(texts), len(texts), context="test")
        submission = pd.DataFrame({"id": test_df["id"].values})
        for i, cls in enumerate(CLASSES):
            submission[cls] = probs[:, i]

        answers = pd.read_csv(private_test_file)
        log_loss_value = grader.official_log_loss(submission, answers)

        metrics = {"log_loss": round(log_loss_value, 6)}
        val_score = grader.to_val_score(log_loss_value)
        if verbose:
            print(f"[spooky_author] official test log_loss={log_loss_value:.4f}")
        return BenchmarkResult(metrics=metrics, val_score=val_score)

    raise ValueError(f"Unknown mode: {mode!r}. Expected 'val' or 'test'.")

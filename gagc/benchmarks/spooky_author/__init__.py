"""Spooky Author Identification Benchmark (MLE-Bench Lite).

Quick-start:

    from gagc.benchmarks.spooky_author import evaluate, TASK

    result = evaluate(
        predict_script="./working/predict.py",
        mode="val",
        val_file="./input/spooky_author/val.csv",
    )
    print(f"log_loss={result.metrics['log_loss']:.4f}  val_score={result.val_score:.4f}")
"""
from gagc.benchmarks.spooky_author.harness import BenchmarkResult, evaluate
from gagc.benchmarks.spooky_author.task import TASK

__all__ = ["evaluate", "BenchmarkResult", "TASK"]

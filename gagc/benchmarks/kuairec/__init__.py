"""KuaiRec Watch Time Prediction Benchmark (Generative Regression).

Quick-start:

    from gagc.benchmarks.kuairec import evaluate, TASK

    result = evaluate(
        train_script = './working/best.py',
        train_data   = './input/train_data.npy',
        test_data    = './input/test_data.npy',
    )
    print(f"xAUC={result.xauc:.4f}  MAE={result.mae:.4f}")
"""
from gagc.benchmarks.kuairec.harness import BenchmarkResult, evaluate
from gagc.benchmarks.kuairec.task import TASK

__all__ = ["evaluate", "BenchmarkResult", "TASK"]

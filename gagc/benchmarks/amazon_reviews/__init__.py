"""Amazon Reviews Sequential Recommendation Benchmark.

Quick-start:

    from gagc.benchmarks.amazon_reviews import evaluate, evaluate_val_fast, TASK

    # Fast val feedback during training:
    result = evaluate_val_fast(predict_script='./working/predict.py',
                               data_dir='./input/trainval')
    print(result.primary_metric)

    # Final test evaluation (call once after training):
    result = evaluate(predict_script='./working/predict.py',
                      data_dir='./input/trainval',
                      test_dir='./input/test',
                      mode='test')
    print(result.summary_table())
"""
from gagc.benchmarks.amazon_reviews.harness import (
    BenchmarkResult,
    DatasetResult,
    evaluate,
    evaluate_val_fast,
)
from gagc.benchmarks.amazon_reviews.task import DATASETS, TASK

__all__ = [
    "DATASETS",
    "TASK",
    "BenchmarkResult",
    "DatasetResult",
    "evaluate",
    "evaluate_val_fast",
]

"""Protocol guard — enforces the frozen predict() contract.

Mirrors gagc.benchmarks.amazon_reviews.protocol: validates a trial's
predict.py before any metrics are recorded.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Callable

import numpy as np

from gagc.benchmarks.spooky_author.task import CLASSES


class ProtocolViolation(Exception):
    """Raised when an agent violates the evaluation protocol."""


def load_predict_fn(predict_script: str) -> Callable:
    """Import predict_script and return its `predict` callable.

    Each call uses a unique module name derived from the script path so that
    loading predict.py from multiple workspaces in the same process does not
    cause sys.modules collisions.

    Parameters
    ----------
    predict_script:
        Absolute or relative path to a Python script that exposes:
            def predict(texts: list[str]) -> np.ndarray  # shape (n, 3)

    Raises
    ------
    FileNotFoundError
        If the script does not exist.
    ProtocolViolation
        If the script does not define a callable named 'predict'.
    """
    path = os.path.abspath(predict_script)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"predict_script not found: {path}")

    module_name = f"_spooky_predict_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "predict") or not callable(module.predict):
        raise ProtocolViolation(
            f"'{path}' must define a callable named 'predict'. "
            "Expected signature: predict(texts: list[str]) -> np.ndarray of shape (n, 3)"
        )
    return module.predict


def check_probabilities(probs: object, num_texts: int, *, context: str = "") -> np.ndarray:
    """Validate predict() output: shape (n, len(CLASSES)), rows in [0,1] summing to 1.

    Returns the validated array.

    Raises
    ------
    ProtocolViolation
        On shape mismatch or invalid probabilities.
    """
    prefix = f"[{context}] " if context else ""

    try:
        arr = np.asarray(probs, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ProtocolViolation(
            f"{prefix}predict() return value is not array-like: {exc}"
        ) from exc

    expected_shape = (num_texts, len(CLASSES))
    if arr.shape != expected_shape:
        raise ProtocolViolation(
            f"{prefix}predict() returned shape {arr.shape}, expected {expected_shape}."
        )
    if not np.all(np.isclose(arr.sum(axis=1), 1.0, atol=1e-6)):
        raise ProtocolViolation(
            f"{prefix}predict() rows must be probabilities summing to 1."
        )
    if not ((arr >= 0.0) & (arr <= 1.0)).all():
        raise ProtocolViolation(f"{prefix}predict() probabilities must be in [0, 1].")

    return arr

"""Protocol guard for KuaiRec benchmark."""
from __future__ import annotations

import importlib.util
import os
from typing import Callable


class ProtocolViolation(Exception):
    pass


def load_evaluate_fn(train_script: str) -> Callable:
    """Import train_script and return its `evaluate_model` callable.

    The function signature expected:
        def evaluate_model(train_data_path, test_data_path) -> dict[str, float]
    Returns {"xAUC": float, "MAE": float}.
    """
    path = os.path.abspath(train_script)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"train_script not found: {path}")

    module_name = f"_gr_train_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn_name = "evaluate_model"
    if not hasattr(module, fn_name) or not callable(getattr(module, fn_name)):
        raise ProtocolViolation(
            f"'{path}' must define a callable named '{fn_name}'. "
            "Expected signature: evaluate_model(train_data_path, test_data_path) -> dict"
        )
    return getattr(module, fn_name)

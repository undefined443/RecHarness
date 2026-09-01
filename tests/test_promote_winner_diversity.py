"""Regression coverage for promote_winner's train.py sync gate.

promote_winner keeps best.py and train.py in sync for the training benchmarks
(amazon_reviews / kuairec / spooky_author), where both are the same source file.
For diversity_v3 script_path is config.yaml and train.py is the frozen DPP
algorithm ("NEVER mutate") -- syncing the promoted config content into it turns
train.py into YAML text and crashes every subsequent trial. The gate keys on
_BENCHMARK_MODE, not on train.py merely existing on disk.
"""

from __future__ import annotations

import json

import pytest

from gagc import tools

_FROZEN_TRAIN_PY = "def scatter(*args, **kwargs):\n    return None\n"
_OLD_CONFIG = "scatter:\n  slidingWindowSize: 20\n"
_NEW_CONFIG = "scatter:\n  slidingWindowSize: 15\n"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(tools, "_STORE", None)
    monkeypatch.setattr(tools, "_LOGS_ROOT", "")
    monkeypatch.setattr(tools, "_WORKSPACE_ROOT", "")
    monkeypatch.setattr(tools, "_LAST_TRIAL_RESULTS", [])


def _winner_json(new_code):
    return json.dumps([
        {
            "val_score": 0.55,
            "val_metrics": {"MAE": 0.1},
            "spec": {"arm": "tune_window_config", "delta": -1.0},
            "mutated_code_content": new_code,
            "error_message": None,
        }
    ])


def test_diversity_v3_promote_leaves_train_py_untouched(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "diversity_v3")
    config_path = tmp_path / "config.yaml"
    train_path = tmp_path / "train.py"
    config_path.write_text(_OLD_CONFIG, encoding="utf-8")
    train_path.write_text(_FROZEN_TRAIN_PY, encoding="utf-8")

    out = json.loads(tools.promote_winner(
        trial_results_json=_winner_json(_NEW_CONFIG),
        script_path=str(config_path),
    ))

    assert out["promoted"] is True
    assert config_path.read_text(encoding="utf-8") == _NEW_CONFIG
    assert train_path.read_text(encoding="utf-8") == _FROZEN_TRAIN_PY
    assert out["train_py_updated"] is False


def test_training_benchmark_promote_still_syncs_train_py(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "amazon_reviews")
    best_path = tmp_path / "best.py"
    train_path = tmp_path / "train.py"
    old_code = "MODEL = 'baseline'\n"
    new_code = "MODEL = 'tuned'\n"
    best_path.write_text(old_code, encoding="utf-8")
    train_path.write_text(old_code, encoding="utf-8")

    out = json.loads(tools.promote_winner(
        trial_results_json=_winner_json(new_code),
        script_path=str(best_path),
    ))

    assert out["promoted"] is True
    assert best_path.read_text(encoding="utf-8") == new_code
    assert train_path.read_text(encoding="utf-8") == new_code
    assert out["train_py_updated"] is True

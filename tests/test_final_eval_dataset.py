"""Coverage for evaluate_final_incumbent's diversity_v3 dataset switch.

diversity_v3 search runs on a small sample; the final report must run on the full
production dataset. evaluate_final_incumbent swaps config.yaml's data.sample_path
to _DIVERSITY_FINAL_SAMPLE_PATH for that one call and restores the file afterward,
recording full_dataset_eval / evaluated_sample_path in the result. With no final
path configured it stays on the search sample and flags an hr= subset.
"""

from __future__ import annotations

import json
import types

import pytest
import yaml

from gagc import tools

_SEARCH_SAMPLE = "/data/diversity/sample_0721_clean/hr=00"
_FULL_SAMPLE = "/data/diversity/sample_0721_clean"
_VEC = "/data/diversity/vec_0721"


def _write_config(path):
    path.write_text(
        yaml.safe_dump(
            {"data": {"sample_path": _SEARCH_SAMPLE, "vec_path": _VEC},
             "scatter": {"slidingWindowSize": 20}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "diversity_v3")
    monkeypatch.setattr(tools, "_LOGS_ROOT", "")
    monkeypatch.setattr(tools, "_DIVERSITY_FINAL_SAMPLE_PATH", "")


def _stub_evaluate(capture, exc=None):
    def _run(workspace_dir, **_kwargs):
        with open(f"{workspace_dir}/config.yaml", encoding="utf-8") as f:
            capture["sample_path"] = yaml.safe_load(f)["data"]["sample_path"]
        if exc is not None:
            raise exc
        return types.SimpleNamespace(
            ok=True, metrics={"combined_pass_rate_mean": 0.03}, primary_metric=0.03,
            num_requests=589413, num_errors=0, contingency_table="",
            error_message=None, stdout_tail="", stderr_tail="",
        )
    return _run


def test_switches_to_full_dataset_and_restores(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "_DIVERSITY_FINAL_SAMPLE_PATH", _FULL_SAMPLE)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    original = config_path.read_text(encoding="utf-8")
    capture = {}
    monkeypatch.setattr("gagc.benchmarks.diversity_v3.harness.evaluate", _stub_evaluate(capture))

    out = json.loads(tools.evaluate_final_incumbent(script_path=str(config_path)))

    assert capture["sample_path"] == _FULL_SAMPLE
    assert config_path.read_text(encoding="utf-8") == original
    assert out["full_dataset_eval"] is True
    assert out["evaluated_sample_path"] == _FULL_SAMPLE
    assert "warning" not in out


def test_explicit_sample_path_arg_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "_DIVERSITY_FINAL_SAMPLE_PATH", _FULL_SAMPLE)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    capture = {}
    monkeypatch.setattr("gagc.benchmarks.diversity_v3.harness.evaluate", _stub_evaluate(capture))

    tools.evaluate_final_incumbent(script_path=str(config_path), sample_path="/data/other")

    assert capture["sample_path"] == "/data/other"


def test_no_final_path_keeps_search_sample_and_warns(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    capture = {}
    monkeypatch.setattr("gagc.benchmarks.diversity_v3.harness.evaluate", _stub_evaluate(capture))

    out = json.loads(tools.evaluate_final_incumbent(script_path=str(config_path)))

    assert capture["sample_path"] == _SEARCH_SAMPLE
    assert out["full_dataset_eval"] is False
    assert out["evaluated_sample_path"] == _SEARCH_SAMPLE
    assert "hr= partition subset" in out["warning"]


def test_config_restored_when_eval_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "_DIVERSITY_FINAL_SAMPLE_PATH", _FULL_SAMPLE)
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    original = config_path.read_text(encoding="utf-8")
    monkeypatch.setattr(
        "gagc.benchmarks.diversity_v3.harness.evaluate",
        _stub_evaluate({}, exc=RuntimeError("prepare.py died")),
    )

    with pytest.raises(RuntimeError):
        tools.evaluate_final_incumbent(script_path=str(config_path))

    assert config_path.read_text(encoding="utf-8") == original

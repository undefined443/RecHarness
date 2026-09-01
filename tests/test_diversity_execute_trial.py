"""Integration seam: _execute_diversity_trial dispatching to the diversity_reactor.

Covers the wiring that neither the reactor unit tests nor the real-model smoke
test exercise: the reactor-backend branch in _execute_diversity_trial reading the
trial's config.yaml, honouring reactor_attempts, writing the mutation back, and
the no-model / generation-failure guards -- all with a fake model, no network.
"""

import json
import pathlib
import shutil

import pytest

from gagc import tools
from gagc.benchmarks.diversity_v3 import harness as _harness
from gagc.schemas import MutationSpec

_TEMPLATE = "gagc/templates/diversity_dpp/config.yaml"
_TEMPLATE_TEXT = pathlib.Path(_TEMPLATE).read_text()


class _FakeModel:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return type("_Msg", (), {"content": self._responses[min(self.calls - 1, len(self._responses) - 1)]})()


def _spec(**extra):
    base = {
        "implementation_backend": "diversity_reactor",
        "implementation_prompt": "edit scatter only",
        "allowed_files": ["config.yaml"],
        "reactor_attempts": 3,
    }
    base.update(extra)
    return MutationSpec(
        dimension="tune_dwPower", delta=1.0, estimated_cost_secs=60.0, code_diff="",
        arm="tune_diversity_strength", code_hint="raise dwPower", hypothesis="diversification too weak",
        **base,
    )


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    shutil.copy(_TEMPLATE, cfg)
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "diversity_v3")
    monkeypatch.setattr(tools, "_LOGS_ROOT", str(tmp_path))
    return cfg


def _stub_eval(monkeypatch, *, ok=True, primary=0.42):
    def fake_evaluate(workspace_dir, **kw):
        return _harness.EvalResult(
            ok=ok, metrics={"combined_pass_rate_mean": primary}, primary_metric=primary,
            error_message=None if ok else "boom",
        )
    monkeypatch.setattr(_harness, "evaluate", fake_evaluate)
    monkeypatch.setattr(_harness, "decide_keep", lambda *a, **k: (True, "ok"))


def test_reactor_branch_writes_mutation_and_reaches_evaluate(workspace, monkeypatch):
    mutated = workspace.read_text().replace("dwPower: 1.0", "dwPower: 2.0")
    monkeypatch.setattr(tools, "_LLM_MODEL", _FakeModel([mutated]))
    _stub_eval(monkeypatch, primary=0.55)

    out = json.loads(tools._execute_diversity_trial(_spec(), str(workspace), 60.0, 0))

    assert "dwPower: 2.0" in workspace.read_text()          # mutation written back
    assert out["error_message"] is None
    assert out["val_score"] == 0.55                          # evaluate result surfaced


def test_missing_model_returns_crash(workspace, monkeypatch):
    monkeypatch.setattr(tools, "_LLM_MODEL", None)
    called = []
    monkeypatch.setattr(_harness, "evaluate", lambda *a, **k: called.append(1))

    out = json.loads(tools._execute_diversity_trial(_spec(), str(workspace), 60.0, 0))

    assert out["val_score"] == tools._CRASH_SCORE
    assert "no LLM model" in out["error_message"]
    assert not called                                       # never got to evaluate
    assert workspace.read_text() == _TEMPLATE_TEXT           # config untouched


def test_generation_failure_returns_crash_and_honours_reactor_attempts(workspace, monkeypatch):
    model = _FakeModel(["not: valid: yaml:"])                # always rejected by _validate
    monkeypatch.setattr(tools, "_LLM_MODEL", model)
    monkeypatch.setattr(_harness, "evaluate", lambda *a, **k: pytest.fail("evaluate should not run"))

    out = json.loads(tools._execute_diversity_trial(_spec(reactor_attempts=2), str(workspace), 60.0, 0))

    assert out["val_score"] == tools._CRASH_SCORE
    assert "could not produce" in out["error_message"]
    assert model.calls == 2                                  # reactor_attempts respected

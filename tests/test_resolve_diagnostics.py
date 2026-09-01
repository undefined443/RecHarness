"""Unit coverage for tools._resolve_diagnostics.

propose_action_group's diagnostics argument used to require the LLM to paste
the previous best TrialResult JSON verbatim -- which GLM-5.2 truncates once the
context grows, crashing the search loop. The "__LAST_BEST__" sentinel moves
that lookup into the tool (read the cached previous group, pick its best valid
trial); a literal JSON object stays accepted, and a corrupt one degrades to {}.
"""

from __future__ import annotations

import json

import pytest

from gagc import tools


def _result(val_score, **extra):
    base = {"val_score": val_score, "spec": {"arm": "tune_x"}, "error_message": None}
    base.update(extra)
    return base


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    monkeypatch.setattr(tools, "_LAST_TRIAL_RESULTS", [])
    monkeypatch.setattr(tools, "_LOGS_ROOT", "")


def test_sentinel_returns_best_valid_trial_from_cache(monkeypatch):
    monkeypatch.setattr(tools, "_LAST_TRIAL_RESULTS", [
        _result(0.41),
        _result(0.57, stdout_tail="best one"),
        _result(tools._CRASH_SCORE, error_message="boom"),
    ])

    diag = tools._resolve_diagnostics("__LAST_BEST__")

    assert diag["val_score"] == 0.57
    assert diag["stdout_tail"] == "best one"


def test_sentinel_returns_empty_when_cache_is_empty():
    assert tools._resolve_diagnostics("__LAST_BEST__") == {}


def test_sentinel_returns_empty_when_group_all_failed(monkeypatch):
    monkeypatch.setattr(tools, "_LAST_TRIAL_RESULTS", [
        _result(tools._CRASH_SCORE),
        _result(tools._TIMEOUT_SCORE, timed_out=True),
    ])

    assert tools._resolve_diagnostics("__LAST_BEST__") == {}


@pytest.mark.parametrize("raw", ["", "   ", "{}"])
def test_blank_and_braces_return_empty(raw):
    assert tools._resolve_diagnostics(raw) == {}


def test_corrupt_json_degrades_to_empty():
    truncated = '{"val_score": 0.42, "stdout_tail": "aaaa'
    assert tools._resolve_diagnostics(truncated) == {}


def test_literal_json_object_still_accepted():
    payload = {"val_score": 0.5, "error_message": "x"}
    assert tools._resolve_diagnostics(json.dumps(payload)) == payload


def test_non_object_json_returns_empty():
    assert tools._resolve_diagnostics("[1, 2, 3]") == {}

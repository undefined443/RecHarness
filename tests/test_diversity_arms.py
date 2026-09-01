"""diversity_v3 arm wiring: every arm delegates implementation to diversity_reactor."""

from __future__ import annotations

from gagc import tools
from gagc.schemas import MutationSpec


def test_diversity_reactor_backend_attached_to_every_arm(monkeypatch):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "diversity_v3")

    for arm in tools._active_arms():
        fields = tools._claude_code_backend_fields_for_arm(arm)
        assert fields["implementation_backend"] == "diversity_reactor", arm
        assert fields["allowed_files"] == ["config.yaml"]
        assert fields["reactor_attempts"] == 3
        assert fields["implementation_prompt"].strip()


def test_diversity_backend_not_claude_code(monkeypatch):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "diversity_v3")

    spec = MutationSpec(
        dimension="tune_dwPower",
        delta=1.0,
        estimated_cost_secs=60.0,
        code_diff="",
        **tools._claude_code_backend_fields_for_arm("tune_diversity_strength"),
    )
    assert tools._uses_diversity_reactor_backend(spec) is True
    assert tools._uses_claude_code_backend(spec) is False


def test_prepare_specs_clears_payload_for_backend_delegated_spec(monkeypatch):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "diversity_v3")

    raw = [{
        "dimension": "tune_dwPower",
        "arm": "tune_diversity_strength",
        "delta": 1.0,
        "estimated_cost_secs": 60.0,
        "implementation_backend": "diversity_reactor",
        "code_content": "scatter: {should: be, wiped: true}",
        "code_edits": [{"find": "a", "replace": "b"}],
    }]
    prepared, err = tools._prepare_specs_for_execution(
        specs_json=__import__("json").dumps(raw),
        hypotheses_json="{}",
        trial_group_id="t",
        script_path="/tmp/config.yaml",
    )
    assert err is None
    assert prepared[0]["code_content"] == ""
    assert prepared[0]["code_edits"] == []
    assert prepared[0]["implementation_backend"] == "diversity_reactor"

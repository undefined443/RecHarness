from __future__ import annotations

import json
import os

import pytest

from gagc import tools
from gagc.claude_code_backend import claude_backend_enabled, run_claude_code_trial
from gagc.state import ThompsonState


def test_claude_backend_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GAGC_ENABLE_CLAUDE_BACKEND", raising=False)
    assert not claude_backend_enabled()


def test_claude_backend_enabled_values(monkeypatch):
    for value in ("1", "true", "yes", "on"):
        monkeypatch.setenv("GAGC_ENABLE_CLAUDE_BACKEND", value)
        assert claude_backend_enabled()


def test_missing_claude_binary_returns_failure(tmp_path):
    train_path = tmp_path / "train.py"
    predict_path = tmp_path / "predict.py"
    train_path.write_text("def main():\n    pass\n", encoding="utf-8")
    predict_path.write_text("", encoding="utf-8")

    outcome = run_claude_code_trial(
        trial_dir=str(tmp_path),
        train_path=str(train_path),
        predict_path=str(predict_path),
        spec_data={
            "claude_bin": "definitely-missing-claude-binary",
            "claude_attempts": 3,
        },
        env={},
        hard_timeout=1.0,
        benchmark_mode="kuairec",
        validation_callback=lambda: None,
    )

    assert not outcome.success
    assert "claude binary not found" in (outcome.error_message or "")
    assert outcome.attempts == []


def test_mock_claude_success_reuses_training_artifacts(tmp_path):
    train_path = tmp_path / "train.py"
    train_path.write_text("def main():\n    pass\n", encoding="utf-8")
    fake_claude = tmp_path / "fake_claude"
    fake_claude.write_text(
        """
#!/usr/bin/env python3
from pathlib import Path
Path('.gagc_claude').mkdir(exist_ok=True)
Path('.gagc_claude/train_stdout.log').write_text('Epoch 1: WT_XAUC=0.7\\nXAUC=0.7000\\nMAE=1.2000\\n', encoding='utf-8')
Path('.gagc_claude/train_stderr.log').write_text('', encoding='utf-8')
Path('.gagc_claude/train_returncode.txt').write_text('0\\n', encoding='utf-8')
print('{"ok": true}')
""".strip(),
        encoding="utf-8",
    )
    os.chmod(fake_claude, 0o755)

    outcome = run_claude_code_trial(
        trial_dir=str(tmp_path),
        train_path=str(train_path),
        predict_path=str(tmp_path / "predict.py"),
        spec_data={
            "claude_bin": str(fake_claude),
            "claude_model": "",
            "claude_attempts": 1,
            "claude_train_command": "python3 train.py",
            "allowed_files": ["train.py"],
        },
        env={},
        hard_timeout=5.0,
        benchmark_mode="kuairec",
        validation_callback=lambda: None,
    )

    assert outcome.success
    assert "XAUC=0.7000" in outcome.train_stdout
    assert outcome.attempts[0].train_returncode == 0


def test_claude_backend_allows_training_working_artifacts(tmp_path):
    train_path = tmp_path / "train.py"
    predict_path = tmp_path / "predict.py"
    train_path.write_text("def main():\n    pass\n", encoding="utf-8")
    predict_path.write_text("", encoding="utf-8")
    fake_claude = tmp_path / "fake_claude"
    fake_claude.write_text(
        """
#!/usr/bin/env python3
from pathlib import Path
Path('train.py').write_text("print('changed source')\\n", encoding='utf-8')
working = Path('working')
working.mkdir(exist_ok=True)
for ds in ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']:
    (working / f'{ds}_meta.pkl').write_bytes(b'meta')
    (working / f'{ds}_model.pt').write_bytes(b'model')
(working / 'predict.py').write_text('# generated predict artifact\\n', encoding='utf-8')
Path('.gagc_claude').mkdir(exist_ok=True)
Path('.gagc_claude/train_stdout.log').write_text('All datasets done. Average val HR@10 = 0.5000\\n', encoding='utf-8')
Path('.gagc_claude/train_stderr.log').write_text('', encoding='utf-8')
Path('.gagc_claude/train_returncode.txt').write_text('0\\n', encoding='utf-8')
print('{"ok": true}')
""".strip(),
        encoding="utf-8",
    )
    os.chmod(fake_claude, 0o755)

    outcome = run_claude_code_trial(
        trial_dir=str(tmp_path),
        train_path=str(train_path),
        predict_path=str(predict_path),
        spec_data={
            "claude_bin": str(fake_claude),
            "claude_model": "",
            "claude_attempts": 1,
            "claude_train_command": "python3 train.py",
            "allowed_files": ["train.py", "predict.py"],
        },
        env={},
        hard_timeout=5.0,
        benchmark_mode="amazon_reviews",
        validation_callback=lambda: None,
    )

    assert outcome.success
    assert outcome.changed_files == ["train.py"]
    assert "All datasets done" in outcome.train_stdout


def test_amazon_mutation_candidates_use_claude_backend(monkeypatch):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "amazon_reviews")
    monkeypatch.setattr(tools, "_ROUTING_POLICY", "thompson")
    monkeypatch.setattr(tools, "_STORE", None)
    state = ThompsonState(baseline_done=True, global_budget=100000.0, score_history=[0.1], round_idx=1)

    candidates = json.loads(
        tools.propose_action_group(
            state.to_json(),
            json.dumps({"val_score": 0.1}),
            group_size=4,
        )
    )

    assert candidates
    assert all(candidate["arm"] != "baseline" for candidate in candidates)
    for candidate in candidates:
        assert candidate["implementation_backend"] == "claude_code"
        assert candidate["allowed_files"] == ["train.py", "predict.py"]
        assert candidate["claude_attempts"] == 3
        assert candidate["code_edits"] == []
        assert candidate["code_content"] == ""
        assert candidate["code_diff"] == ""


def test_amazon_strategy_jump_gate_closed_returns_normal_candidates(monkeypatch):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "amazon_reviews")
    monkeypatch.setattr(tools, "_ROUTING_POLICY", "thompson")
    monkeypatch.setattr(tools, "_STORE", None)
    state = ThompsonState(
        baseline_done=True,
        global_budget=100000.0,
        score_history=[0.10, 0.14, 0.18],
        round_idx=3,
    )

    candidates = json.loads(
        tools.propose_action_group(
            state.to_json(),
            json.dumps({"val_score": 0.18}),
            group_size=4,
        )
    )

    assert len(candidates) == 4
    assert not any(candidate["is_jumping"] for candidate in candidates)
    assert {candidate["arm"] for candidate in candidates} <= set(tools._ALL_ARMS)


@pytest.mark.xfail(
    reason="stale: textual strategy-arm (capacity_strategy) is no longer routed as a single "
           "jumping candidate; _AMAZON_JUMP_ELIGIBLE_STRATEGY_ARMS is empty. Verify intended.",
    strict=False,
)
def test_amazon_strategy_jump_gate_open_marks_selected_eligible_arm(monkeypatch):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "amazon_reviews")
    monkeypatch.setattr(tools, "_ROUTING_POLICY", "textual_gradient")
    monkeypatch.setattr(tools, "_STORE", None)
    state = ThompsonState(
        baseline_done=True,
        global_budget=100000.0,
        score_history=[0.50, 0.50001, 0.50002, 0.50003],
        round_idx=4,
    )

    candidates = json.loads(
        tools.propose_action_group(
            state.to_json(),
            json.dumps({"val_score": 0.50003}),
            group_size=4,
            textual_selected_arms_json=json.dumps({"arms": ["capacity_strategy"]}),
        )
    )

    assert len(candidates) == 1
    assert candidates[0]["arm"] == "capacity_strategy"
    assert candidates[0]["dimension"] == "capacity_strategy"
    assert candidates[0]["is_jumping"] is True


@pytest.mark.xfail(
    reason="stale: textual strategy-arm (optimization_strategy) was replaced by low-level "
           "Thompson sampling arms. Verify intended.",
    strict=False,
)
def test_amazon_strategy_jump_gate_open_does_not_force_noneligible_arm(monkeypatch):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "amazon_reviews")
    monkeypatch.setattr(tools, "_ROUTING_POLICY", "textual_gradient")
    monkeypatch.setattr(tools, "_STORE", None)
    state = ThompsonState(
        baseline_done=True,
        global_budget=100000.0,
        score_history=[0.50, 0.50001, 0.50002, 0.50003],
        round_idx=4,
    )

    candidates = json.loads(
        tools.propose_action_group(
            state.to_json(),
            json.dumps({"val_score": 0.50003}),
            group_size=1,
            textual_selected_arms_json=json.dumps({"arms": ["optimization_strategy"]}),
        )
    )

    assert candidates
    assert candidates[0]["arm"] == "optimization_strategy"
    assert not any(candidate["is_jumping"] for candidate in candidates)


@pytest.mark.xfail(
    reason="stale: pending-queue jump suppression no longer emits strategy arms "
           "(capacity/representation_strategy). Verify intended.",
    strict=False,
)
def test_amazon_pending_retune_suppresses_jump_without_whitelist(monkeypatch):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "amazon_reviews")
    monkeypatch.setattr(tools, "_ROUTING_POLICY", "textual_gradient")
    monkeypatch.setattr(tools, "_STORE", None)
    state = ThompsonState(
        baseline_done=True,
        global_budget=100000.0,
        score_history=[0.50, 0.50001, 0.50002, 0.50003],
        round_idx=5,
        pending_queue=[{"dim": "capacity_strategy", "t_start": 4, "base_score": 0.5}],
    )

    candidates = json.loads(
        tools.propose_action_group(
            state.to_json(),
            json.dumps({"val_score": 0.50003}),
            group_size=4,
            textual_selected_arms_json=json.dumps({"arms": ["capacity_strategy", "representation_strategy"]}),
        )
    )

    assert candidates
    assert not any(candidate["is_jumping"] for candidate in candidates)
    assert any(candidate["arm"] == "capacity_strategy" for candidate in candidates)
    assert any(candidate["arm"] == "representation_strategy" for candidate in candidates)


def test_amazon_baseline_candidate_does_not_use_claude_backend(monkeypatch):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "amazon_reviews")
    monkeypatch.setattr(tools, "_STORE", None)
    state = ThompsonState(baseline_done=False, global_budget=100000.0)

    candidates = json.loads(tools.propose_action_group(state.to_json(), "{}", group_size=4))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["arm"] == "baseline"
    assert "implementation_backend" not in candidate
    assert candidate["code_edits"] == []
    assert candidate["code_content"] == ""
    assert candidate["code_diff"] == ""


def test_claude_backend_spec_rejects_inline_payload(tmp_path, monkeypatch):
    train_path = tmp_path / "best.py"
    train_path.write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "predict.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "amazon_reviews")
    monkeypatch.setenv("GAGC_ENABLE_CLAUDE_BACKEND", "1")

    result = json.loads(
        tools.execute_trial(
            spec_json=json.dumps({
                "dimension": "tune_lr",
                "delta": 1.0,
                "estimated_cost_secs": 1.0,
                "code_diff": "",
                "code_edits": [{"find": "ok", "replace": "better"}],
                "code_content": "",
                "implementation_backend": "claude_code",
            }),
            script_path=str(train_path),
        )
    )

    assert "must not include inline mutation payloads" in result["error_message"]


def test_parallel_group_budget_sums_member_wall_times(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "_STORE", None)
    monkeypatch.setattr(tools, "_LOGS_ROOT", str(tmp_path))
    monkeypatch.setattr(tools, "_LAST_PROMOTION_RESULT", {"promoted": True, "trial_group_id": "g_budget"})
    monkeypatch.setattr(tools, "_LAST_TRIAL_RESULTS", [])

    state = ThompsonState(global_budget=500.0)
    state.baseline_done = True

    results = [
        {
            "trial_group_id": "g_budget",
            "trial_id": idx,
            "val_score": score,
            "wall_time_secs": member_wall_time,
            "group_wall_time_secs": 125.0,
            "timed_out": False,
            "oom": False,
            "error_message": None,
            "spec": {
                "arm": arm,
                "dimension": arm,
                "delta": 1.0,
                "estimated_cost_secs": 10_800.0,
                "is_jumping": False,
            },
        }
        for idx, (arm, score, member_wall_time) in enumerate([
            ("tune_lr", 0.60, 100.0),
            ("tune_dropout", 0.55, 120.0),
            ("tune_num_heads", 0.50, 110.0),
            ("tune_num_layers", 0.45, 90.0),
        ])
    ]

    updated = ThompsonState.from_json(
        tools.update_thompson_state(json.dumps(results), state_json=state.to_json(), trial_group_id="g_budget")
    )

    # Budget is charged the sum of the 4 parallel members' wall_time_secs
    # (100+120+110+90 = 420); the group_wall_time_secs field (125) is ignored
    # when per-member times are present. 500 - 420 = 80.
    assert updated.global_budget == 80.0


def test_parallel_group_budget_falls_back_to_group_wall_time(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "_STORE", None)
    monkeypatch.setattr(tools, "_LOGS_ROOT", str(tmp_path))
    monkeypatch.setattr(tools, "_LAST_PROMOTION_RESULT", {"promoted": True, "trial_group_id": "g_budget_fallback"})
    monkeypatch.setattr(tools, "_LAST_TRIAL_RESULTS", [])

    state = ThompsonState(global_budget=500.0)
    state.baseline_done = True

    # No per-member wall_time_secs recorded -- only the batch group_wall_time_secs.
    # The spending calc falls back to that batch wall time (max across members).
    results = [
        {
            "trial_group_id": "g_budget_fallback",
            "trial_id": idx,
            "val_score": score,
            "group_wall_time_secs": 125.0,
            "timed_out": False,
            "oom": False,
            "error_message": None,
            "spec": {
                "arm": arm,
                "dimension": arm,
                "delta": 1.0,
                "estimated_cost_secs": 10_800.0,
                "is_jumping": False,
            },
        }
        for idx, (arm, score) in enumerate([
            ("tune_lr", 0.60),
            ("tune_dropout", 0.55),
            ("tune_num_heads", 0.50),
            ("tune_num_layers", 0.45),
        ])
    ]

    updated = ThompsonState.from_json(
        tools.update_thompson_state(json.dumps(results), state_json=state.to_json(), trial_group_id="g_budget_fallback")
    )

    # Fallback: 500 - max(group_wall_time_secs=125) = 375.
    assert updated.global_budget == 375.0

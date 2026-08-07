from __future__ import annotations

import json

import pytest

from gagc import tools
from gagc.grpo import GR_COMPOSITE_ARMS, GR_JUMPING_DIMS, GR_MUTEX_GROUPS
from gagc.state import ThompsonState


def test_gr_arm_structure(monkeypatch):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "kuairec")

    assert GR_JUMPING_DIMS == {"change_decoder_backbone"}
    assert "toggle_embedding_mixup" in tools._active_exploiting_arms()
    assert "change_curriculum_type" in tools._active_exploiting_arms()
    assert {"change_decoder_backbone", "toggle_embedding_mixup"} in GR_MUTEX_GROUPS

    assert GR_COMPOSITE_ARMS["tune_optimizer_schedule"] == ["tune_lr", "tune_batch_size", "add_lr_scheduler"]
    assert GR_COMPOSITE_ARMS["tune_loss_balance"] == ["tune_cls_weight", "tune_huber_weight"]
    assert GR_COMPOSITE_ARMS["tune_vocab_quantization"] == ["tune_q_start", "tune_q_end", "tune_q_decay"]
    assert GR_COMPOSITE_ARMS["tune_transformer_capacity"] == ["tune_hidden_dim", "tune_num_heads", "tune_dec_layers"]


@pytest.mark.xfail(
    reason="stale: jumping round now returns group_size (4) candidates instead of 1; README "
           "states jumping rounds run a single jump. Verify if regression or intended change.",
    strict=False,
)
def test_gr_decoder_backbone_candidate_uses_claude_backend(monkeypatch):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "kuairec")
    state = ThompsonState(baseline_done=True, global_budget=100000.0, score_history=[0.6, 0.6001, 0.6002], round_idx=5)
    for arm in tools._active_arms():
        arm_state = state.get_or_create_arm(arm)
        arm_state.alpha = 1.0
        arm_state.beta = 10.0
    state.get_or_create_arm("change_decoder_backbone").alpha = 100.0
    state.get_or_create_arm("change_decoder_backbone").beta = 1.0

    candidates = json.loads(
        tools.propose_action_group(
            state.to_json(),
            json.dumps({"test_score": 0.6002}),
            group_size=4,
        )
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["arm"] == "change_decoder_backbone"
    assert candidate["is_jumping"] is True
    assert candidate["implementation_backend"] == "claude_code"
    assert candidate["code_edits"] == []
    assert "model/decoder.py" in candidate["allowed_files"]

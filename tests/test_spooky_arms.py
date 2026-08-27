from __future__ import annotations

from gagc import tools
from gagc.grpo import SPOOKY_COMPOSITE_ARMS, SPOOKY_JUMPING_DIMS, SPOOKY_MUTEX_GROUPS


def test_spooky_arm_structure(monkeypatch):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "spooky_author")

    assert SPOOKY_JUMPING_DIMS == {"change_architecture"}
    assert "change_architecture" in tools._active_jumping_dims()
    assert "change_architecture" not in tools._active_exploiting_arms()
    assert "tune_weight_decay" in tools._active_exploiting_arms()
    assert {"change_architecture", "tune_hidden_dim", "tune_dropout"} in SPOOKY_MUTEX_GROUPS

    assert SPOOKY_COMPOSITE_ARMS["tune_optimizer_schedule"] == ["tune_lr", "tune_batch_size", "add_lr_scheduler"]
    assert SPOOKY_COMPOSITE_ARMS["tune_vectorizer"] == ["tune_ngram_range", "tune_max_features"]
    assert SPOOKY_COMPOSITE_ARMS["tune_capacity_regularization"] == ["tune_hidden_dim", "tune_dropout"]

    active_arms = tools._active_arms()
    assert set(active_arms) == set(SPOOKY_COMPOSITE_ARMS.keys()) | {"tune_weight_decay", "change_architecture"}

    # Every active arm (composite or standalone) must have a dimension hint,
    # except composite arms whose hint comes from their own entry.
    hints = tools._active_dimension_hints()
    for arm in active_arms:
        assert arm in hints, f"missing dimension hint for arm {arm!r}"


def test_spooky_claude_backend_attached_only_to_architecture_arm(monkeypatch):
    monkeypatch.setattr(tools, "_BENCHMARK_MODE", "spooky_author")

    fields = tools._claude_code_backend_fields_for_arm("change_architecture")
    assert fields["implementation_backend"] == "claude_code"
    assert "train.py" in fields["allowed_files"]
    assert "predict.py" in fields["allowed_files"]

    assert tools._claude_code_backend_fields_for_arm("tune_lr") == {}
    assert tools._claude_code_backend_fields_for_arm("tune_optimizer_schedule") == {}

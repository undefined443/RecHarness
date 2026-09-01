"""Unit tests for gagc.diversity_reactor (the empty-payload config generator)."""

import textwrap

from gagc.diversity_reactor import _extract_yaml, _validate, generate_config_mutation
from gagc.schemas import MutationSpec

_BASE_CONFIG = textwrap.dedent(
    """\
    data:
      sample_path: /data/sample
      vec_path: /data/vec
    exposure_probs: [1.0, 1.0, 0.4]
    vecsim_pass_rate_threshold: 0.6
    baseline_metrics:
      std1_top4_mean: 6.4381
    scatter:
      multiWinNum: 2
      dppConfigList:
        - slidingWindowSize: 30
          fstDefConfigMap:
            FIRST_POS: {dwPower: 1.0}
            DEFAULT: {dwPower: 1.0}
        - slidingWindowSize: 10
          fstDefConfigMap:
            FIRST_POS: {dwPower: 1.0}
            DEFAULT: {dwPower: 1.0}
    """
)


class _FakeModel:
    """Returns a scripted response per invoke() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        text = self._responses.pop(0)
        return type("_Msg", (), {"content": text})()


def _spec():
    return MutationSpec(
        dimension="tune_dwPower",
        delta=1.0,
        estimated_cost_secs=60.0,
        code_diff="",
        arm="tune_diversity_strength",
        code_hint="Raise dwPower to strengthen diversification.",
    )


def _with_dwpower(value):
    return _BASE_CONFIG.replace("dwPower: 1.0", f"dwPower: {value}")


def test_extract_yaml_strips_fences():
    raw = "Here you go:\n```yaml\nfoo: 1\nbar: 2\n```\nthanks"
    assert _extract_yaml(raw) == "foo: 1\nbar: 2"


def test_validate_accepts_real_scatter_change():
    import yaml

    assert _validate(yaml.safe_load(_BASE_CONFIG), _with_dwpower(2.0)) is None


def test_validate_rejects_noop():
    import yaml

    msg = _validate(yaml.safe_load(_BASE_CONFIG), _BASE_CONFIG)
    assert msg is not None and "no change" in msg


def test_validate_rejects_frozen_section_edit():
    import yaml

    tampered = _with_dwpower(2.0).replace("/data/sample", "/data/other")
    msg = _validate(yaml.safe_load(_BASE_CONFIG), tampered)
    assert msg is not None and "data" in msg


def test_validate_rejects_bad_yaml():
    import yaml

    msg = _validate(yaml.safe_load(_BASE_CONFIG), "scatter: [unclosed")
    assert msg is not None and "YAML" in msg


def test_generate_returns_first_valid_candidate():
    model = _FakeModel([f"```yaml\n{_with_dwpower(2.0)}```"])
    out = generate_config_mutation(model, _BASE_CONFIG, _spec(), max_attempts=3)
    assert out is not None and "dwPower: 2.0" in out
    assert model.calls == 1


def test_generate_retries_then_succeeds_with_feedback():
    model = _FakeModel(["not: valid: yaml:", _with_dwpower(3.0)])
    out = generate_config_mutation(model, _BASE_CONFIG, _spec(), max_attempts=3)
    assert out is not None and "dwPower: 3.0" in out
    assert model.calls == 2


def test_generate_gives_up_after_max_attempts():
    model = _FakeModel([_BASE_CONFIG, _BASE_CONFIG, _BASE_CONFIG])
    out = generate_config_mutation(model, _BASE_CONFIG, _spec(), max_attempts=3)
    assert out is None
    assert model.calls == 3

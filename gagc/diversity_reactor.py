"""LLM-driven config.yaml mutation for diversity_v3 trials.

diversity_v3 has no claude_code implementation backend, and its orchestrator
prompt keeps mutation payloads out of tool arguments, so a trial spec can reach
execution with an empty ``code_edits`` / ``code_content`` / ``code_diff``. Rather
than silently evaluating the unmodified (baseline) config, this module asks the
run's own LLM to produce the concrete mutated config.yaml and validates it (YAML
syntax, structural shape, frozen sections, and that ``scatter:`` actually
changed) before the trial proceeds.

Structurally a lighter sibling of ``gagc.claude_code_backend``: an
attempt/feedback loop, but one text generation per attempt with no subprocess
and no training run.
"""

from __future__ import annotations

import json
from typing import Any

import yaml

from gagc.schemas import MutationSpec

# config.yaml sections prepare.py reads and the search must never touch.
_FROZEN_KEYS = ("data", "exposure_probs", "baseline_metrics", "vecsim_pass_rate_threshold")


def generate_config_mutation(
    model: Any,
    current_config: str,
    spec: MutationSpec,
    max_attempts: int = 3,
) -> str | None:
    """Return a validated mutated config.yaml, or None if every attempt fails.

    Args:
        model: The run's LangChain chat model (``.invoke([...]) -> AIMessage``).
        current_config: Full text of the trial's incumbent config.yaml.
        spec: The trial mutation spec. ``dimension`` / ``delta`` plus ``extra``'s
            ``arm``, ``code_hint``, ``hypothesis``, and ``gagc_memory_context``
            steer the edit.
        max_attempts: Generation attempts before giving up.

    Returns:
        The mutated config.yaml text, or None.
    """
    original = yaml.safe_load(current_config)
    feedback: str | None = None
    for attempt in range(1, max_attempts + 1):
        prompt = _build_prompt(current_config, spec, feedback, attempt, max_attempts)
        raw = model.invoke([{"role": "user", "content": prompt}]).content
        candidate = _extract_yaml(raw if isinstance(raw, str) else str(raw))
        feedback = _validate(original, candidate)
        if feedback is None:
            return candidate
    return None


def _build_prompt(
    current_config: str,
    spec: MutationSpec,
    feedback: str | None,
    attempt: int,
    max_attempts: int,
) -> str:
    """Assemble the single-turn generation prompt for one attempt."""
    arm = spec.extra.get("arm") or spec.dimension
    hint = spec.extra.get("code_hint") or ""
    hypothesis = spec.extra.get("hypothesis") or ""
    memory = spec.extra.get("gagc_memory_context") or {}
    task = spec.extra.get("implementation_prompt") or (
        "Apply ONE hyperparameter mutation to config.yaml's scatter: section."
    )
    lines = [
        "You tune the DPP re-ranking config for the diversity_v3 benchmark.",
        task,
        "Return the WHOLE config.yaml file.",
        "",
        f"Arm: {arm}",
        f"Dimension: {spec.dimension}",
        f"Delta direction: {spec.delta:+g} (positive = increase, negative = decrease)",
    ]
    if hint:
        lines.append(f"Edit hint: {hint}")
    if hypothesis:
        lines.append(f"Hypothesis: {hypothesis}")
    if memory:
        lines.append(f"Run memory: {json.dumps(memory)[:2000]}")
    lines += [
        "",
        "Rules:",
        "- Change ONLY keys under `scatter:`. Keep `data:`, `exposure_probs:`,",
        "  `baseline_metrics:`, and `vecsim_pass_rate_threshold:` exactly as given.",
        "- `scatter.dppConfigList` has 2 windows, each with FIRST_POS and DEFAULT",
        "  sub-blocks (4 blocks total). Apply the change consistently across every",
        "  block the hypothesis implies -- normally all 4.",
        "- Keep it valid YAML with the same overall structure.",
        "- Output the COMPLETE config.yaml and nothing else: no prose, no code fences.",
    ]
    if feedback:
        lines += [
            "",
            f"Attempt {attempt}/{max_attempts}. The previous attempt was rejected:",
            feedback,
        ]
    lines += ["", "Current config.yaml:", current_config]
    return "\n".join(lines)


def _extract_yaml(text: str) -> str:
    """Strip Markdown code fences / stray prose around a YAML document."""
    text = text.strip()
    if "```" not in text:
        return text
    fenced = [block for block in text.split("```")[1::2]]
    best = max(fenced, key=len, default=text).strip()
    first, _, rest = best.partition("\n")
    if first.strip().isalpha():  # drop a leading language tag like "yaml"
        best = rest
    return best.strip()


def _validate(original: dict, candidate: str) -> str | None:
    """Return rejection feedback for a candidate config, or None if it passes."""
    try:
        parsed = yaml.safe_load(candidate)
    except yaml.YAMLError as exc:
        return f"not valid YAML: {exc}"
    if not isinstance(parsed, dict):
        return "top level is not a YAML mapping"

    for key in _FROZEN_KEYS:
        if parsed.get(key) != original.get(key):
            return f"`{key}:` was modified; it must stay exactly as in the current config"

    scatter = parsed.get("scatter")
    if not isinstance(scatter, dict):
        return "`scatter:` section is missing or not a mapping"

    orig_windows = original.get("scatter", {}).get("dppConfigList") or []
    windows = scatter.get("dppConfigList")
    if not isinstance(windows, list) or len(windows) != len(orig_windows):
        return f"`scatter.dppConfigList` must be a list of {len(orig_windows)} windows"
    for i, win in enumerate(windows):
        block = win.get("fstDefConfigMap") if isinstance(win, dict) else None
        if not isinstance(block, dict) or "FIRST_POS" not in block or "DEFAULT" not in block:
            return f"window {i} must keep fstDefConfigMap with FIRST_POS and DEFAULT sub-blocks"

    if scatter == original.get("scatter"):
        return "no change was made to `scatter:`; apply the requested mutation"
    return None

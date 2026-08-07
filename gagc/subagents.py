from __future__ import annotations

from gagc.prompts import TRIAL_SYSTEM_PROMPT
from gagc.tools import execute_trial

# Trial subagent definition for use with create_deep_agent(subagents=[...]).
# Each trial in the GRPO group runs as an isolated subagent with a structured
# TrialResult response so the orchestrator receives clean JSON.
#
# response_format is specified as a plain dict schema here to keep the module
# free of a hard deepagents import at definition time; the agent.py wiring
# passes it through to create_deep_agent unchanged.

TRIAL_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "spec": {
            "type": "object",
            "properties": {
                "dimension": {"type": "string"},
                "delta": {"type": "number"},
                "estimated_cost_secs": {"type": "number"},
                "code_diff": {"type": "string"},
            },
            "required": ["dimension", "delta", "estimated_cost_secs", "code_diff"],
        },
        "wall_time_secs": {"type": "number"},
        "timed_out": {"type": "boolean"},
        "oom": {"type": "boolean"},
        "val_score": {"type": "number"},
        "convergence_trace": {"type": "array", "items": {"type": "number"}},
        "error_message": {"type": ["string", "null"]},
    },
    "required": [
        "spec",
        "wall_time_secs",
        "timed_out",
        "oom",
        "val_score",
        "convergence_trace",
        "error_message",
    ],
}

trial_subagent = {
    "name": "mle-trial",
    "description": (
        "Executes one candidate code mutation on the training script and returns "
        "a scored TrialResult. Use this for each member of the GRPO candidate group. "
        "Pass slot_id (0-7) in the task input to assign a dedicated GPU and CPU set."
    ),
    "system_prompt": TRIAL_SYSTEM_PROMPT,
    "tools": [execute_trial],
    "response_format": TRIAL_RESULT_SCHEMA,
}

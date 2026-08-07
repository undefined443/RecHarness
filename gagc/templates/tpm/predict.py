"""Prediction compatibility module for the KuaiRec TPM cold start.

KuaiRec benchmark trials execute ``best.py`` directly and parse metrics from
stdout. This module keeps the standard ``train.py`` + ``predict.py`` template
layout used by RecHarness workspaces.
"""
from __future__ import annotations


def predict(*_args, **_kwargs):
    """KuaiRec templates do not support per-user candidate scoring."""
    raise NotImplementedError(
        "TPM KuaiRec evaluation is performed by running train.py/best.py; "
        "candidate-level predict() is only used by Amazon Reviews templates."
    )

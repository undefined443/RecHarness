"""Popular — predict function.

Scores candidates by global interaction frequency.
Falls back to random noise when model.pkl is absent (untrained state).
"""
from __future__ import annotations

import os
import pickle
import random

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
_popularity: dict[int, int] | None = None


def _load():
    global _popularity
    if _popularity is None:
        if os.path.isfile(_MODEL_PATH):
            with open(_MODEL_PATH, "rb") as f:
                _popularity = pickle.load(f)
        else:
            _popularity = {}


def predict(user_id: int, history: list[int], candidates: list[int]) -> list[float]:
    _load()
    if not _popularity:
        # Model not trained yet — return small random scores so ranking is not trivially biased
        return [random.random() for _ in candidates]
    return [float(_popularity.get(c, 0)) for c in candidates]

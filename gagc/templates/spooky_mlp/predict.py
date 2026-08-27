"""Spooky Author Identification — predict function.

Uses importlib to load train.py by absolute path, avoiding sys.modules
collision when multiple templates are evaluated in the same process.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import torch

_HERE       = os.path.dirname(os.path.abspath(__file__))
_CKPT_PATH  = os.path.join(_HERE, "checkpoint.pt")

_model      = None
_vectorizer = None
_device     = None


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        f"spooky_train_{os.getpid()}", os.path.join(_HERE, "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load():
    global _model, _vectorizer, _device
    if _model is not None:
        return

    train_mod = _load_train_module()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(_CKPT_PATH, map_location=_device, weights_only=False)

    _vectorizer = checkpoint["vectorizer"]
    model = train_mod.create_model(checkpoint["input_dim"], _device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _model = model


def predict(texts: list[str]) -> np.ndarray:
    """Return (len(texts), 3) probabilities, columns ordered EAP, HPL, MWS."""
    _load()
    X = _vectorizer.transform(texts)
    X_tensor = torch.from_numpy(X.toarray()).float().to(_device)
    with torch.no_grad():
        logits = _model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    # Guard against float32 rounding so rows sum to exactly 1 for the protocol check.
    return probs / probs.sum(axis=1, keepdims=True)

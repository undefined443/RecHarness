"""BERT4Rec — predict function.

At inference time the last position of the history is replaced with [MASK]
and the model output at that position scores the candidates.

Uses importlib to load train.py by absolute path, avoiding sys.modules
collision when multiple templates are evaluated in the same process.
"""
from __future__ import annotations

import importlib.util
import os
import pickle

import torch

_HERE       = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_HERE, "model.pt")
_META_PATH  = os.path.join(_HERE, "meta.pkl")

_model  = None
_meta   = None
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"bert4rec_train_{os.getpid()}", os.path.join(_HERE, "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BERT4RecModel


def _load():
    global _model, _meta, _device
    if _model is not None:
        return

    with open(_META_PATH, "rb") as f:
        _meta = pickle.load(f)

    BERT4RecModel = _load_model_class()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = BERT4RecModel(
        itemnum=_meta["itemnum"],
        mask_token=_meta["mask_token"],
        hidden=_meta["hidden"],
        maxlen=_meta["maxlen"],
        num_blocks=_meta["num_blocks"],
        num_heads=_meta["num_heads"],
        dropout=0.0,
    ).to(_device)
    m.load_state_dict(torch.load(_MODEL_PATH, map_location=_device, weights_only=True))
    m.eval()
    _model = m


def predict(user_id: int, history: list[int], candidates: list[int]) -> list[float]:
    _load()
    maxlen     = _meta["maxlen"]
    mask_token = _meta["mask_token"]

    hist   = history[-(maxlen - 1):]
    seq    = hist + [mask_token]
    padded = [0] * (maxlen - len(seq)) + seq
    inp    = torch.tensor([padded], dtype=torch.long, device=_device)

    with torch.no_grad():
        out      = _model(inp)
        mask_rep = out[0, -1]
        cand_t   = torch.tensor(candidates, dtype=torch.long, device=_device)
        cand_emb = _model.item_emb(cand_t)
        scores   = (cand_emb @ mask_rep).cpu().tolist()

    return scores

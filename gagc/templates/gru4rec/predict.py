"""GRU4Rec — predict function.

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
        f"gru4rec_train_{os.getpid()}", os.path.join(_HERE, "train.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GRU4RecModel


def _load():
    global _model, _meta, _device
    if _model is not None:
        return

    with open(_META_PATH, "rb") as f:
        _meta = pickle.load(f)

    GRU4RecModel = _load_model_class()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = GRU4RecModel(
        itemnum=_meta["itemnum"],
        hidden=_meta["hidden"],
        num_layers=_meta["num_layers"],
        dropout=0.0,
    ).to(_device)
    m.load_state_dict(torch.load(_MODEL_PATH, map_location=_device, weights_only=True))
    m.eval()
    _model = m


def predict(user_id: int, history: list[int], candidates: list[int]) -> list[float]:
    _load()
    maxlen = _meta["maxlen"]
    hist   = history[-maxlen:]
    padded = [0] * (maxlen - len(hist)) + hist
    seq    = torch.tensor([padded], dtype=torch.long, device=_device)

    with torch.no_grad():
        user_rep = _model(seq).squeeze(0)
        cand_t   = torch.tensor(candidates, dtype=torch.long, device=_device)
        cand_emb = _model.item_emb(cand_t)
        scores   = (cand_emb @ user_rep).cpu().tolist()

    return scores

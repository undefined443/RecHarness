"""Per-dataset HSTU predict — routes by dataset_name kwarg."""
from __future__ import annotations

import importlib.util
import os
import pickle

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]
DATASETS = [
    part.strip()
    for part in os.environ.get("GAGC_DATASETS", ",".join(DEFAULT_DATASETS)).split(",")
    if part.strip()
]

_models: dict = {}
_metas: dict = {}
_device = None


def _load_model_class():
    spec = importlib.util.spec_from_file_location(
        f"hstu_perdataset_train_{os.getpid()}", os.path.join(_HERE, "train.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HSTURec


def _ensure_loaded() -> None:
    global _device
    if _models:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = _load_model_class()
    working_dir = os.path.join(_HERE, "working")
    for dataset_name in DATASETS:
        meta_path = os.path.join(working_dir, f"{dataset_name}_meta.pkl")
        ckpt_path = os.path.join(working_dir, f"{dataset_name}_model.pt")
        if not os.path.isfile(meta_path) or not os.path.isfile(ckpt_path):
            continue
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        _metas[dataset_name] = meta
        model = model_cls(
            itemnum=meta["itemnum"],
            hidden_units=meta["hidden_units"],
            num_blocks=meta["num_blocks"],
            num_heads=meta["num_heads"],
            dropout_rate=0.0,
            maxlen=meta["maxlen"],
        ).to(_device)
        model.load_state_dict(torch.load(ckpt_path, map_location=_device, weights_only=True))
        model.eval()
        _models[dataset_name] = model


def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    _ensure_loaded()
    if dataset_name and dataset_name in _models:
        model = _models[dataset_name]
        meta = _metas[dataset_name]
    elif _models:
        first_dataset = next(iter(_models))
        model = _models[first_dataset]
        meta = _metas[first_dataset]
    else:
        return [0.0] * len(candidates)

    maxlen = meta["maxlen"]
    seq_arr = [0] * maxlen
    for idx, item in enumerate(reversed(history[-maxlen:])):
        seq_arr[maxlen - 1 - idx] = item
    seq_t = torch.tensor([seq_arr], dtype=torch.long, device=_device)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=_device)
    with torch.no_grad():
        out = model(seq_t)
        user_emb = out[0, -1]
        scores = torch.matmul(model.item_emb(cand_t), user_emb).cpu().tolist()
    return scores

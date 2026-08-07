"""GRU4Rec — Session-based recommendation with GRU.

Reference: Hidasi et al., ICLR 2016.

Hyper-parameters:
  HIDDEN_UNITS : GRU hidden size
  NUM_LAYERS   : number of GRU layers
  DROPOUT_RATE : dropout between GRU layers
  LR           : Adam learning rate
  BATCH_SIZE   : training batch size
  NUM_EPOCHS   : total training epochs
  MAXLEN       : sequence truncation length
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ── Hyper-parameters ──────────────────────────────────────────────────
MAXLEN       = int(os.environ.get("GRU4REC_MAXLEN",     "50"))
HIDDEN_UNITS = int(os.environ.get("GRU4REC_HIDDEN",     "128"))
NUM_LAYERS   = int(os.environ.get("GRU4REC_LAYERS",     "1"))
DROPOUT_RATE = float(os.environ.get("GRU4REC_DROPOUT",  "0.1"))
LR           = float(os.environ.get("GRU4REC_LR",       "1e-3"))
BATCH_SIZE   = int(os.environ.get("GRU4REC_BATCH",      "256"))
NUM_EPOCHS   = int(os.environ.get("GRU4REC_EPOCHS",     "20"))

DATA_DIR  = os.environ.get("GAGC_DATA_DIR", "./input/trainval")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "model.pt")
META_OUT  = os.path.join(os.path.dirname(__file__), "meta.pkl")

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


def _read_split(path: str) -> dict[int, list[int]]:
    from collections import defaultdict
    d: dict[int, list[int]] = defaultdict(list)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                d[int(parts[0])].append(int(parts[1]))
    return dict(d)


def _load_itemnum(data_dir: str) -> int:
    """Read itemnum from dataset_stats.json (covers all splits including test candidates)."""
    import json
    stats_path = os.path.join(data_dir, "dataset_stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        return max(v["itemnum"] for v in stats.values())
    itemnum = 0
    for ds in DATASETS:
        path = os.path.join(data_dir, f"{ds}_train.txt")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    itemnum = max(itemnum, int(parts[1]))
    return itemnum


def _load_all(data_dir: str):
    user_train: dict[int, list[int]] = {}
    for ds in DATASETS:
        train = _read_split(os.path.join(data_dir, f"{ds}_train.txt"))
        for u, seq in train.items():
            user_train[u] = user_train.get(u, []) + seq
    itemnum = _load_itemnum(data_dir)
    return user_train, itemnum


class SeqDataset(Dataset):
    def __init__(self, user_train: dict[int, list[int]], maxlen: int):
        self.sequences = []
        for seq in user_train.values():
            for i in range(1, len(seq)):
                hist   = seq[max(0, i - maxlen): i]
                target = seq[i]
                padded = [0] * (maxlen - len(hist)) + hist
                self.sequences.append((padded, target))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        hist, tgt = self.sequences[idx]
        return torch.tensor(hist, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


class GRU4RecModel(nn.Module):
    def __init__(self, itemnum: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.item_emb = nn.Embedding(itemnum + 1, hidden, padding_idx=0)
        self.gru = nn.GRU(
            hidden, hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_drop = nn.Dropout(dropout)

    def encode(self, seqs):
        """Return the hidden state at the last non-padding timestep."""
        emb  = self.item_emb(seqs)          # (B, L, H)
        out, _ = self.gru(emb)              # (B, L, H)
        lens = (seqs != 0).sum(dim=1) - 1  # last real index
        lens = lens.clamp(min=0)
        idx  = lens.view(-1, 1, 1).expand(-1, 1, out.size(2))
        h    = out.gather(1, idx).squeeze(1)   # (B, H)
        return self.out_drop(h)

    def forward(self, seqs):
        return self.encode(seqs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_train, itemnum = _load_all(DATA_DIR)
    print(f"users={len(user_train)}, items={itemnum}", flush=True)

    dataset = SeqDataset(user_train, MAXLEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = GRU4RecModel(itemnum, HIDDEN_UNITS, NUM_LAYERS, DROPOUT_RATE).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for seqs, targets in loader:
            seqs, targets = seqs.to(device), targets.to(device)
            user_rep = model(seqs)                     # (B, H)
            logits   = user_rep @ model.item_emb.weight.T  # (B, I+1)
            loss = loss_fn(logits, targets)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()

        avg = total_loss / max(len(loader), 1)
        print(f"epoch {epoch:3d}  loss={avg:.4f}  val_score: 0.0", flush=True)

    torch.save(model.state_dict(), MODEL_OUT)
    with open(META_OUT, "wb") as f:
        pickle.dump({"itemnum": itemnum, "maxlen": MAXLEN, "hidden": HIDDEN_UNITS,
                     "num_layers": NUM_LAYERS, "dropout": DROPOUT_RATE}, f)
    print(f"Saved model → {MODEL_OUT}", flush=True)


if __name__ == "__main__":
    main()

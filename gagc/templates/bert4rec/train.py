"""BERT4Rec — Bidirectional Encoder Representations for Sequential Recommendation.

Reference: Sun et al., CIKM 2019.

Trains with cloze-task masking (MASK token); predicts by placing [MASK] at
the end of the history sequence and scoring candidates against the mask output.

Hyper-parameters:
  MAXLEN       : max history length
  HIDDEN_UNITS : hidden / embedding dimension
  NUM_BLOCKS   : number of Transformer encoder blocks
  NUM_HEADS    : attention heads
  DROPOUT_RATE : dropout
  MASK_PROB    : proportion of items masked during training
  LR           : Adam learning rate
  BATCH_SIZE   : training batch size
  NUM_EPOCHS   : total training epochs
"""
from __future__ import annotations

import os
import pickle
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ── Hyper-parameters ──────────────────────────────────────────────────
MAXLEN       = int(os.environ.get("BERT4REC_MAXLEN",     "50"))
HIDDEN_UNITS = int(os.environ.get("BERT4REC_HIDDEN",     "64"))
NUM_BLOCKS   = int(os.environ.get("BERT4REC_BLOCKS",     "2"))
NUM_HEADS    = int(os.environ.get("BERT4REC_HEADS",      "2"))
DROPOUT_RATE = float(os.environ.get("BERT4REC_DROPOUT",  "0.2"))
MASK_PROB    = float(os.environ.get("BERT4REC_MASKPROB", "0.2"))
LR           = float(os.environ.get("BERT4REC_LR",       "1e-3"))
BATCH_SIZE   = int(os.environ.get("BERT4REC_BATCH",      "256"))
NUM_EPOCHS   = int(os.environ.get("BERT4REC_EPOCHS",     "20"))

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


class ClozeDataset(Dataset):
    """Each sample: padded sequence with random items masked → predict masked items."""

    def __init__(self, user_train: dict[int, list[int]], itemnum: int,
                 maxlen: int, mask_prob: float, mask_token: int):
        self.sequences = [seq[-maxlen:] for seq in user_train.values() if len(seq) >= 2]
        self.itemnum   = itemnum
        self.maxlen    = maxlen
        self.mask_prob = mask_prob
        self.mask_tok  = mask_token

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        masked, labels = [], []
        for item in seq:
            if random.random() < self.mask_prob:
                masked.append(self.mask_tok)
                labels.append(item)
            else:
                masked.append(item)
                labels.append(0)           # 0 = not masked → ignored in loss

        L = len(masked)
        pad = self.maxlen - L
        masked = [0] * pad + masked
        labels = [0] * pad + labels

        return (
            torch.tensor(masked, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


class BERT4RecModel(nn.Module):
    def __init__(self, itemnum: int, mask_token: int, hidden: int,
                 maxlen: int, num_blocks: int, num_heads: int, dropout: float):
        super().__init__()
        vocab_size = itemnum + 2   # 0 = pad, mask_token = itemnum+1
        self.item_emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos_emb  = nn.Embedding(maxlen, hidden)
        self.emb_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads,
            dim_feedforward=hidden * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_blocks)
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, seqs):
        B, L = seqs.shape
        pos = torch.arange(L, device=seqs.device).unsqueeze(0).expand(B, -1)
        x = self.item_emb(seqs) + self.pos_emb(pos)
        x = self.emb_drop(x)
        key_mask = (seqs == 0)
        x = self.encoder(x, src_key_padding_mask=key_mask)
        return self.out_norm(x)   # (B, L, H)


def main():
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_train, itemnum = _load_all(DATA_DIR)
    mask_token = itemnum + 1
    print(f"users={len(user_train)}, items={itemnum}, mask_token={mask_token}", flush=True)

    dataset = ClozeDataset(user_train, itemnum, MAXLEN, MASK_PROB, mask_token)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model   = BERT4RecModel(itemnum, mask_token, HIDDEN_UNITS, MAXLEN,
                            NUM_BLOCKS, NUM_HEADS, DROPOUT_RATE).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            out     = model(seqs)                                   # (B, L, H)
            logits  = out @ model.item_emb.weight.T                 # (B, L, V)
            loss    = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()

        avg = total_loss / max(len(loader), 1)
        print(f"epoch {epoch:3d}  loss={avg:.4f}  val_score: 0.0", flush=True)

    torch.save(model.state_dict(), MODEL_OUT)
    with open(META_OUT, "wb") as f:
        pickle.dump({"itemnum": itemnum, "mask_token": mask_token,
                     "maxlen": MAXLEN, "hidden": HIDDEN_UNITS,
                     "num_blocks": NUM_BLOCKS, "num_heads": NUM_HEADS,
                     "dropout": DROPOUT_RATE}, f)
    print(f"Saved model → {MODEL_OUT}", flush=True)


if __name__ == "__main__":
    main()

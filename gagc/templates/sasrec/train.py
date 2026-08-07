"""SASRec — Self-Attentive Sequential Recommendation.

Reference: Kang & McAuley, ICDM 2018.

Key hyper-parameters (all tunable by RecHarness):
  MAXLEN       : max history length
  HIDDEN_UNITS : embedding / attention dimension
  NUM_BLOCKS   : number of self-attention blocks
  NUM_HEADS    : number of attention heads
  DROPOUT_RATE : dropout on attention weights
  LR           : Adam learning rate
  BATCH_SIZE   : training batch size
  NUM_EPOCHS   : total training epochs
  L2_EMB       : L2 regularization on embeddings
"""
from __future__ import annotations

import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ── Hyper-parameters ──────────────────────────────────────────────────
MAXLEN       = int(os.environ.get("SASREC_MAXLEN",       "50"))
HIDDEN_UNITS = int(os.environ.get("SASREC_HIDDEN",       "50"))
NUM_BLOCKS   = int(os.environ.get("SASREC_BLOCKS",       "2"))
NUM_HEADS    = int(os.environ.get("SASREC_HEADS",        "1"))
DROPOUT_RATE = float(os.environ.get("SASREC_DROPOUT",   "0.5"))
LR           = float(os.environ.get("SASREC_LR",         "1e-3"))
BATCH_SIZE   = int(os.environ.get("SASREC_BATCH",        "128"))
NUM_EPOCHS   = int(os.environ.get("SASREC_EPOCHS",       "201"))
L2_EMB       = float(os.environ.get("SASREC_L2",         "0.0"))

DATA_DIR  = os.environ.get("GAGC_DATA_DIR",  "./input/trainval")
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

# ── Data loading ──────────────────────────────────────────────────────

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
    # Fallback: scan train files only (may undercount if test has unseen items)
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


# ── Negative Sampling Dataset ────────────────────────────────────────

class SASRecDataset(Dataset):
    """Generate (seq, pos, neg) triplets for BCE loss training."""

    def __init__(self, user_train: dict[int, list[int]], itemnum: int, maxlen: int):
        self.user_train = user_train
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.users = [u for u, seq in user_train.items() if len(seq) >= 2]

    def __len__(self):
        return len(self.users) * 5  # oversample for diversity

    def __getitem__(self, idx):
        u = self.users[idx % len(self.users)]
        seq_full = self.user_train[u]

        # Sample a random position in the sequence
        seq = np.zeros(self.maxlen, dtype=np.int64)
        pos = np.zeros(self.maxlen, dtype=np.int64)
        neg = np.zeros(self.maxlen, dtype=np.int64)

        nxt = seq_full[-1]
        idx_pos = self.maxlen - 1
        ts = set(seq_full)

        for item in reversed(seq_full[:-1]):
            seq[idx_pos] = item
            pos[idx_pos] = nxt
            # Negative sampling
            neg_item = random.randint(1, self.itemnum)
            while neg_item in ts:
                neg_item = random.randint(1, self.itemnum)
            neg[idx_pos] = neg_item
            nxt = item
            idx_pos -= 1
            if idx_pos == -1:
                break

        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


# ── Model ─────────────────────────────────────────────────────────────

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, H)
        out = self.conv1(x.transpose(-1, -2))  # (B, H, L)
        out = self.relu(self.dropout1(out))
        out = self.dropout2(self.conv2(out))
        out = out.transpose(-1, -2)  # (B, L, H)
        return out + x


class SASRecModel(nn.Module):
    def __init__(self, itemnum: int, hidden: int, maxlen: int,
                 num_blocks: int, num_heads: int, dropout: float):
        super().__init__()
        self.item_emb = nn.Embedding(itemnum + 1, hidden, padding_idx=0)
        self.pos_emb  = nn.Embedding(maxlen, hidden)
        self.emb_dropout = nn.Dropout(dropout)
        self.hidden = hidden

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers     = nn.ModuleList()
        self.forward_layernorms   = nn.ModuleList()
        self.forward_layers       = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=False)
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden, dropout))

        self.last_layernorm = nn.LayerNorm(hidden, eps=1e-8)

    def log2feats(self, log_seqs):
        """Convert log sequence to feature representations."""
        seqs = self.item_emb(log_seqs)
        seqs *= (self.hidden ** 0.5)  # Embedding scaling

        B, L = log_seqs.shape
        positions = torch.arange(L, device=log_seqs.device).unsqueeze(0).expand(B, -1)
        seqs += self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0)
        seqs = seqs * (~timeline_mask).unsqueeze(-1).float()

        # Causal mask
        tl = seqs.shape[1]
        attention_mask = torch.triu(torch.ones(tl, tl, device=log_seqs.device), diagonal=1).bool()

        for i in range(len(self.attention_layers)):
            seqs_t = seqs.transpose(0, 1)  # (L, B, H)
            Q = self.attention_layernorms[i](seqs_t)
            mha_out, _ = self.attention_layers[i](Q, seqs_t, seqs_t, attn_mask=attention_mask)
            seqs = (Q + mha_out).transpose(0, 1)  # (B, L, H)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs * (~timeline_mask).unsqueeze(-1).float()

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        """Forward pass for training with BCE loss."""
        log_feats = self.log2feats(log_seqs)  # (B, L, H)

        pos_embs = self.item_emb(pos_seqs)  # (B, L, H)
        neg_embs = self.item_emb(neg_seqs)  # (B, L, H)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)  # (B, L)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)  # (B, L)

        return pos_logits, neg_logits


# ── Training ──────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_train, itemnum = _load_all(DATA_DIR)
    print(f"users={len(user_train)}, items={itemnum}", flush=True)
    if not user_train:
        datasets = ",".join(DATASETS)
        raise RuntimeError(
            f"No training users loaded from {DATA_DIR} for GAGC_DATASETS={datasets}. "
            "Expected files like <dataset>_train.txt with 'user item' rows; "
            "rerun preprocessing with --force-preprocess if the split files are empty."
        )
    if itemnum <= 0:
        raise RuntimeError(f"No items loaded from {DATA_DIR}; check dataset_stats.json and split files.")

    dataset = SASRecDataset(user_train, itemnum, MAXLEN)
    # Use single worker to avoid multiprocessing issues with random sampling
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    model = SASRecModel(itemnum, HIDDEN_UNITS, MAXLEN, NUM_BLOCKS, NUM_HEADS, DROPOUT_RATE).to(device)

    # Xavier initialization
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98))
    bce_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        count = 0

        for seqs, pos, neg in loader:
            seqs = seqs.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_logits, neg_logits = model(seqs, pos, neg)

            # Only compute loss on non-padding positions
            indices = (pos != 0)
            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)

            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            # L2 regularization on embeddings
            if L2_EMB > 0:
                loss += L2_EMB * torch.norm(model.item_emb.weight)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            count += 1

        avg = total_loss / max(count, 1)
        print(f"epoch {epoch:3d}  loss={avg:.4f}  val_score: 0.0", flush=True)

    torch.save(model.state_dict(), MODEL_OUT)
    with open(META_OUT, "wb") as f:
        pickle.dump({
            "itemnum": itemnum,
            "maxlen": MAXLEN,
            "hidden": HIDDEN_UNITS,
            "num_blocks": NUM_BLOCKS,
            "num_heads": NUM_HEADS,
            "dropout": DROPOUT_RATE
        }, f)
    print(f"Saved model → {MODEL_OUT}", flush=True)


if __name__ == "__main__":
    main()

"""SASRec v2 — BPR loss + GELU Linear FFN + early stopping.

Improvements over sasrec template:
  - BPR loss instead of BCE
  - GELU activation with Linear layers (not Conv1d)
  - batch_first=True MultiheadAttention
  - maxlen=128 (up from 50)
  - dropout=0.5 (up from 0.2)
  - Early stopping with patience=20 and best-checkpoint saving
  - AdamW optimizer with weight decay

Key hyper-parameters (all tunable by RecHarness):
  MAXLEN       : max history length
  HIDDEN_UNITS : embedding / attention dimension
  NUM_BLOCKS   : number of self-attention blocks
  NUM_HEADS    : number of attention heads
  DROPOUT_RATE : dropout on attention weights and FFN
  LR           : AdamW learning rate
  BATCH_SIZE   : training batch size
  NUM_EPOCHS   : max training epochs
  WEIGHT_DECAY : AdamW weight decay
  PATIENCE     : early stopping patience (epochs without val improvement)
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
MAXLEN       = int(os.environ.get("SASREC_MAXLEN",       "128"))
HIDDEN_UNITS = int(os.environ.get("SASREC_HIDDEN",       "64"))
NUM_BLOCKS   = int(os.environ.get("SASREC_BLOCKS",       "2"))
NUM_HEADS    = int(os.environ.get("SASREC_HEADS",        "1"))
DROPOUT_RATE = float(os.environ.get("SASREC_DROPOUT",   "0.5"))
LR           = float(os.environ.get("SASREC_LR",         "1e-3"))
BATCH_SIZE   = int(os.environ.get("SASREC_BATCH",        "128"))
NUM_EPOCHS   = int(os.environ.get("SASREC_EPOCHS",       "200"))
WEIGHT_DECAY = float(os.environ.get("SASREC_WD",         "1e-4"))
PATIENCE     = int(os.environ.get("SASREC_PATIENCE",     "20"))

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


def _load_valid(data_dir: str) -> dict[int, list[int]]:
    user_valid: dict[int, list[int]] = {}
    for ds in DATASETS:
        valid = _read_split(os.path.join(data_dir, f"{ds}_valid.txt"))
        for u, seq in valid.items():
            user_valid[u] = user_valid.get(u, []) + seq
    return user_valid


# ── BPR Dataset ───────────────────────────────────────────────────────

class SASRecDataset(Dataset):
    """Generate (seq, pos, neg) triplets for BPR loss training."""

    def __init__(self, user_train: dict[int, list[int]], itemnum: int, maxlen: int):
        self.user_train = user_train
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.users = [u for u, seq in user_train.items() if len(seq) >= 2]

    def __len__(self):
        return len(self.users) * 5

    def __getitem__(self, idx):
        u = self.users[idx % len(self.users)]
        seq_full = self.user_train[u]

        seq = np.zeros(self.maxlen, dtype=np.int64)
        pos = np.zeros(self.maxlen, dtype=np.int64)
        neg = np.zeros(self.maxlen, dtype=np.int64)

        nxt = seq_full[-1]
        ts = set(seq_full)
        slot = self.maxlen - 1

        for item in reversed(seq_full[:-1]):
            seq[slot] = item
            pos[slot] = nxt
            neg_item = random.randint(1, self.itemnum)
            while neg_item in ts:
                neg_item = random.randint(1, self.itemnum)
            neg[slot] = neg_item
            nxt = item
            slot -= 1
            if slot == -1:
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
        self.fc1 = nn.Linear(hidden_units, hidden_units * 4)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_units * 4, hidden_units)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, H)
        return x + self.dropout2(self.fc2(self.dropout1(self.act(self.fc1(x)))))


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
                nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(hidden, dropout))

        self.last_layernorm = nn.LayerNorm(hidden, eps=1e-8)

    def log2feats(self, log_seqs):
        """Convert log sequence to feature representations. Returns (B, L, H)."""
        seqs = self.item_emb(log_seqs)
        seqs *= (self.hidden ** 0.5)

        B, L = log_seqs.shape
        positions = torch.arange(L, device=log_seqs.device).unsqueeze(0).expand(B, -1)
        seqs = seqs + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)

        pad_mask = (log_seqs == 0)
        seqs = seqs.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # Causal attention mask: upper-triangular = True means "ignore"
        causal_mask = torch.triu(
            torch.ones(L, L, device=log_seqs.device, dtype=torch.bool), diagonal=1
        )

        for i in range(len(self.attention_layers)):
            normed = self.attention_layernorms[i](seqs)
            mha_out, _ = self.attention_layers[i](
                normed, normed, normed,
                attn_mask=causal_mask,
                key_padding_mask=pad_mask,
            )
            # masked_fill instead of multiply: NaN * 0 = NaN, but masked_fill assigns 0.0
            # directly. Without this, NaN from all-padding softmax in block N infects
            # keys in block N+1 (NaN + -inf = NaN, not -inf), poisoning real positions.
            seqs = (seqs + mha_out).masked_fill(pad_mask.unsqueeze(-1), 0.0)
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return self.last_layernorm(seqs)

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        """Forward pass returning (pos_scores, neg_scores) for BPR loss."""
        log_feats = self.log2feats(log_seqs)  # (B, L, H)

        pos_embs = self.item_emb(pos_seqs)  # (B, L, H)
        neg_embs = self.item_emb(neg_seqs)  # (B, L, H)

        pos_scores = (log_feats * pos_embs).sum(dim=-1)  # (B, L)
        neg_scores = (log_feats * neg_embs).sum(dim=-1)  # (B, L)

        return pos_scores, neg_scores


# ── Quick val HR@10 ───────────────────────────────────────────────────

def _quick_val(model, user_train, user_valid, itemnum, maxlen, device, n_users=200):
    model.eval()
    users = [u for u in user_valid if u in user_train]
    random.shuffle(users)
    users = users[:n_users]
    hit = 0
    with torch.no_grad():
        for u in users:
            hist = user_train[u][-maxlen:]
            padded = [0] * (maxlen - len(hist)) + hist
            seq = torch.tensor([padded], dtype=torch.long, device=device)
            log_feats = model.log2feats(seq)        # (1, L, H)
            user_rep = log_feats[0, -1, :]          # (H,)

            pos_items = user_valid[u]
            if not pos_items:
                continue
            target = pos_items[-1]

            # 99 random negatives + target
            neg_pool = set()
            all_items = set(user_train[u]) | set(pos_items)
            while len(neg_pool) < 99:
                ni = random.randint(1, itemnum)
                if ni not in all_items:
                    neg_pool.add(ni)
            cands = list(neg_pool) + [target]
            cand_t = torch.tensor(cands, dtype=torch.long, device=device)
            scores = (model.item_emb(cand_t) @ user_rep).cpu().numpy()
            rank = int((scores > scores[-1]).sum())
            if rank < 10:
                hit += 1
    model.train()
    return hit / max(len(users), 1)


# ── Training ──────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_train, itemnum = _load_all(DATA_DIR)
    user_valid = _load_valid(DATA_DIR)
    print(f"users={len(user_train)}, items={itemnum}", flush=True)

    dataset = SASRecDataset(user_train, itemnum, MAXLEN)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    model = SASRecModel(
        itemnum, HIDDEN_UNITS, MAXLEN, NUM_BLOCKS, NUM_HEADS, DROPOUT_RATE
    ).to(device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=WEIGHT_DECAY
    )

    best_val = -1.0
    no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        count = 0

        for seqs, pos, neg in loader:
            seqs = seqs.to(device)
            pos  = pos.to(device)
            neg  = neg.to(device)

            pos_scores, neg_scores = model(seqs, pos, neg)

            # BPR loss on non-padding positions
            mask = (pos != 0)
            bpr_loss = -torch.log(
                torch.sigmoid(pos_scores[mask] - neg_scores[mask]) + 1e-8
            ).mean()

            opt.zero_grad()
            bpr_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += bpr_loss.item()
            count += 1

        avg = total_loss / max(count, 1)

        # Evaluate every epoch for early stopping signal
        val_hr = _quick_val(model, user_train, user_valid, itemnum, MAXLEN, device)
        print(f"epoch {epoch:3d}  loss={avg:.4f}  val_score: {val_hr:.4f}", flush=True)

        if val_hr > best_val:
            best_val = val_hr
            no_improve = 0
            torch.save(model.state_dict(), MODEL_OUT)
            with open(META_OUT, "wb") as f:
                pickle.dump({
                    "itemnum": itemnum,
                    "maxlen": MAXLEN,
                    "hidden": HIDDEN_UNITS,
                    "num_blocks": NUM_BLOCKS,
                    "num_heads": NUM_HEADS,
                    "dropout": DROPOUT_RATE,
                }, f)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={PATIENCE})", flush=True)
                break

    print(f"Best val HR@10={best_val:.4f}  model → {MODEL_OUT}", flush=True)


if __name__ == "__main__":
    main()

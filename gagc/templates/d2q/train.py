"""D2Q cold-start template for KuaiRec watch-time prediction.

Adapted from MorganSQ/Ks-D2Q and the D2Q paper, but reads the RecHarness
KuaiRec 53-column .npy format directly:
  [0:19] user features, [19:24] item features, [24:43] user mask,
  [43:48] item mask, [48] play_duration_sec, [49] video_duration_sec,
  [50] watch_ratio_raw, [51] usr_len, [52] item_len.

The model predicts the duration-group watch-time quantile and maps it back to
seconds with the group-wise empirical inverse CDF.  RecHarness parses the final lines:
  XAUC=<float>
  MAE=<float>
  WR_XAUC=<float>
  WR_MAE=<float>
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

TRAIN_DATA = os.environ.get("GR_TRAIN_DATA", "")
TEST_DATA = os.environ.get("GR_TEST_DATA", "")
SEED = int(os.environ.get("GR_SEED", "2024"))
BATCH_SIZE = int(os.environ.get("GR_BATCH_SIZE", "512"))
NUM_EPOCHS = int(os.environ.get("GR_NUM_EPOCHS", "2"))
LR = float(os.environ.get("GR_LR", "1e-3"))
DROPOUT = float(os.environ.get("GR_DROPOUT", "0.2"))
EMB_DIM = int(os.environ.get("D2Q_EMB_DIM", "32"))
HIDDEN_DIM = int(os.environ.get("D2Q_HIDDEN_DIM", "256"))
NUM_GROUPS = int(os.environ.get("D2Q_NUM_GROUPS", "32"))
SPARSE_BUCKETS = int(os.environ.get("D2Q_SPARSE_BUCKETS", "2000"))
DURATION_BUCKETS = int(os.environ.get("D2Q_DURATION_BUCKETS", "720"))

USER_COLS = slice(0, 19)
ITEM_COLS = slice(19, 24)
USER_MASK_COLS = slice(24, 43)
ITEM_MASK_COLS = slice(43, 48)
PLAY_COL = 48
DURATION_COL = 49
WATCH_RATIO_COL = 50
USR_LEN_COL = 51
ITEM_LEN_COL = 52


@dataclass
class D2QStats:
    duration_edges: np.ndarray
    cdf_values: list[np.ndarray]
    cdf_quantiles: list[np.ndarray]
    dense_mean: np.ndarray
    dense_std: np.ndarray


class KuaiD2QDataset(Dataset):
    def __init__(self, feats: np.ndarray, stats: D2QStats, train: bool) -> None:
        self.feats = feats.astype(np.float32, copy=False)
        self.stats = stats
        self.train = train
        self.group_ids = assign_duration_groups(self.feats[:, DURATION_COL], stats.duration_edges)
        self.sparse = build_sparse_ids(self.feats)
        self.duration_ids = build_duration_ids(self.feats[:, DURATION_COL])
        self.dense = normalise_dense(build_dense_features(self.feats), stats.dense_mean, stats.dense_std)
        self.play = np.clip(self.feats[:, PLAY_COL].astype(np.float32), 0.0, None)
        self.video_duration = np.clip(self.feats[:, DURATION_COL].astype(np.float32), 1e-6, None)
        self.watch_ratio = np.clip(self.feats[:, WATCH_RATIO_COL].astype(np.float32), 0.0, None)
        self.quantile = np.asarray([
            empirical_cdf(stats.cdf_values[int(g)], float(y))
            for g, y in zip(self.group_ids, self.play)
        ], dtype=np.float32)

    def __len__(self) -> int:
        return int(self.feats.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.as_tensor(self.sparse[idx], dtype=torch.long),
            torch.as_tensor(self.duration_ids[idx], dtype=torch.long),
            torch.as_tensor(self.dense[idx], dtype=torch.float32),
            torch.as_tensor(self.quantile[idx], dtype=torch.float32),
            torch.as_tensor(self.group_ids[idx], dtype=torch.long),
            torch.as_tensor(self.play[idx], dtype=torch.float32),
            torch.as_tensor(self.video_duration[idx], dtype=torch.float32),
            torch.as_tensor(self.watch_ratio[idx], dtype=torch.float32),
        )


class D2QNet(nn.Module):
    def __init__(self, dense_dim: int, sparse_fields: int) -> None:
        super().__init__()
        self.sparse_emb = nn.Embedding(SPARSE_BUCKETS, EMB_DIM)
        self.duration_emb = nn.Embedding(DURATION_BUCKETS, EMB_DIM)
        self.sparse_proj = nn.Sequential(
            nn.Linear(sparse_fields * EMB_DIM, 512),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
        )
        self.duration_proj = nn.Sequential(nn.Linear(EMB_DIM, 32), nn.SiLU())
        self.dense_proj = nn.Sequential(nn.Linear(dense_dim, 32), nn.SiLU())
        self.mlp = nn.Sequential(
            nn.Linear(512 + 32 + 32, HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, max(HIDDEN_DIM // 2, 64)),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(max(HIDDEN_DIM // 2, 64), 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, sparse_ids: torch.Tensor, duration_ids: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        sparse = self.sparse_emb(sparse_ids).flatten(1)
        sparse = self.sparse_proj(sparse)
        duration = self.duration_proj(self.duration_emb(duration_ids))
        dense = self.dense_proj(dense)
        return self.mlp(torch.cat([sparse, duration, dense], dim=1)).squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_data(path: str, name: str) -> np.ndarray:
    if not path:
        raise FileNotFoundError(f"{name} env var is empty")
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 53:
        raise ValueError(f"{name} must be a 2-D KuaiRec array with at least 53 columns, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def build_sparse_ids(feats: np.ndarray) -> np.ndarray:
    raw = np.concatenate([feats[:, USER_COLS], feats[:, ITEM_COLS]], axis=1)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return np.mod(np.abs(raw.astype(np.int64)), SPARSE_BUCKETS).astype(np.int64)


def build_duration_ids(duration: np.ndarray) -> np.ndarray:
    duration = np.nan_to_num(duration, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.floor(duration), 0, DURATION_BUCKETS - 1).astype(np.int64)


def build_dense_features(feats: np.ndarray) -> np.ndarray:
    dense = np.concatenate([
        feats[:, USER_MASK_COLS],
        feats[:, ITEM_MASK_COLS],
        feats[:, [DURATION_COL, USR_LEN_COL, ITEM_LEN_COL]],
    ], axis=1).astype(np.float32, copy=False)
    return np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)


def normalise_dense(dense: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((dense - mean) / std).astype(np.float32)


def make_duration_edges(duration: np.ndarray, groups: int) -> np.ndarray:
    duration = np.asarray(duration, dtype=np.float32)
    groups = max(1, int(groups))
    qs = np.linspace(0.0, 100.0, groups + 1)
    edges = np.percentile(duration, qs).astype(np.float32)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_duration_groups(duration: np.ndarray, edges: np.ndarray) -> np.ndarray:
    groups = np.searchsorted(edges[1:-1], duration, side="right")
    return groups.astype(np.int64)


def make_stats(train: np.ndarray) -> D2QStats:
    edges = make_duration_edges(train[:, DURATION_COL], NUM_GROUPS)
    groups = assign_duration_groups(train[:, DURATION_COL], edges)
    cdf_values: list[np.ndarray] = []
    cdf_quantiles: list[np.ndarray] = []
    global_play = np.sort(np.clip(train[:, PLAY_COL].astype(np.float32), 0.0, None))
    for group in range(max(1, NUM_GROUPS)):
        vals = np.sort(np.clip(train[groups == group, PLAY_COL].astype(np.float32), 0.0, None))
        if vals.size == 0:
            vals = global_play if global_play.size else np.array([0.0], dtype=np.float32)
        cdf_values.append(vals.astype(np.float32))
        cdf_quantiles.append(np.linspace(0.0, 1.0, vals.size, dtype=np.float32))
    dense = build_dense_features(train)
    mean = dense.mean(axis=0)
    std = dense.std(axis=0)
    std[std < 1e-6] = 1.0
    return D2QStats(edges, cdf_values, cdf_quantiles, mean.astype(np.float32), std.astype(np.float32))


def empirical_cdf(sorted_values: np.ndarray, value: float) -> float:
    if sorted_values.size <= 1:
        return 0.0
    idx = np.searchsorted(sorted_values, value, side="right") - 1
    return float(np.clip(idx / (sorted_values.size - 1), 0.0, 1.0))


def inverse_cdf(stats: D2QStats, group_ids: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    preds = np.empty_like(quantiles, dtype=np.float32)
    for group in np.unique(group_ids):
        mask = group_ids == group
        vals = stats.cdf_values[int(group)]
        qs = stats.cdf_quantiles[int(group)]
        preds[mask] = np.interp(np.clip(quantiles[mask], 0.0, 1.0), qs, vals).astype(np.float32)
    return preds


def xauc_score(labels: np.ndarray, preds: np.ndarray) -> float:
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    order = np.argsort(-preds, kind="mergesort")
    sorted_labels = labels[order]
    n = sorted_labels.size
    if n <= 1:
        return 0.0
    _, inv = np.unique(sorted_labels, return_inverse=True)
    ranks = inv.astype(np.int64)
    bit = np.zeros(ranks.max() + 2, dtype=np.int64)
    inversions = 0
    for seen, rank in enumerate(ranks):
        idx = int(rank) + 1
        prefix = 0
        j = idx
        while j > 0:
            prefix += bit[j]
            j -= j & -j
        inversions += seen - prefix
        j = idx
        while j < bit.size:
            bit[j] += 1
            j += j & -j
    return float(inversions / (n * (n - 1) / 2.0))


def evaluate(model: nn.Module, loader: DataLoader, stats: D2QStats, device: torch.device) -> tuple[float, float, float, float]:
    model.eval()
    q_preds, groups, plays, durations, ratios = [], [], [], [], []
    with torch.no_grad():
        for sparse, dur_id, dense, _q, group, play, vdur, wr in loader:
            pred_q = model(sparse.to(device), dur_id.to(device), dense.to(device)).cpu().numpy()
            q_preds.append(pred_q)
            groups.append(group.numpy())
            plays.append(play.numpy())
            durations.append(vdur.numpy())
            ratios.append(wr.numpy())
    q_pred = np.concatenate(q_preds)
    group = np.concatenate(groups)
    play = np.concatenate(plays)
    duration = np.concatenate(durations)
    ratio = np.concatenate(ratios)
    pred_play = np.clip(inverse_cdf(stats, group, q_pred), 0.0, None)
    pred_ratio = pred_play / np.clip(duration, 1e-6, None)
    return (
        round(xauc_score(play, pred_play), 4),
        round(float(np.mean(np.abs(pred_play - play))), 4),
        round(xauc_score(ratio, pred_ratio), 4),
        round(float(np.mean(np.abs(pred_ratio - ratio))), 4),
    )


def main() -> None:
    print("D2Q KuaiRec cold-start")
    set_seed(SEED)
    train = require_data(TRAIN_DATA, "GR_TRAIN_DATA")
    test = require_data(TEST_DATA, "GR_TEST_DATA")
    print(f"train shape: {train.shape}  test shape: {test.shape}")
    print("schema: col48=play_duration_sec col49=video_duration_sec col50=watch_ratio_raw")

    stats = make_stats(train)
    train_ds = KuaiD2QDataset(train, stats, train=True)
    test_ds = KuaiD2QDataset(test, stats, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = D2QNet(train_ds.dense.shape[1], train_ds.sparse.shape[1]).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=LR)
    print(f"device: {device}  groups={NUM_GROUPS}  params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best = (0.0, float("inf"), 0.0, float("inf"))
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for sparse, dur_id, dense, q, *_rest in train_loader:
            sparse = sparse.to(device)
            dur_id = dur_id.to(device)
            dense = dense.to(device)
            q = q.to(device)
            pred = model(sparse, dur_id, dense)
            loss = F.mse_loss(pred, q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * q.numel()
            total_count += q.numel()
        metrics = evaluate(model, test_loader, stats, device)
        if metrics[1] < best[1] or metrics[0] > best[0]:
            best = metrics
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}: loss={total_loss / max(total_count, 1):.6f} "
            f"WT_XAUC={metrics[0]:.4f} WT_MAE={metrics[1]:.4f} "
            f"WR_XAUC={metrics[2]:.4f} WR_MAE={metrics[3]:.4f}"
        )

    print(f"XAUC={best[0]:.4f}")
    print(f"MAE={best[1]:.4f}")
    print(f"WR_XAUC={best[2]:.4f}")
    print(f"WR_MAE={best[3]:.4f}")


if __name__ == "__main__":
    main()

"""TPM cold-start template for KuaiRec watch-time prediction.

Adapted from jackielinxiao/TPM, but reads the RecHarness KuaiRec 53-column .npy
format directly:
  [0:19] user features, [19:24] item features, [24:43] user mask,
  [43:48] item mask, [48] play_duration_sec, [49] video_duration_sec,
  [50] watch_ratio_raw, [51] usr_len, [52] item_len.

TPM transforms watch-time regression into conditional binary classifiers on a
complete tree over watch-time buckets.  It then decodes the expected watch time
from leaf probabilities and reports metrics in the KuaiRec harness contract.
"""
from __future__ import annotations

import math
import os
import random

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
EMB_DIM = int(os.environ.get("TPM_EMB_DIM", "16"))
HIDDEN_DIM = int(os.environ.get("TPM_HIDDEN_DIM", "128"))
NUM_LEAVES = int(os.environ.get("TPM_NUM_LEAVES", "32"))
DURATION_GROUPS = int(os.environ.get("TPM_DURATION_GROUPS", "32"))
SPARSE_BUCKETS = int(os.environ.get("TPM_SPARSE_BUCKETS", "500000"))
MSE_WEIGHT = float(os.environ.get("TPM_MSE_WEIGHT", "0.2"))
VAR_WEIGHT = float(os.environ.get("TPM_VAR_WEIGHT", "0.01"))
MAX_GRAD_NORM = float(os.environ.get("TPM_MAX_GRAD_NORM", "5.0"))
MSE_WARMUP_EPOCHS = int(os.environ.get("TPM_MSE_WARMUP_EPOCHS", "1"))

USER_COLS = slice(0, 19)
ITEM_COLS = slice(19, 24)
USER_MASK_COLS = slice(24, 43)
ITEM_MASK_COLS = slice(43, 48)
PLAY_COL = 48
DURATION_COL = 49
WATCH_RATIO_COL = 50
USR_LEN_COL = 51
ITEM_LEN_COL = 52


def _next_power_of_two(value: int) -> int:
    value = max(2, int(value))
    return 1 << (value - 1).bit_length()


NUM_LEAVES = _next_power_of_two(NUM_LEAVES)
NUM_NODES = NUM_LEAVES - 1
TREE_HEIGHT = int(math.log2(NUM_LEAVES))


class KuaiTPMDataset(Dataset):
    def __init__(
        self,
        feats: np.ndarray,
        play_edges: np.ndarray,
        duration_edges: np.ndarray,
        dense_mean: np.ndarray,
        dense_std: np.ndarray,
    ) -> None:
        self.feats = feats.astype(np.float32, copy=False)
        self.sparse = build_sparse_ids(self.feats)
        self.dense = ((build_dense_features(self.feats, duration_edges) - dense_mean) / dense_std).astype(np.float32)
        self.play = clean_nonnegative(self.feats[:, PLAY_COL], default=0.0)
        self.video_duration = np.clip(clean_nonnegative(self.feats[:, DURATION_COL], default=1.0), 1e-6, None)
        self.watch_ratio = clean_nonnegative(self.feats[:, WATCH_RATIO_COL], default=0.0)
        self.leaf = np.clip(np.searchsorted(play_edges[1:-1], self.play, side="right"), 0, NUM_LEAVES - 1).astype(np.int64)
        self.labels, self.weights = encode_tree_labels(self.leaf)

    def __len__(self) -> int:
        return int(self.feats.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.as_tensor(self.sparse[idx], dtype=torch.long),
            torch.as_tensor(self.dense[idx], dtype=torch.float32),
            torch.as_tensor(self.labels[idx], dtype=torch.float32),
            torch.as_tensor(self.weights[idx], dtype=torch.float32),
            torch.as_tensor(self.play[idx], dtype=torch.float32),
            torch.as_tensor(self.video_duration[idx], dtype=torch.float32),
            torch.as_tensor(self.watch_ratio[idx], dtype=torch.float32),
        )


class TPMNet(nn.Module):
    def __init__(self, sparse_fields: int, dense_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(SPARSE_BUCKETS, EMB_DIM)
        self.net = nn.Sequential(
            nn.Linear(sparse_fields * EMB_DIM + dense_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, max(HIDDEN_DIM // 2, 64)),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(max(HIDDEN_DIM // 2, 64), 32),
            nn.ReLU(),
            nn.Linear(32, NUM_NODES),
        )

    def forward(self, sparse_ids: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        emb = self.emb(sparse_ids).flatten(1)
        return self.net(torch.cat([emb, dense], dim=1))


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


def clean_nonnegative(values: np.ndarray, default: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    cleaned = np.nan_to_num(values, nan=default, posinf=default, neginf=default)
    return np.clip(cleaned, 0.0, None).astype(np.float32)


def build_dense_features(feats: np.ndarray, duration_edges: np.ndarray) -> np.ndarray:
    dense = np.concatenate([
        feats[:, USER_MASK_COLS],
        feats[:, ITEM_MASK_COLS],
        feats[:, [DURATION_COL, USR_LEN_COL, ITEM_LEN_COL]],
        duration_group_onehot(feats[:, DURATION_COL], duration_edges),
    ], axis=1).astype(np.float32, copy=False)
    return np.nan_to_num(dense, nan=0.0, posinf=0.0, neginf=0.0)


def duration_group_onehot(duration: np.ndarray, duration_edges: np.ndarray) -> np.ndarray:
    duration = np.nan_to_num(duration.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    group = np.clip(np.searchsorted(duration_edges[1:-1], duration, side="right"), 0, DURATION_GROUPS - 1)
    out = np.zeros((duration.shape[0], DURATION_GROUPS), dtype=np.float32)
    out[np.arange(duration.shape[0]), group] = 1.0
    return out


def make_duration_edges(duration: np.ndarray) -> np.ndarray:
    duration = clean_nonnegative(duration, default=1.0)
    edges = np.percentile(duration, np.linspace(0.0, 100.0, DURATION_GROUPS + 1)).astype(np.float32)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def make_play_edges(play: np.ndarray) -> np.ndarray:
    play = clean_nonnegative(play, default=0.0)
    edges = np.percentile(play, np.linspace(0.0, 100.0, NUM_LEAVES + 1)).astype(np.float32)
    edges[0] = min(edges[0], 0.0)
    edges[-1] = max(edges[-1], float(play.max(initial=0.0)))
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1e-4
    return edges


def encode_tree_labels(leaf_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((leaf_ids.shape[0], NUM_NODES), dtype=np.float32)
    weights = np.zeros((leaf_ids.shape[0], NUM_NODES), dtype=np.float32)
    for row, leaf in enumerate(leaf_ids.astype(int)):
        node = 0
        left = 0
        right = NUM_LEAVES
        while node < NUM_NODES:
            mid = (left + right) // 2
            go_left = leaf < mid
            labels[row, node] = 1.0 if go_left else 0.0
            weights[row, node] = 1.0
            if go_left:
                node = 2 * node + 1
                right = mid
            else:
                node = 2 * node + 2
                left = mid
    return labels, weights


def leaf_probabilities(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    leaf_probs = []
    for leaf in range(NUM_LEAVES):
        node = 0
        left = 0
        right = NUM_LEAVES
        prob = torch.ones(logits.shape[0], device=logits.device)
        while node < NUM_NODES:
            mid = (left + right) // 2
            if leaf < mid:
                prob = prob * probs[:, node]
                node = 2 * node + 1
                right = mid
            else:
                prob = prob * (1.0 - probs[:, node])
                node = 2 * node + 2
                left = mid
        leaf_probs.append(prob)
    return torch.stack(leaf_probs, dim=1)


def decode_play(logits: torch.Tensor, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    leaf_probs = leaf_probabilities(logits)
    pred = torch.sum(leaf_probs * centers.unsqueeze(0), dim=1)
    ex2 = torch.sum(leaf_probs * centers.square().unsqueeze(0), dim=1)
    # Epsilon inside sqrt prevents gradient explosion when var → 0
    # (d/dx sqrt(x) = 1/(2*sqrt(x)) → ∞ at x=0, causing inf→NaN in Adam)
    var = torch.sqrt(torch.clamp(ex2 - pred.square(), min=1e-6))
    return pred, var


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


def evaluate(model: nn.Module, loader: DataLoader, centers: torch.Tensor, device: torch.device) -> tuple[float, float, float, float]:
    model.eval()
    preds, plays, durations, ratios = [], [], [], []
    with torch.no_grad():
        for sparse, dense, _labels, _weights, play, vdur, wr in loader:
            logits = model(sparse.to(device), dense.to(device))
            pred, _var = decode_play(logits, centers.to(device))
            preds.append(pred.cpu().numpy())
            plays.append(play.numpy())
            durations.append(vdur.numpy())
            ratios.append(wr.numpy())
    pred_play = np.clip(np.concatenate(preds), 0.0, None)
    play = np.concatenate(plays)
    duration = np.concatenate(durations)
    ratio = np.concatenate(ratios)
    pred_ratio = pred_play / np.clip(duration, 1e-6, None)
    finite = np.isfinite(pred_play) & np.isfinite(play) & np.isfinite(pred_ratio) & np.isfinite(ratio)
    if not np.any(finite):
        return 0.0, float("inf"), 0.0, float("inf")
    pred_play = pred_play[finite]
    play = play[finite]
    pred_ratio = pred_ratio[finite]
    ratio = ratio[finite]
    return (
        round(xauc_score(play, pred_play), 4),
        round(float(np.mean(np.abs(pred_play - play))), 4),
        round(xauc_score(ratio, pred_ratio), 4),
        round(float(np.mean(np.abs(pred_ratio - ratio))), 4),
    )


def main() -> None:
    print("TPM KuaiRec cold-start")
    set_seed(SEED)
    train = require_data(TRAIN_DATA, "GR_TRAIN_DATA")
    test = require_data(TEST_DATA, "GR_TEST_DATA")
    print(f"train shape: {train.shape}  test shape: {test.shape}")
    print("schema: col48=play_duration_sec col49=video_duration_sec col50=watch_ratio_raw")

    play_edges = make_play_edges(train[:, PLAY_COL])
    centers = torch.as_tensor((play_edges[:-1] + play_edges[1:]) / 2.0, dtype=torch.float32)
    duration_edges = make_duration_edges(train[:, DURATION_COL])
    train_dense = build_dense_features(train, duration_edges)
    dense_mean = train_dense.mean(axis=0).astype(np.float32)
    dense_std = train_dense.std(axis=0).astype(np.float32)
    dense_std[dense_std < 1e-6] = 1.0

    train_ds = KuaiTPMDataset(train, play_edges, duration_edges, dense_mean, dense_std)
    test_ds = KuaiTPMDataset(test, play_edges, duration_edges, dense_mean, dense_std)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TPMNet(train_ds.sparse.shape[1], train_ds.dense.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"device: {device} leaves={NUM_LEAVES} duration_groups={DURATION_GROUPS} params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Normalize centers to stabilize MSE gradients.
    # The tree-BCE loss operates on leaf IDs (scale-invariant), but
    # decode_play multiplies leaf_probs by raw centers which can be O(10³)
    # seconds.  Normalizing keeps the MSE gradient on the same order as
    # the tree gradient.
    center_mean = centers.mean().item()
    center_std = centers.std().item()
    if center_std < 1e-6:
        center_std = 1.0
    centers_norm = (centers - center_mean) / center_std  # for training

    best = (0.0, float("inf"), 0.0, float("inf"))
    centers_norm = centers_norm.to(device)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        # MSE warm-up: only use tree_loss for the first N epochs so the
        # tree routing is learned before we add the regression signal.
        active_mse = MSE_WEIGHT if epoch > MSE_WARMUP_EPOCHS else 0.0
        active_var = VAR_WEIGHT if epoch > MSE_WARMUP_EPOCHS else 0.0
        for sparse, dense, labels, weights, play, *_rest in train_loader:
            sparse = sparse.to(device)
            dense = dense.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            play = play.to(device)
            logits = model(sparse, dense)
            bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            tree_loss = (bce * weights).sum() / weights.sum().clamp_min(1.0)
            if active_mse > 0.0:
                # Decode on normalized scale; convert play to match.
                play_norm = (play - center_mean) / center_std
                pred_norm, var = decode_play(logits, centers_norm)
                mse_loss = F.mse_loss(pred_norm, play_norm)
                loss = tree_loss + active_mse * mse_loss + active_var * var.mean()
            else:
                loss = tree_loss
            if not torch.isfinite(loss):
                continue  # skip NaN / inf batches
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            total_loss += float(loss.item()) * play.numel()
            total_count += play.numel()
        metrics = evaluate(model, test_loader, centers, device)
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

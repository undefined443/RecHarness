"""GR (Generative Regression) — KuaiRec Watch Time Prediction.

Cold-start template for RecHarness. All hyper-parameters are read from environment
variables so RecHarness can mutate them via code diffs without touching the source.

Environment variables
---------------------
Data paths (required):
    GR_TRAIN_DATA       path to train .npy file
    GR_TEST_DATA        path to test .npy file

Architecture:
    GR_FEAT_DIM         int   (default 128)  — embedding / feature dimension
    GR_HIDDEN_DIM       int   (default 128)  — encoder hidden / decoder d_model
    GR_N_HEAD           int   (default 8)    — transformer attention heads
    GR_DEC_LAYERS       int   (default 3)    — decoder transformer layers
    GR_DROPOUT          float (default 0.1)
    GR_FFN_DIM          int   (default 256)  — FFN hidden dim in decoder

Training:
    GR_BATCH_SIZE       int   (default 512)
    GR_NUM_EPOCHS       int   (default 2)   — short proxy run for RecHarness search
    GR_LR               float (default 5e-4)
    GR_SEED             int   (default 2024)

Loss:
    GR_CLS_WEIGHT       float (default 10.0) — weight for cross-entropy loss
    GR_HUBER_WEIGHT     float (default 1.0)  — weight for Huber regression loss

Vocabulary:
    GR_Q_START          float (default 0.9999)
    GR_Q_END            float (default 0.9)
    GR_Q_DECAY          float (default 0.99)
    GR_EPSILON          float (default 1e-6)

Decoding:
    GR_WINDOW_SIZE      int   (default 20)   — soft-argmax window size

Model variants:
    GR_USE_MIXUP        0/1   (default 0)    — embedding mixup
    GR_USE_CURRICULUM   0/1   (default 0)    — curriculum learning
    GR_CURRICULUM_TYPE  linear|exp|sigmoid   (default linear)
    GR_CURRICULUM_DECAY float (default 0.9999)
    GR_TEACHER_FORCE    float (default 0.5)  — initial teacher force ratio

Official KuaiRec WR npy schema:
    col48 = play_duration_sec, col49 = video_duration_sec, col50 = watch_ratio_raw

Output (printed at end, parsed by the RecHarness harness):
    XAUC=<WT_XAUC>
    MAE=<WT_MAE>
    WR_XAUC=<WR_XAUC>
    WR_MAE=<WR_MAE>
"""
from __future__ import annotations

import copy
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── make model subpackage importable when run as a script ─────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.decoder import Decoder
from model.encoder import Encoder
from model.transformer import Seq2Seq

# ═════════════════════════════════════════════════════════════════════════════
# Hyper-parameters from env
# ═════════════════════════════════════════════════════════════════════════════

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))

def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))

def _env_bool(name: str, default: int) -> bool:
    return bool(int(os.environ.get(name, default)))

def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


TRAIN_DATA  = _env_str("GR_TRAIN_DATA", "")
TEST_DATA   = _env_str("GR_TEST_DATA", "")
TARGET_COL  = 48  # play_duration_sec — watch-time target in seconds
VIDEO_DURATION_COL = 49  # video_duration_sec — duration feature + WR recovery
WATCH_RATIO_COL = 50  # watch_ratio_raw — WR metric ground truth
FEAT_DIM    = _env_int("GR_FEAT_DIM", 128)
HIDDEN_DIM  = _env_int("GR_HIDDEN_DIM", 128)
N_HEAD      = _env_int("GR_N_HEAD", 8)
DEC_LAYERS  = _env_int("GR_DEC_LAYERS", 3)
DROPOUT     = _env_float("GR_DROPOUT", 0.1)
FFN_DIM     = _env_int("GR_FFN_DIM", 256)
BATCH_SIZE  = _env_int("GR_BATCH_SIZE", 512)
NUM_EPOCHS  = _env_int("GR_NUM_EPOCHS", 2)
LR          = _env_float("GR_LR", 5e-4)
SEED        = _env_int("GR_SEED", 2024)
CLS_WEIGHT  = _env_float("GR_CLS_WEIGHT", 10.0)
HUBER_WEIGHT = _env_float("GR_HUBER_WEIGHT", 1.0)
Q_START     = _env_float("GR_Q_START", 0.9999)
Q_END       = _env_float("GR_Q_END", 0.9)
Q_DECAY     = _env_float("GR_Q_DECAY", 0.99)
EPSILON     = _env_float("GR_EPSILON", 1e-6)
WINDOW_SIZE = _env_int("GR_WINDOW_SIZE", 20)
USE_MIXUP   = _env_bool("GR_USE_MIXUP", 0)
USE_CURRICULUM = _env_bool("GR_USE_CURRICULUM", 0)
CURRICULUM_TYPE  = _env_str("GR_CURRICULUM_TYPE", "linear")
CURRICULUM_DECAY = _env_float("GR_CURRICULUM_DECAY", 0.9999)
TEACHER_FORCE    = _env_float("GR_TEACHER_FORCE", 0.5)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH   = os.path.join(SCRIPT_DIR, "best_model.pth")


# ═════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def seconds_to_hms(seconds: float):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return h, m, s


def init_weights(m: nn.Module) -> None:
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# Dynamic vocabulary
# ═════════════════════════════════════════════════════════════════════════════

def get_genral_vocab_dynamic_q(
    watch_ratio_all_for_vocab,
    q_start: float = 0.9999,
    q_end: float = 0.9,
    q_decay_rate: float = 0.99,
    epsilon: float = 1e-6,
) -> list:
    vocab = []
    data_sorted = np.sort(watch_ratio_all_for_vocab)[::-1]
    data = data_sorted.copy()
    err = float('inf')
    q = q_start
    zi = float('1000')
    while err > epsilon and int(zi) != 0:
        zi = np.percentile(data, q * 100)
        if int(zi) == 0:
            break
        data = np.where(data < zi, data, data - zi)
        vocab.append(int(zi))
        err = np.max(np.abs(data / (data_sorted + 1e-10)))
        q = max(q_end, q * q_decay_rate)
    return vocab


def reduce_numbers(number: int, vocab_int: list[int]) -> tuple[int, list[int]]:
    result = []
    remainder = number
    for v in vocab_int:
        while remainder >= v:
            remainder -= v
            result.append(v)
    return remainder, sorted(result, reverse=True)


def label_process(vocab, numeric_vocab, train_len, watch_ratio_all_for_vocab, device):
    special_tokens = ['<pad>', '<sos>', '<eos>']
    _pad_idx, sos_idx, eos_idx = 0, 1, 2
    vocab_int = [int(v) for v in numeric_vocab]
    vocab_map = {int(v): idx + len(special_tokens) for idx, v in enumerate(numeric_vocab)}

    durations_list = []
    for v in numeric_vocab:
        durations_list.append(int(v))
    # pad duration for special tokens
    duration_tensor = torch.tensor(
        [0, 0, 0] + durations_list, dtype=torch.float32, device=device
    )

    all_labels = []
    for ratio_int in tqdm(watch_ratio_all_for_vocab, desc="label_process"):
        remainder, tokens = reduce_numbers(int(ratio_int), vocab_int)
        if remainder != 0:
            raise ValueError(f"target {ratio_int} cannot be represented by current vocab; remainder={remainder}")
        seq = [sos_idx]
        for t in tokens:
            if t in vocab_map:
                seq.append(vocab_map[t])
        seq.append(eos_idx)
        all_labels.append(torch.tensor(seq, dtype=torch.long))

    train_labels = all_labels[:train_len]
    test_labels  = all_labels[train_len:]
    return duration_tensor, duration_tensor, train_labels, test_labels


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class Dataset_Kuai(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        alist = self.features[idx]
        usr_feas  = torch.tensor(alist[:19],    dtype=torch.long)
        item_feas = torch.tensor(
            list(alist[19:24]) + [alist[-1]], dtype=torch.long
        )
        umsk  = torch.tensor(alist[24:43], dtype=torch.long)
        imsk  = torch.tensor(alist[43:48], dtype=torch.long)
        play_time = torch.tensor([alist[TARGET_COL]], dtype=torch.float32)
        video_duration = torch.tensor([alist[VIDEO_DURATION_COL]], dtype=torch.float32)
        watch_ratio = torch.tensor([alist[WATCH_RATIO_COL]], dtype=torch.float32)
        label = self.labels[idx]
        return usr_feas, item_feas, umsk, imsk, play_time, video_duration, watch_ratio, label


def collate_fn(batch):
    usr_feas, item_feas, umsks, imsks, play_times, video_durations, watch_ratios, labels = zip(*batch)
    X_batch      = torch.stack([torch.cat([u, it]) for u, it in zip(usr_feas, item_feas)])
    qmsk_batch   = torch.stack(list(umsks))
    imsk_batch   = torch.stack(list(imsks))
    play_times_b = torch.stack(list(play_times))
    video_durations_b = torch.stack(list(video_durations))
    watch_ratios_b = torch.stack(list(watch_ratios))
    labels_padded = pad_sequence(
        [l.clone().detach() for l in labels], batch_first=True, padding_value=0
    )
    label_masks = pad_sequence(
        [torch.ones(len(l), dtype=torch.long) for l in labels],
        batch_first=True, padding_value=0,
    )
    return (
        X_batch, qmsk_batch, imsk_batch,
        play_times_b, video_durations_b, watch_ratios_b,
        labels_padded, label_masks,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════

def _inverse_pairs(data: list) -> int:
    if len(data) <= 1:
        return 0

    def merge(left, right):
        arr, cnt = [], left[1] + right[1]
        la, ra = left[0], right[0]
        i, j = len(la) - 1, len(ra) - 1
        while i >= 0 and j >= 0:
            if la[i] > ra[j]:
                arr.append(la[i]); cnt += j + 1; i -= 1
            else:
                arr.append(ra[j]); j -= 1
        while i >= 0: arr.append(la[i]); i -= 1
        while j >= 0: arr.append(ra[j]); j -= 1
        return arr[::-1], cnt

    def mergesort(a):
        if len(a) == 1:
            return a, 0
        mid = len(a) // 2
        return merge(mergesort(a[:mid]), mergesort(a[mid:]))

    return mergesort(data)[1]


def xauc_score(labels: np.ndarray, preds: np.ndarray) -> float:
    lp = list(zip(labels.reshape(-1), preds.reshape(-1)))
    sorted_lp = sorted(lp, key=lambda x: x[1], reverse=True)
    n = len(sorted_lp)
    pairs_cnt = n * (n - 1) / 2
    labels_sorted = [e[0] for e in sorted_lp]
    total_positive = _inverse_pairs(labels_sorted)
    return total_positive / pairs_cnt if pairs_cnt > 0 else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(test_loader, model, window_size, use_embedding_mixup, durations, device, max_len=20):
    model.eval()
    all_pred_wt, all_gt_wt, all_vdur, all_gt_wr = [], [], [], []

    for (X_batch, qmsk, imsk, play_times, video_durations, watch_ratios,
         _, _) in tqdm(test_loader, desc="eval", leave=False):
        X_batch    = X_batch.to(device)
        qmsk       = qmsk.to(device)
        imsk       = imsk.to(device)
        play_times = play_times.to(device)
        video_durations = video_durations.to(device)
        watch_ratios = watch_ratios.to(device)

        enc_src, src_mask = model.get_encoder_result(X_batch, qmsk, imsk)
        B = X_batch.size(0)

        # Auto-regressive decoding
        trg = torch.ones(B, 1, dtype=torch.long, device=device)  # <sos> = 1
        generated = trg.clone()

        for _ in range(max_len - 1):
            trg_mask = model.make_trg_mask(generated)
            out, _ = model.decoder(generated, enc_src, trg_mask, src_mask)
            next_token = out[:, -1:, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)

        # Find first EOS (index 2) per sequence
        eos_mask = (generated == 2)
        eos_positions = eos_mask.float().argmax(dim=1)
        has_eos = eos_mask.any(dim=1)
        eos_positions = torch.where(has_eos, eos_positions, torch.full_like(eos_positions, generated.shape[1]))

        # Windowed soft-argmax over next-token logits.  Use generated[:, :-1]
        # so the initial <sos> position is not counted as a watch-time token.
        decode_input = generated[:, :-1]
        trg_mask = model.make_trg_mask(decode_input)
        logits, _ = model.decoder(decode_input, enc_src, trg_mask, src_mask)

        softmax_out = F.softmax(logits, dim=-1)
        vocab_size  = softmax_out.shape[-1]
        seq_len     = softmax_out.shape[1]

        argmax_idx   = softmax_out.argmax(dim=-1)
        indices      = torch.arange(vocab_size, device=device).unsqueeze(0).unsqueeze(0).expand(B, seq_len, -1)
        start_idx    = argmax_idx.unsqueeze(-1) - window_size // 2
        end_idx      = argmax_idx.unsqueeze(-1) + window_size // 2 + 1
        win_mask     = (indices >= start_idx) & (indices < end_idx)
        weighted_dur = torch.sum(softmax_out * win_mask * durations.unsqueeze(0).unsqueeze(0), dim=-1)

        # Mask positions after EOS
        range_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        valid_mask    = range_indices < eos_positions.unsqueeze(1)
        weighted_dur  = weighted_dur * valid_mask.float()

        pred_watch = weighted_dur.sum(dim=1) / 1000
        all_pred_wt.extend(pred_watch.cpu().numpy())
        all_gt_wt.extend(play_times.squeeze(-1).cpu().numpy())
        all_vdur.extend(video_durations.squeeze(-1).cpu().numpy())
        all_gt_wr.extend(watch_ratios.squeeze(-1).cpu().numpy())

    all_pred_wt = np.array(all_pred_wt)
    all_gt_wt   = np.array(all_gt_wt)
    all_vdur    = np.array(all_vdur)
    all_gt_wr   = np.array(all_gt_wr)
    all_pred_wr = all_pred_wt / np.clip(all_vdur, 1e-6, None)

    wt_xauc = round(float(xauc_score(all_gt_wt, all_pred_wt)), 4)
    wt_mae  = round(float(np.mean(np.abs(all_pred_wt - all_gt_wt))), 4)
    wr_xauc = round(float(xauc_score(all_gt_wr, all_pred_wr)), 4)
    wr_mae  = round(float(np.mean(np.abs(all_pred_wr - all_gt_wr))), 4)
    return wt_xauc, wt_mae, wr_xauc, wr_mae


# ═════════════════════════════════════════════════════════════════════════════
# Training helpers
# ═════════════════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, lr: float):
    emb_params, other_params = [], []
    for name, param in model.named_parameters():
        if 'embedding' in name:
            emb_params.append(param)
        else:
            other_params.append(param)
    return torch.optim.Adam([
        {'params': emb_params, 'weight_decay': 1e-4},
        {'params': other_params, 'weight_decay': 0},
    ], lr=lr)


def compute_teacher_force_ratio(step: int) -> float:
    if not USE_CURRICULUM:
        return TEACHER_FORCE
    t = TEACHER_FORCE
    d = CURRICULUM_DECAY
    if CURRICULUM_TYPE == 'linear':
        return max(0.0, t - d * step)
    elif CURRICULUM_TYPE == 'exp':
        return t * (d ** step)
    elif CURRICULUM_TYPE == 'sigmoid':
        return t * d / (d + math.exp(step / d))
    return t


def train_one_epoch(epoch, model, train_loader, optimizer, criterion, durations, device):
    model.train()
    epoch_loss = 0
    for num_step, (X, qmsk, imsk, play_times, _video_durations, _watch_ratios,
                   seq_labels, seq_masks) in enumerate(train_loader, 1):
        X          = X.to(device)
        qmsk       = qmsk.to(device)
        imsk       = imsk.to(device)
        play_times = play_times.to(device)
        seq_labels = seq_labels.to(device=device, dtype=torch.long)
        seq_masks  = seq_masks.to(device)

        optimizer.zero_grad()
        tfr = compute_teacher_force_ratio(num_step - 1)

        output = model(X, qmsk, imsk, seq_labels[:, :-1], seq_masks[:, :-1], tfr)
        output_softmax = F.softmax(output, dim=-1)

        vocab_size = output.shape[-1]
        seq_len    = output.shape[1]

        argmax_idx = output_softmax.argmax(dim=-1)
        indices    = torch.arange(vocab_size, device=device).unsqueeze(0).unsqueeze(0).expand(
            output.shape[0], seq_len, -1
        )
        start_idx  = argmax_idx.unsqueeze(-1) - WINDOW_SIZE // 2
        end_idx    = argmax_idx.unsqueeze(-1) + WINDOW_SIZE // 2 + 1
        win_mask   = (indices >= start_idx) & (indices < end_idx)
        weighted_dur = torch.sum(output_softmax * win_mask * durations.unsqueeze(0).unsqueeze(0), dim=-1)
        weighted_dur = weighted_dur * seq_masks[:, 1:]
        pre_watch    = weighted_dur.sum(dim=1) / 1000

        logits   = output.contiguous().view(-1, vocab_size)
        trg_flat = seq_labels[:, 1:].contiguous().view(-1)
        loss_ce  = criterion(logits, trg_flat)
        loss_hub = F.huber_loss(play_times.squeeze(), pre_watch, reduction='sum')
        loss     = CLS_WEIGHT * loss_ce + HUBER_WEIGHT * loss_hub

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if num_step == 1 or num_step % 500 == 0:
            print(f"Epoch {epoch}/{NUM_EPOCHS} step {num_step}: CE={loss_ce.item():.4f} Huber={loss_hub.item():.4f}")

    return epoch_loss


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    if not TRAIN_DATA or not TEST_DATA:
        raise ValueError("GR_TRAIN_DATA and GR_TEST_DATA env vars must be set")

    print(f"FEAT_DIM={FEAT_DIM} HIDDEN_DIM={HIDDEN_DIM} N_HEAD={N_HEAD} DEC_LAYERS={DEC_LAYERS}")
    print(f"DROPOUT={DROPOUT} FFN_DIM={FFN_DIM} BATCH_SIZE={BATCH_SIZE} NUM_EPOCHS={NUM_EPOCHS}")
    print(f"LR={LR} CLS_WEIGHT={CLS_WEIGHT} HUBER_WEIGHT={HUBER_WEIGHT}")
    print(f"Q_START={Q_START} Q_END={Q_END} Q_DECAY={Q_DECAY} WINDOW_SIZE={WINDOW_SIZE}")
    print(f"USE_MIXUP={USE_MIXUP} USE_CURRICULUM={USE_CURRICULUM} TEACHER_FORCE={TEACHER_FORCE}")

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_feats = np.load(TRAIN_DATA)
    test_feats  = np.load(TEST_DATA)
    print(f"train shape: {train_feats.shape}  test shape: {test_feats.shape}")

    target_train = train_feats[:, TARGET_COL]
    target_test  = test_feats[:, TARGET_COL]
    target_all   = np.concatenate([target_train, target_test])
    target_int   = (target_all * 1000).astype(int)

    # Encode video-duration column for item_duration_embedding.
    dur_all   = np.concatenate([train_feats[:, VIDEO_DURATION_COL], test_feats[:, VIDEO_DURATION_COL]])
    encoder   = LabelEncoder()
    encoder.fit(np.sort(np.unique(dur_all)))
    enc_dur   = encoder.transform(dur_all)
    train_feats = np.column_stack([train_feats, enc_dur[:len(train_feats)]])
    test_feats  = np.column_stack([test_feats,  enc_dur[len(train_feats):]])
    print(f"after duration encoding — train: {train_feats.shape}  test: {test_feats.shape}")

    feats_all           = np.concatenate([train_feats, test_feats])
    max_values_per_col  = np.amax(feats_all, axis=0)

    # ── Vocabulary ────────────────────────────────────────────────────────
    special_tokens  = ['<pad>', '<sos>', '<eos>']
    numeric_vocab   = get_genral_vocab_dynamic_q(
        target_int, q_start=Q_START, q_end=Q_END, q_decay_rate=Q_DECAY, epsilon=EPSILON
    )
    vocab = special_tokens + numeric_vocab
    print(f"vocab size: {len(vocab)}")

    durations, _, train_labels, test_labels = label_process(
        vocab, numeric_vocab, len(train_feats), target_int, device
    )

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_ds     = Dataset_Kuai(train_feats, train_labels)
    test_ds      = Dataset_Kuai(test_feats,  test_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    enc = Encoder(input_dim=FEAT_DIM * 5, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT)
    dec = Decoder(
        d_model=HIDDEN_DIM, n_head=N_HEAD, ffn_hidden=FFN_DIM,
        dec_voc_size=len(vocab), drop_prob=DROPOUT, n_layers=DEC_LAYERS,
    )
    model = Seq2Seq(enc, dec, USE_MIXUP, max_values_per_col, FEAT_DIM, device).to(device)
    model.apply(init_weights)
    print(f"trainable parameters: {count_parameters(model):,}")

    optimizer = build_optimizer(model, LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    best_mae   = float('inf')
    best_xauc  = 0.0
    best_wr_mae = float('inf')
    best_wr_xauc = 0.0
    best_state = None
    start_time = time.time()

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(epoch, model, train_loader, optimizer, criterion, durations, device)

        xauc, mae, wr_xauc, wr_mae = evaluate(test_loader, model, WINDOW_SIZE, USE_MIXUP, durations, device)
        print(f"Epoch {epoch}/{NUM_EPOCHS}: WT_XAUC={xauc:.4f}  WT_MAE={mae:.4f}  WR_XAUC={wr_xauc:.4f}  WR_MAE={wr_mae:.4f}")

        if mae < best_mae:
            best_mae   = mae
            best_xauc  = xauc
            best_wr_mae = wr_mae
            best_wr_xauc = wr_xauc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, SAVE_PATH)

        elapsed = time.time() - start_time
        eh, em, es = seconds_to_hms(elapsed)
        remaining = (elapsed / epoch) * (NUM_EPOCHS - epoch)
        rh, rm, rs = seconds_to_hms(remaining)
        print(f"Elapsed {eh}h{em}m{es:.0f}s  Remaining ~{rh}h{rm}m{rs:.0f}s")
        print(f"Best so far: XAUC={best_xauc:.4f}  MAE={best_mae:.4f}  WR_XAUC={best_wr_xauc:.4f}  WR_MAE={best_wr_mae:.4f}")

    # ── Final output (parsed by the RecHarness harness) ──────────────────
    print(f"XAUC={best_xauc:.4f}")
    print(f"MAE={best_mae:.4f}")
    print(f"WR_XAUC={best_wr_xauc:.4f}")
    print(f"WR_MAE={best_wr_mae:.4f}")


if __name__ == "__main__":
    main()

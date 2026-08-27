#!/usr/bin/env python3
"""prepare_gr_data.py — Preprocess local KuaiRec data for GR training.

Self-contained pipeline that takes raw KuaiRec files and produces the
train_data.npy / test_data.npy expected by gr.sh.

Usage:
    python prepare_gr_data.py --output-dir ./input --raw-dir /data/kuairec

Options:
    --output-dir DIR     Where to write train_data.npy / test_data.npy  [./input]
    --raw-dir    DIR     Local directory containing the raw CSV files    [required]
    --no-cache           Reprocess even if output arrays already exist
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Normalize raw interaction matrices for the GR input pipeline.
# Converts big_matrix.csv + small_matrix.csv → *_processed.csv
# ---------------------------------------------------------------------------

def normalize_matrices(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize watch_ratio and video_duration columns (max-min method).

    Reads big_matrix.csv and small_matrix.csv from raw_dir.
    Returns (df_big_processed, df_small_processed) with columns:
        item_id, user_id, timestamp, watch_ratio_normed, duration_normed
    """
    big_path   = raw_dir / "big_matrix.csv"
    small_path = raw_dir / "small_matrix.csv"

    print("  Loading big_matrix.csv ...")
    df_big   = pd.read_csv(str(big_path))
    print("  Loading small_matrix.csv ...")
    df_small = pd.read_csv(str(small_path))

    # Normalise column names — original KuaiRec uses 'video_id'; rename to 'item_id'
    for df in (df_big, df_small):
        if "video_id" in df.columns and "item_id" not in df.columns:
            df.rename(columns={"video_id": "item_id"}, inplace=True)
        # Compute watch_ratio if absent but play_duration / video_duration present
        if "watch_ratio" not in df.columns:
            if "play_duration" in df.columns and "video_duration" in df.columns:
                df["watch_ratio"] = df["play_duration"] / df["video_duration"].clip(lower=1)
            else:
                raise ValueError(
                    "big_matrix.csv must contain either 'watch_ratio' or "
                    "both 'play_duration' and 'video_duration' columns."
                )

    # --- Duration normalization (max-min across both matrices) ---
    dur_col = "video_duration" if "video_duration" in df_big.columns else "duration_ms"
    dur_max = max(df_big[dur_col].max(), df_small[dur_col].max())
    dur_min = min(df_big[dur_col].min(), df_small[dur_col].min())
    denom   = dur_max - dur_min if dur_max != dur_min else 1.0
    df_big["duration_normed"]   = (df_big[dur_col]   - dur_min) / denom
    df_small["duration_normed"] = (df_small[dur_col] - dur_min) / denom

    # --- Watch-ratio normalization (per-item max-min) ---
    df_all   = pd.concat([df_big[["item_id", "watch_ratio"]],
                          df_small[["item_id", "watch_ratio"]]], axis=0)
    max_y    = df_all.groupby("item_id")["watch_ratio"].max()
    min_y    = df_all.groupby("item_id")["watch_ratio"].min()
    denom_y  = (max_y - min_y).clip(lower=1e-9)

    def _normed(df):
        item_ids    = df["item_id"].values
        raw_ratio   = df["watch_ratio"].values
        mn          = min_y.reindex(item_ids).values
        dm          = denom_y.reindex(item_ids).values
        normed      = (raw_ratio - mn) / dm
        normed      = np.nan_to_num(normed, nan=0.0)
        return normed

    df_big["watch_ratio_normed"]   = _normed(df_big)
    df_small["watch_ratio_normed"] = _normed(df_small)

    # Carry raw columns alongside normalized ones so the downstream pipeline
    # can report MAE in raw seconds; normalized-space MAE is not comparable.
    # KuaiRec raw matrices store play_duration / video_duration in *milliseconds*.
    # Convert to seconds here to match the unit the paper reports MAE in.
    df_big["watch_ratio_raw"]    = df_big["watch_ratio"].astype(float)
    df_small["watch_ratio_raw"]  = df_small["watch_ratio"].astype(float)
    df_big["video_duration_sec"]   = df_big[dur_col].astype(float)   / 1000.0
    df_small["video_duration_sec"] = df_small[dur_col].astype(float) / 1000.0
    if "play_duration" in df_big.columns:
        df_big["play_duration_sec"]   = df_big["play_duration"].astype(float)   / 1000.0
        df_small["play_duration_sec"] = df_small["play_duration"].astype(float) / 1000.0
    else:
        # No play_duration → reconstruct in seconds via WR * video_dur_sec.
        df_big["play_duration_sec"]   = df_big["watch_ratio_raw"]   * df_big["video_duration_sec"]
        df_small["play_duration_sec"] = df_small["watch_ratio_raw"] * df_small["video_duration_sec"]

    # Keep only the columns needed downstream
    keep = ["user_id", "item_id", "timestamp",
            "watch_ratio_normed", "duration_normed",
            "watch_ratio_raw", "video_duration_sec", "play_duration_sec"]
    # 'timestamp' may be absent in some versions
    keep = [c for c in keep if c in df_big.columns]
    df_big   = df_big[keep].copy()
    df_small = df_small[keep].copy()

    # Save processed files so subsequent runs are fast
    proc_big   = raw_dir / "big_matrix_processed.csv"
    proc_small = raw_dir / "small_matrix_processed.csv"
    df_big.to_csv(str(proc_big),   index=False)
    df_small.to_csv(str(proc_small), index=False)
    print(f"  Saved {proc_big}")
    print(f"  Saved {proc_small}")

    return df_big, df_small


# ---------------------------------------------------------------------------
# Build item_categories.csv from a local multi-category CSV.
# ---------------------------------------------------------------------------

def build_item_categories_from_local(raw_dir: Path) -> Path:
    """Convert video_raw_categories_multi.csv → item_categories.csv format.

    data_process.py expects item_categories.csv with columns:
        item_id (index), feat  (Python-literal list of up to 4 int tags)
    """
    out_path = raw_dir / "item_categories.csv"
    if out_path.exists():
        return out_path

    src = raw_dir / "video_raw_categories_multi.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Provide the local category feature file."
        )

    df = pd.read_csv(str(src))

    # Identify tag columns — typically 'feat0'..'feat3' or 'category_0'..'category_3'
    tag_cols = [c for c in df.columns if c.startswith(("feat", "category"))]
    if not tag_cols:
        # Treat all non-id columns as tag columns
        id_col   = "item_id" if "item_id" in df.columns else "video_id"
        tag_cols = [c for c in df.columns if c != id_col]
    tag_cols = tag_cols[:4]  # cap at 4

    if "item_id" not in df.columns and "video_id" in df.columns:
        df.rename(columns={"video_id": "item_id"}, inplace=True)

    # Encode each tag column: string/numeric → integer via LabelEncoder; NaN → -1
    for col in tag_cols:
        nan_mask = df[col].isna()
        lbe = LabelEncoder()
        df[col] = lbe.fit_transform(df[col].fillna("__NAN__").astype(str))
        df[col] = df[col].where(~nan_mask, -1)   # restore NaN positions → -1
        df.loc[df[col] != -1, col] += 1           # shift: reserve 0 for unknown

    def _row_to_list(row):
        vals = [int(row[c]) for c in tag_cols]
        while len(vals) < 4:
            vals.append(-1)
        return vals

    df["feat"] = df.apply(_row_to_list, axis=1)
    df[["item_id", "feat"]].to_csv(str(out_path), index=False)
    print(f"  Built {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# data_process pipeline (embedded from snailma0229/GR data_process.py)
# ---------------------------------------------------------------------------

ORDINAL_COLS = [
    "user_active_degree", "is_live_streamer", "is_video_author",
    "follow_user_num_range", "fans_user_num_range",
    "friend_user_num_range", "register_days_range",
]
ONEHOT_COLS = [f"onehot_feat{x}" for x in range(18)]


def _label_encode_col(series: pd.Series) -> pd.Series:
    series = series.map(lambda x: chr(0) if x == "UNKNOWN" else x)
    lbe = LabelEncoder()
    encoded = lbe.fit_transform(series)
    unknown_in_classes = (chr(0) in lbe.classes_.tolist()
                          or -124 in lbe.classes_.tolist())
    if not unknown_in_classes:
        encoded = encoded + 1
    return pd.Series(encoded, index=series.index)


def load_item_category_feat(data_raw: Path) -> pd.DataFrame:
    filepath = data_raw / "item_categories.csv"
    df_raw = pd.read_csv(str(filepath), header=0)
    df_raw["feat"] = df_raw["feat"].map(eval)
    df_feat = pd.DataFrame(df_raw["feat"].tolist(),
                           columns=["feat0", "feat1", "feat2", "feat3"])
    df_feat.index.name = "item_id"
    df_feat[df_feat.isna()] = -1
    df_feat = (df_feat + 1).astype(int)
    return df_feat


def load_item_duration(data_raw: Path,
                       df_big: pd.DataFrame | None = None,
                       df_small: pd.DataFrame | None = None) -> pd.Series:
    duration_path = data_raw / "video_duration_normed.csv"
    if duration_path.exists():
        s = pd.read_csv(str(duration_path), header=0)["duration_normed"]
        s.index.name = "item_id"
        return s

    cols = ["item_id", "duration_normed"]
    if df_big is None:
        df_big   = pd.read_csv(str(data_raw / "big_matrix_processed.csv"),   usecols=cols)
    else:
        df_big   = df_big[cols]
    if df_small is None:
        df_small = pd.read_csv(str(data_raw / "small_matrix_processed.csv"), usecols=cols)
    else:
        df_small = df_small[cols]

    combined = pd.concat([df_big, df_small], axis=0)
    vmean    = combined.groupby("item_id")["duration_normed"].mean()
    vmean.to_csv(str(duration_path), index=False)
    vmean.index.name = "item_id"
    return vmean


def load_user_feat(data_raw: Path, user_filter=None) -> pd.DataFrame:
    # Try user_features.csv first, then user_features_raw.csv
    for fname in ("user_features.csv", "user_features_raw.csv"):
        fp = data_raw / fname
        if fp.exists():
            break
    else:
        raise FileNotFoundError(
            "Neither user_features.csv nor user_features_raw.csv found in "
            f"{data_raw}. Please download the KuaiRec user features file."
        )

    needed = ["user_id"] + ORDINAL_COLS + ONEHOT_COLS
    df_user = pd.read_csv(str(fp), usecols=lambda c: c in needed)

    for col in ORDINAL_COLS:
        if col in df_user.columns:
            df_user[col] = _label_encode_col(df_user[col])

    for col in ONEHOT_COLS:
        if col in df_user.columns:
            df_user[col] = df_user[col].fillna(-124)
            df_user[col] = _label_encode_col(df_user[col])
        else:
            df_user[col] = 0

    df_user = df_user.set_index("user_id")

    if user_filter is not None:
        df_user = df_user.reindex(user_filter).fillna(0)
    return df_user


def build_dataset(df_interact: pd.DataFrame,
                  df_feat: pd.DataFrame,
                  df_user: pd.DataFrame,
                  df_item: pd.DataFrame) -> pd.DataFrame:
    df = df_interact.join(df_feat[["feat0", "feat1", "feat2", "feat3"]],
                          on="item_id", how="left")
    df = df.join(
        df_item[["duration_normed"]].rename(
            columns={"duration_normed": "item_duration_normed"}),
        on="item_id", how="left")
    df = df.join(df_user, on="user_id", how="left")
    return df


def _build_npy(df: pd.DataFrame) -> np.ndarray:
    data_sets = []
    for index, data in df.iterrows():
        if index % 50000 == 0:
            print(f"    Processing row {index:,} ...")

        usr_list = data[["user_id"] + ONEHOT_COLS].values.tolist()
        usr_list = [0 if i == 12345 else int(i) + 1 for i in usr_list]
        usr_list[0] = usr_list[0] - 1
        usr_len = len(usr_list)
        assert usr_len == 19

        item_list = [int(data["item_id"]),
                     int(data["feat0"]), int(data["feat1"]),
                     int(data["feat2"]), int(data["feat3"])]
        item_list = item_list[:5]
        item_len  = 1 + sum(1 for v in item_list[1:] if v != 0)
        item_list = item_list + [0] * (5 - len(item_list))

        usr_mask  = [0.0 if v == 0 and idx != 0 else 1.0
                     for idx, v in enumerate(usr_list)]
        item_mask = [1.0] * item_len + [0.0] * (5 - item_len)

        # ── 53-col layout (raw watch-time space, matches paper MAE) ─────
        #   [48] play_duration_sec  — WT MAE GT
        #   [49] watch_ratio_raw    — Huber GT + vocab source (×1000 → token)
        #   [50] video_duration_sec — multiplies pre_WR back to seconds
        #   [51] usr_len, [52] item_len
        play_dur_sec  = float(data.get("play_duration_sec", 0.0))
        watch_ratio   = float(data.get("watch_ratio_raw", 0.0))
        video_dur_sec = float(data.get("video_duration_sec", 0.0))

        row = (usr_list + item_list + usr_mask + item_mask
               + [play_dur_sec, watch_ratio, video_dur_sec,
                  float(usr_len), float(item_len)])
        data_sets.append(row)

    return np.array(data_sets, dtype=float)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args) -> None:
    output_dir = Path(args.output_dir)
    raw_dir    = Path(args.raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_out = output_dir / "train_data.npy"
    test_out  = output_dir / "test_data.npy"

    if train_out.exists() and test_out.exists() and not args.no_cache:
        print("Output files already exist:")
        print(f"  {train_out}")
        print(f"  {test_out}")
        print("Use --no-cache to force re-generation.")
        return

    # ── Step 1: Locate local raw files ───────────────────────────────────
    print("\n=== Step 1: Locate local raw files ===")
    df_big   = None
    df_small = None
    data_raw = raw_dir
    if not data_raw.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {data_raw}")
    print(f"  Using local files in {data_raw}")

    # ── Step 2: Normalization (only if raw matrices present) ─────────────
    big_proc   = data_raw / "big_matrix_processed.csv"
    small_proc = data_raw / "small_matrix_processed.csv"

    if big_proc.exists() and small_proc.exists():
        print("\n=== Step 2: Loading pre-normalised matrices ===")
        keep = ["user_id", "item_id",
                "watch_ratio_normed", "duration_normed",
                "watch_ratio_raw", "video_duration_sec", "play_duration_sec"]
        keep_ts = keep + ["timestamp"]
        try:
            df_big   = pd.read_csv(str(big_proc),   usecols=lambda c: c in keep_ts)
            df_small = pd.read_csv(str(small_proc), usecols=lambda c: c in keep_ts)
        except ValueError:
            df_big   = pd.read_csv(str(big_proc),   usecols=lambda c: c in keep)
            df_small = pd.read_csv(str(small_proc), usecols=lambda c: c in keep)
        # If processed CSV is from an older run that lacks raw cols, recompute
        # from the raw matrices so this script is idempotent across upgrades.
        missing_raw = any(c not in df_big.columns for c in
                          ("watch_ratio_raw", "video_duration_sec", "play_duration_sec"))
        if missing_raw:
            print("  [upgrade] processed CSV missing raw cols → re-normalising")
            df_big, df_small = normalize_matrices(data_raw)
        print(f"  big_matrix_processed:   {len(df_big):,} rows")
        print(f"  small_matrix_processed: {len(df_small):,} rows")
    else:
        print("\n=== Step 2: Normalizing raw matrices ===")
        df_big, df_small = normalize_matrices(data_raw)
        big_proc   = data_raw / "big_matrix_processed.csv"
        small_proc = data_raw / "small_matrix_processed.csv"

    # ── Step 3: Build item_categories.csv if needed ───────────────────────
    item_cat = data_raw / "item_categories.csv"
    if not item_cat.exists() and (data_raw / "video_raw_categories_multi.csv").exists():
        print("\n=== Step 3: Building item_categories.csv ===")
        build_item_categories_from_local(data_raw)
    elif not item_cat.exists():
        raise FileNotFoundError(
            f"item_categories.csv not found in {data_raw}. "
            "Provide item_categories.csv or video_raw_categories_multi.csv locally."
        )

    # ── Step 4: data_process pipeline ─────────────────────────────────────
    print("\n=== Step 4: Data processing ===")

    print("  Loading item category features...")
    df_feat = load_item_category_feat(data_raw)

    print("  Loading item duration features...")
    vmean   = load_item_duration(data_raw, df_big, df_small)
    df_item = df_feat.join(vmean, on="item_id", how="left")

    print("  Loading user features (all users for training)...")
    df_user_train = load_user_feat(data_raw)

    val_users = df_small["user_id"].unique()
    print(f"  Loading user features (test users: {len(val_users):,})...")
    df_user_val = load_user_feat(data_raw, user_filter=val_users)

    print("  Building wide table — training split...")
    df_train = build_dataset(df_big, df_feat, df_user_train, df_item)
    print(f"    shape: {df_train.shape}")

    print("  Building wide table — test split...")
    df_val = build_dataset(df_small, df_feat, df_user_val, df_item)
    print(f"    shape: {df_val.shape}")

    # ── Step 5: Serialize to .npy ─────────────────────────────────────────
    print("\n=== Step 5: Serializing to NumPy ===")

    print("  Converting training split...")
    train_arr = _build_npy(df_train)
    np.save(str(train_out), train_arr)
    print(f"  Saved → {train_out}   shape={train_arr.shape}")

    print("  Converting test split...")
    test_arr = _build_npy(df_val)
    np.save(str(test_out), test_arr)
    print(f"  Saved → {test_out}   shape={test_arr.shape}")

    print("\n=== Done ===")
    print(f"  train_data.npy : {train_out}  ({train_arr.shape[0]:,} × {train_arr.shape[1]})")
    print(f"  test_data.npy  : {test_out}   ({test_arr.shape[0]:,} × {test_arr.shape[1]})")
    print()
    print("Next steps:")
    print(f"  bash gr.sh --train-data {train_out} --test-data {test_out} --gpus 0,1,2,3")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess local KuaiRec data for GR training."
    )
    parser.add_argument("--output-dir",    default="./input",
                        help="Output directory for train_data.npy / test_data.npy  [./input]")
    parser.add_argument("--raw-dir", required=True,
                        help="Local directory containing raw KuaiRec CSV files")
    parser.add_argument("--no-cache",      action="store_true",
                        help="Reprocess even if outputs already exist")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

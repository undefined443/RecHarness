"""Prepare local train/val/test splits for spooky-author-identification.

Two-layer split, matching MLE-Bench Lite's own protocol:

  1. Raw Kaggle train.csv -> public_train (90%) / private_test (10%), via
     train_test_split(test_size=0.1, random_state=0) — same as mlebench's
     own mlebench/competitions/spooky-author-identification/prepare.py.
  2. public_train -> train.csv (80%) / val.csv (20%), stratified by author,
     via train_test_split(test_size=0.2, random_state=42, stratify=author) —
     RecHarness's own search-time validation split.

Trial code (train.py/predict.py) only ever reads train.csv/val.csv/test.csv.
private_test.csv holds the answer key and is only read by the final-incumbent
evaluation step (gagc.benchmarks.spooky_author.harness.evaluate(mode="test")).
"""
from __future__ import annotations

import argparse
import os
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from gagc.benchmarks.spooky_author.task import (
    CLASSES,
    RAW_SPLIT_SEED,
    RAW_SPLIT_TEST_SIZE,
    VAL_SPLIT_SEED,
    VAL_SPLIT_TEST_SIZE,
)

COMPETITION = "spooky-author-identification"


def _ensure_raw_train_csv(raw_dir: str) -> str:
    """Return the path to train.csv under raw_dir, downloading it if missing."""
    train_csv = os.path.join(raw_dir, "train.csv")
    if os.path.isfile(train_csv):
        return train_csv

    # The Kaggle asset is named train.zip (verified via competition_list_files),
    # not train.csv / train.csv.zip.
    train_zip = os.path.join(raw_dir, "train.zip")
    if not os.path.isfile(train_zip):
        os.makedirs(raw_dir, exist_ok=True)
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print(f"[prepare_spooky_data] Downloading {COMPETITION} via Kaggle API to {raw_dir} ...")
        api.competition_download_file(COMPETITION, "train.zip", path=raw_dir)

    if os.path.isfile(train_zip):
        with zipfile.ZipFile(train_zip) as zf:
            zf.extractall(raw_dir)

    if not os.path.isfile(train_csv):
        raise FileNotFoundError(
            f"train.csv not found under {raw_dir} after download. Make sure "
            "~/.kaggle/kaggle.json is configured and you have accepted the "
            f"competition rules at https://www.kaggle.com/c/{COMPETITION}/rules"
        )
    return train_csv


def _one_hot_answers(df: pd.DataFrame) -> pd.DataFrame:
    onehot = pd.DataFrame(0, index=df.index, columns=["id"] + CLASSES)
    onehot["id"] = df["id"].values
    for cls in CLASSES:
        onehot[cls] = (df["author"].values == cls).astype(int)
    return onehot


def run(raw_dir: str, output_dir: str) -> None:
    train_csv = _ensure_raw_train_csv(raw_dir)
    full_df = pd.read_csv(train_csv)

    public_train, private_test_df = train_test_split(
        full_df, test_size=RAW_SPLIT_TEST_SIZE, random_state=RAW_SPLIT_SEED,
    )
    private_answers = _one_hot_answers(private_test_df)

    train_df, val_df = train_test_split(
        public_train, test_size=VAL_SPLIT_TEST_SIZE, random_state=VAL_SPLIT_SEED,
        stratify=public_train["author"],
    )

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    private_test_df.drop(columns=["author"]).to_csv(
        os.path.join(output_dir, "test.csv"), index=False
    )
    private_answers.to_csv(os.path.join(output_dir, "private_test.csv"), index=False)

    print(
        f"[prepare_spooky_data] train={len(train_df)} val={len(val_df)} "
        f"test={len(private_test_df)} -> {output_dir}"
    )
    print(
        "[prepare_spooky_data] author distribution (train): "
        f"{train_df['author'].value_counts().to_dict()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", required=True,
        help="Directory containing (or to download) the raw Kaggle train.csv",
    )
    parser.add_argument(
        "--output-dir", default="./input/spooky_author",
        help="Output directory for train.csv/val.csv/test.csv/private_test.csv",
    )
    args = parser.parse_args()
    run(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()

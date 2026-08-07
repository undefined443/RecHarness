"""Popular — item frequency baseline training script.

Counts item occurrences in the training set and saves a popularity table.
No GPU needed; runs in seconds.

Output:  model.pkl  (pickle of {item_id: count} dict)
"""
from __future__ import annotations

import os
import pickle
from collections import Counter

DATA_DIR  = os.environ.get("GAGC_DATA_DIR",  "./input/trainval")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "model.pkl")

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


def _load_interactions(data_dir: str) -> Counter:
    counts: Counter = Counter()
    for ds in DATASETS:
        for split in ("train", "valid"):
            path = os.path.join(data_dir, f"{ds}_{split}.txt")
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        counts[int(parts[1])] += 1
    return counts


def main():
    counts = _load_interactions(DATA_DIR)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(dict(counts), f)
    print(f"Saved popularity table ({len(counts)} items) → {MODEL_OUT}")
    # Emit a dummy val_score so RecHarness's stdout parser has something to read
    # (real score comes from evaluate_val_fast via execute_trial)
    print("val_score: 0.0")


if __name__ == "__main__":
    main()

"""
compute_dataset_stats.py — Pre-compute per-dataset statistics
==============================================================

Scans train / valid / test split files and writes the global maximum user ID
and item ID for each dataset into ``dataset_stats.json`` in the same directory.

Usage
-----
    python compute_dataset_stats.py \\
        --data_dir /path/to/trainval \\
        --test_dir /path/to/test \\
        --datasets Movies_and_TV Industrial_and_Scientific Electronics CDs_and_Vinyl \\
        --output_dir .   # defaults to the directory of this script

The resulting JSON looks like:

    {
        "Movies_and_TV": {
            "usernum": 123456,
            "itemnum": 78901
        },
        ...
    }

``train.py`` (and similar scripts) can then load this file to get the correct
``itemnum`` without ever touching the test-split files at runtime.
"""

from __future__ import annotations

import argparse
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATASETS = [
    "Movies_and_TV",
    "Industrial_and_Scientific",
    "Electronics",
    "CDs_and_Vinyl",
]


def _scan_file(filepath: str) -> tuple[int, int]:
    """Return (max_user_id, max_item_id) from a split file.

    Each line is expected to be ``<user_id> <item_id>``.
    Returns (0, 0) if the file does not exist.
    """
    max_user, max_item = 0, 0
    if not os.path.isfile(filepath):
        return max_user, max_item
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, i = line.split()
            u, i = int(u), int(i)
            max_user = max(max_user, u)
            max_item = max(max_item, i)
    return max_user, max_item


def compute_stats(
    dataset_name: str,
    data_dir: str,
    test_dir: str,
) -> dict[str, int]:
    """Compute usernum and itemnum across all three splits."""
    splits = {
        "train": os.path.join(data_dir, f"{dataset_name}_train.txt"),
        "valid": os.path.join(data_dir, f"{dataset_name}_valid.txt"),
        "test":  os.path.join(test_dir,  f"{dataset_name}_test.txt"),
    }

    max_user, max_item = 0, 0
    for split_name, path in splits.items():
        if not os.path.isfile(path):
            print(f"  [warn] {split_name} file not found: {path}")
            continue
        u, i = _scan_file(path)
        print(f"  {split_name}: max_user={u}  max_item={i}  ({path})")
        max_user = max(max_user, u)
        max_item = max(max_item, i)

    return {"usernum": max_user, "itemnum": max_item}


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute per-dataset usernum/itemnum and write to dataset_stats.json"
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory containing *_train.txt and *_valid.txt files",
    )
    parser.add_argument(
        "--test_dir", default=None,
        help="Directory containing *_test.txt files (defaults to data_dir)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        help="Dataset names to process",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to write dataset_stats.json (defaults to this script's directory)",
    )
    args = parser.parse_args()

    test_dir   = args.test_dir or args.data_dir
    output_dir = args.output_dir or SCRIPT_DIR
    output_path = os.path.join(output_dir, "dataset_stats.json")

    # Load existing stats if present (so we can update incrementally)
    if os.path.isfile(output_path):
        with open(output_path, "r") as f:
            all_stats = json.load(f)
        print(f"Loaded existing stats from {output_path}")
    else:
        all_stats = {}

    for dataset in args.datasets:
        print(f"\nProcessing '{dataset}' ...")
        stats = compute_stats(dataset, args.data_dir, test_dir)
        all_stats[dataset] = stats
        print(f"  => usernum={stats['usernum']}  itemnum={stats['itemnum']}")

    with open(output_path, "w") as f:
        json.dump(all_stats, f, indent=4)
    print(f"\nStats written to {output_path}")


if __name__ == "__main__":
    main()

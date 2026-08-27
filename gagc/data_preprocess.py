from __future__ import annotations

"""
Standalone local-data preprocessing script for Amazon Reviews datasets.

Usage:
    python data_preprocess.py --dataset <DATASET_NAME> [options]

Example:
    python data_preprocess.py \\
        --dataset Industrial_and_Scientific \\
        --local-dir /path/to/local/raw/data \\
        --data_dir /path/to/output/trainval \\
        --test_dir /path/to/output/test

Supported datasets (subset sampling rates apply):
    - Industrial_and_Scientific  (100%)
    - Movies_and_TV              (5%)
    - Electronics                (5%)
    - CDs_and_Vinyl              (33%)
    - Other Amazon categories    (100%, no subsampling)

Output files:
    <data_dir>/<dataset>_train.txt   -- training interactions  (user_id item_id per line)
    <data_dir>/<dataset>_valid.txt   -- validation interactions
    <test_dir>/<dataset>_test.txt    -- test interactions
    <data_dir>/text_name_dict.json.gz -- item text metadata (title, description, timestamp)
"""

import argparse
import gzip
import json
import os
import pickle
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:  # Only required for Amazon Reviews 2023 CSV preprocessing.
    pd = None

# Default subsampling rates per dataset (to keep experiments manageable)
DEFAULT_SAMPLE_RATE = {
    'Movies_and_TV': 0.05,
    'Electronics': 0.05,
    'Industrial_and_Scientific': 1.0,
    'CDs_and_Vinyl': 0.33,
}

def _filter_user_item_5core(
    by_user: dict[str, list[tuple[int, str]]],
    min_interactions: int = 5,
) -> dict[str, list[tuple[int, str]]]:
    """Iteratively remove users/items with fewer than min_interactions."""
    filtered = {u: list(items) for u, items in by_user.items()}
    while True:
        filtered = {
            u: interactions
            for u, interactions in filtered.items()
            if len(interactions) >= min_interactions
        }
        item_counts: dict[str, int] = defaultdict(int)
        for interactions in filtered.values():
            for _, item in interactions:
                item_counts[item] += 1
        valid_items = {item for item, count in item_counts.items() if count >= min_interactions}
        next_filtered = {
            u: [(ts, item) for ts, item in interactions if item in valid_items]
            for u, interactions in filtered.items()
        }
        next_filtered = {
            u: interactions
            for u, interactions in next_filtered.items()
            if len(interactions) >= min_interactions
        }
        if next_filtered == filtered:
            return next_filtered
        filtered = next_filtered


def _write_leave_one_out_splits(
    fname: str,
    by_user: dict[str, list[tuple[int, str]]],
    data_dir: str,
    test_dir: str,
    min_sequence_len: int = 5,
) -> None:
    usermap: dict[str, int] = {}
    itemmap: dict[str, int] = {}
    splits = {"train": [], "valid": [], "test": []}
    for user in sorted(by_user):
        seq_raw = [item for _, item in sorted(by_user[user])]
        if len(seq_raw) < min_sequence_len:
            continue
        user_id = len(usermap) + 1
        usermap[user] = user_id
        seq: list[int] = []
        for item in seq_raw:
            if item not in itemmap:
                itemmap[item] = len(itemmap) + 1
            seq.append(itemmap[item])
        for item_id in seq[:-2]:
            splits["train"].append((user_id, item_id))
        splits["valid"].append((user_id, seq[-2]))
        splits["test"].append((user_id, seq[-1]))

    if not usermap or not splits["train"]:
        raise RuntimeError(
            f"Amazon 2014 preprocessing for {fname} produced no train users. "
            "Check the source file and sample_ratio."
        )

    for split in ("train", "valid"):
        path = os.path.join(data_dir, f"{fname}_{split}.txt")
        with open(path, "w") as f:
            for user_id, item_id in splits[split]:
                f.write(f"{user_id} {item_id}\n")
        print(f"Wrote {path}: {len(splits[split])} interactions", flush=True)
    path = os.path.join(test_dir, f"{fname}_test.txt")
    with open(path, "w") as f:
        for user_id, item_id in splits["test"]:
            f.write(f"{user_id} {item_id}\n")
    print(f"Wrote {path}: {len(splits['test'])} interactions", flush=True)

    text_dict = {"time": defaultdict(dict), "description": {}, "title": {}}
    with open(os.path.join(data_dir, "text_name_dict.json.gz"), "wb") as tf:
        pickle.dump(text_dict, tf)
    print(f"Final users={len(usermap)} items={len(itemmap)}", flush=True)


def _parse_jsonish(line: str) -> dict:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        import ast
        return ast.literal_eval(line)


def _validate_gzip(path: str) -> None:
    with gzip.open(path, "rb") as f:
        while f.read(1024 * 1024):
            pass


def preprocess_amazon_2014_5core(
    fname: str,
    local_dir: str | None = None,
    data_dir: str | None = None,
    test_dir: str | None = None,
    sample_ratio: float | None = None,
    seed: int = 0,
) -> None:
    """Process a local Amazon Reviews 2014 category gzip into RecHarness splits.

    The raw category file is filtered to user/item 5-core, then split by
    leave-one-out: last interaction -> test, second last -> valid, rest -> train.
    Output dataset names are exactly `fname`, e.g. Beauty and Baby.
    """
    random.seed(seed)
    np.random.seed(seed)
    if data_dir is None:
        data_dir = f"./../data_{fname}"
    if test_dir is None:
        test_dir = data_dir
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    if not local_dir:
        raise ValueError("--local-dir is required for Amazon 2014 preprocessing")
    gz_path = os.path.join(local_dir, f"reviews_{fname}.json.gz")
    if not os.path.isfile(gz_path):
        raise FileNotFoundError(f"Expected local raw file {gz_path}")
    _validate_gzip(gz_path)

    by_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Reading {fname}"):
            if not line.strip():
                continue
            rec = _parse_jsonish(line)
            user = rec.get("reviewerID")
            item = rec.get("asin")
            ts = int(rec.get("unixReviewTime", 0) or 0)
            if user and item:
                by_user[user].append((ts, item))

    by_user = defaultdict(list, _filter_user_item_5core(by_user, min_interactions=5))
    users = sorted(by_user)
    if sample_ratio is not None and 0 < sample_ratio < 1:
        users = random.sample(users, max(1, int(len(users) * sample_ratio)))

    _write_leave_one_out_splits(fname, {u: by_user[u] for u in users}, data_dir, test_dir)


def preprocess_amazon_2014_sasrec(
    fname: str,
    local_dir: str | None = None,
    data_dir: str | None = None,
    test_dir: str | None = None,
    sample_ratio: float | None = None,
    seed: int = 0,
) -> None:
    """Process Amazon Reviews 2014 raw category JSON like SASRec's DataProcessing.py.

    SASRec first counts users/items on the raw category file, then keeps an
    interaction iff its raw user count and raw item count are both at least 5.
    This is a single-pass filter based on raw counts, not iterative k-core.
    The remaining per-user sequences are split with leave-one-out.
    """
    random.seed(seed)
    np.random.seed(seed)
    if data_dir is None:
        data_dir = f"./../data_{fname}"
    if test_dir is None:
        test_dir = data_dir
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    if not local_dir:
        raise ValueError("--local-dir is required for Amazon 2014 preprocessing")
    gz_path = os.path.join(local_dir, f"reviews_{fname}.json.gz")
    if not os.path.isfile(gz_path):
        raise FileNotFoundError(f"Expected local raw file {gz_path}")
    _validate_gzip(gz_path)

    user_counts: dict[str, int] = defaultdict(int)
    item_counts: dict[str, int] = defaultdict(int)
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Counting {fname}"):
            if not line.strip():
                continue
            rec = _parse_jsonish(line)
            user = rec.get("reviewerID")
            item = rec.get("asin")
            if user and item:
                user_counts[user] += 1
                item_counts[item] += 1

    by_user: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Filtering {fname}"):
            if not line.strip():
                continue
            rec = _parse_jsonish(line)
            user = rec.get("reviewerID")
            item = rec.get("asin")
            ts = int(rec.get("unixReviewTime", 0) or 0)
            if not user or not item:
                continue
            if user_counts[user] < 5 or item_counts[item] < 5:
                continue
            by_user[user].append((ts, item))

    users = [u for u, interactions in by_user.items() if len(interactions) >= 3]
    if sample_ratio is not None and 0 < sample_ratio < 1:
        users = random.sample(users, max(1, int(len(users) * sample_ratio)))
    users.sort()
    _write_leave_one_out_splits(
        fname,
        {u: by_user[u] for u in users},
        data_dir,
        test_dir,
        min_sequence_len=3,
    )


def preprocess_raw_5core(
    fname: str,
    local_dir: str | None = None,
    data_dir: str | None = None,
    test_dir: str | None = None,
    sample_ratio: float | None = None,
    seed: int = 0,
):
    """
    Preprocess a local Amazon Reviews 2023 5-core benchmark split.

    Args:
        fname:        Dataset/category name (e.g. 'Industrial_and_Scientific').
        local_dir:    Local dataset root containing split CSVs and metadata.
        data_dir:     Output directory for train/valid files and text metadata.
                      Defaults to './../data_<fname>'.
        test_dir:     Output directory for test file. Defaults to data_dir.
        sample_ratio: Fraction of users to keep (0, 1].  If None, uses the
                      DEFAULT_SAMPLE_RATE table, falling back to 1.0 for
                      unknown datasets.
        seed:         Random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    if data_dir is None:
        data_dir = f'./../data_{fname}'
    if test_dir is None:
        test_dir = data_dir

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    if not local_dir:
        raise ValueError("--local-dir is required for Amazon 2023 preprocessing")

    # ------------------------------------------------------------------ #
    # 1. Load benchmark split CSVs                                        #
    # ------------------------------------------------------------------ #
    print("Loading local 5core_last_out data...")
    dataset = {}
    for split in ['train', 'valid', 'test']:
        path = os.path.join(local_dir, 'benchmark', '5core', 'last_out', f'{fname}.{split}.csv')
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Expected local split file {path}")
        dataset[split] = pd.read_csv(path).to_dict(orient='records')

    # ------------------------------------------------------------------ #
    # 2. Load item metadata                                               #
    # ------------------------------------------------------------------ #
    print("Loading local raw_meta data...")
    meta_path = os.path.join(local_dir, 'raw', 'meta_categories', f'meta_{fname}.jsonl')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Expected local metadata file {meta_path}")
    meta_dict = {}
    with open(meta_path, 'r') as f:
        for line in tqdm(f, desc="Loading metadata"):
            record = json.loads(line.strip())
            meta_dict[record['parent_asin']] = [record['title'], record['description']]

    # ------------------------------------------------------------------ #
    # 3. Build raw user / item maps                                       #
    # ------------------------------------------------------------------ #
    usermap = {}
    usernum = 0
    itemmap = {}
    itemnum = 0
    User = defaultdict(list)
    User_s = {'train': defaultdict(list), 'valid': defaultdict(list), 'test': defaultdict(list)}
    id2asin = {}
    time_dict = defaultdict(dict)

    for t in ['train', 'valid', 'test']:
        for l in tqdm(dataset[t], desc=f"Mapping {t}"):
            user_id = l['user_id']
            asin = l['parent_asin']

            if user_id not in usermap:
                usernum += 1
                usermap[user_id] = usernum
            userid = usermap[user_id]

            if asin not in itemmap:
                itemnum += 1
                itemmap[asin] = itemnum
            itemid = itemmap[asin]

            User[userid].append(itemid)
            User_s[t][userid].append(itemid)
            id2asin[itemid] = asin
            time_dict[itemid][userid] = l['timestamp']

    # ------------------------------------------------------------------ #
    # 4. Subsample users                                                  #
    # ------------------------------------------------------------------ #
    if sample_ratio is None:
        sample_ratio = DEFAULT_SAMPLE_RATE.get(fname, 1.0)

    all_users = list(User.keys())
    use_key = random.sample(all_users, int(len(all_users) * sample_ratio))
    print(f'Total users: {len(all_users)}, sampled: {len(use_key)} (ratio={sample_ratio})')

    use_key_dict = {k: 1 for k in use_key}

    # Count interactions for 5-core filtering
    CountU = defaultdict(int)
    CountI = defaultdict(int)
    for key in use_key:
        for t in ['train', 'valid', 'test']:
            for i_ in User_s[t][key]:
                CountI[i_] += 1
                CountU[key] += 1

    # ------------------------------------------------------------------ #
    # 5. Write output files                                               #
    # ------------------------------------------------------------------ #
    usermap_final = {}
    itemmap_final = {}
    usernum_final = 0
    itemnum_final = 0
    use_train_dict = defaultdict(int)
    text_dict = {'time': defaultdict(dict), 'description': {}, 'title': {}}

    for t in ['train', 'valid', 'test']:
        d = dataset[t]
        use_id = defaultdict(int)
        out_path = os.path.join(test_dir if t == 'test' else data_dir, f'{fname}_{t}.txt')
        written = 0

        with open(out_path, 'w') as f:
            for l in tqdm(d, desc=f"Writing {t}"):
                user_id = l['user_id']
                asin = l['parent_asin']
                user_id_ = usermap[user_id]

                # Each user appears only once per split in the CSV; skip duplicates
                if use_id[user_id_] != 0:
                    continue
                use_id[user_id_] = 1

                if use_key_dict.get(user_id_, 0) != 1 or CountU[user_id_] <= 4:
                    continue

                use_items = [it for it in User_s[t][user_id_] if CountI[it] > 4]

                if t == 'train':
                    if len(use_items) <= 4:
                        continue
                    use_train_dict[user_id_] = 1

                    if user_id_ not in usermap_final:
                        usernum_final += 1
                        usermap_final[user_id_] = usernum_final
                    userid = usermap_final[user_id_]

                    for it in use_items:
                        if it not in itemmap_final:
                            itemnum_final += 1
                            itemmap_final[it] = itemnum_final
                        itemid = itemmap_final[it]

                        desc = meta_dict.get(id2asin[it], [None, None])[1]
                        text_dict['description'][itemid] = (
                            desc[0] if isinstance(desc, list) and len(desc) > 0 else
                            desc if isinstance(desc, str) else 'Empty description'
                        )
                        title = meta_dict.get(id2asin[it], [None, None])[0]
                        text_dict['title'][itemid] = title if title else 'Empty title'
                        text_dict['time'][itemid][userid] = time_dict[it].get(user_id_)

                        f.write(f'{userid} {itemid}\n')
                        written += 1
                else:
                    if use_train_dict.get(user_id_, 0) != 1:
                        continue

                    for it in User_s[t][user_id_]:
                        if CountI[it] <= 4:
                            continue

                        if user_id_ not in usermap_final:
                            usernum_final += 1
                            usermap_final[user_id_] = usernum_final
                        userid = usermap_final[user_id_]

                        if it not in itemmap_final:
                            itemnum_final += 1
                            itemmap_final[it] = itemnum_final
                        itemid = itemmap_final[it]

                        desc = meta_dict.get(id2asin[it], [None, None])[1]
                        text_dict['description'][itemid] = (
                            desc[0] if isinstance(desc, list) and len(desc) > 0 else
                            desc if isinstance(desc, str) else 'Empty description'
                        )
                        title = meta_dict.get(id2asin[it], [None, None])[0]
                        text_dict['title'][itemid] = title if title else 'Empty title'
                        text_dict['time'][itemid][userid] = time_dict[it].get(user_id_)

                        f.write(f'{userid} {itemid}\n')
                        written += 1

        print(f"  -> {out_path}  ({written} lines)")

    meta_out = os.path.join(data_dir, f'{fname}_text_name_dict.json.gz')
    with open(meta_out, 'wb') as tf:
        pickle.dump(text_dict, tf)
    print(f"  -> {meta_out}")

    print(f"\nDone. Final users: {usernum_final}, final items: {itemnum_final}")


# --------------------------------------------------------------------------- #
# CLI entry point                                                              #
# --------------------------------------------------------------------------- #
def _parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess Amazon Reviews 2023 5-core data for SASRec training.'
    )
    parser.add_argument('--dataset', required=True,
                        help='Dataset/category name, e.g. Industrial_and_Scientific')
    parser.add_argument('--source', choices=['2023', '2014', '2014-sasrec'], default='2023',
                        help='Amazon Reviews source version (default: 2023)')
    parser.add_argument('--local-dir', '--hf_local_dir', dest='local_dir', required=True,
                        help='Local raw dataset root; no network download is performed')
    parser.add_argument('--data_dir', default=None,
                        help='Output directory for train/valid files (default: ./../data_<dataset>)')
    parser.add_argument('--test_dir', default=None,
                        help='Output directory for test file (default: same as data_dir)')
    parser.add_argument('--sample_ratio', default=None, type=float,
                        help='Fraction of users to keep [0,1]. Overrides built-in defaults.')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed (default: 0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    data_dir = args.data_dir or f'./../data_{args.dataset}'
    test_dir = args.test_dir or data_dir

    if args.source == '2014-sasrec':
        preprocess_amazon_2014_sasrec(
            fname=args.dataset,
            local_dir=args.local_dir,
            data_dir=data_dir,
            test_dir=test_dir,
            sample_ratio=args.sample_ratio,
            seed=args.seed,
        )
    elif args.source == '2014':
        preprocess_amazon_2014_5core(
            fname=args.dataset,
            local_dir=args.local_dir,
            data_dir=data_dir,
            test_dir=test_dir,
            sample_ratio=args.sample_ratio,
            seed=args.seed,
        )
    else:
        preprocess_raw_5core(
            fname=args.dataset,
            local_dir=args.local_dir,
            data_dir=data_dir,
            test_dir=test_dir,
            sample_ratio=args.sample_ratio,
            seed=args.seed,
        )

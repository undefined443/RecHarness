"""Metric computation for KuaiRec watch time prediction evaluation."""
from __future__ import annotations

import math
import numpy as np


def xauc_score(labels: np.ndarray, preds: np.ndarray) -> float:
    """Compute xAUC: fraction of concordant pairs in predicted ranking."""
    label_preds = list(zip(labels.reshape(-1), preds.reshape(-1)))
    sorted_lp = sorted(label_preds, key=lambda x: x[1], reverse=True)
    n = len(sorted_lp)
    pairs_cnt = n * (n - 1) / 2

    labels_sort = [e[0] for e in sorted_lp]
    total_positive = _inverse_pairs(labels_sort)
    return total_positive / pairs_cnt if pairs_cnt > 0 else 0.0


def _inverse_pairs(data: list) -> int:
    if len(data) <= 1:
        return 0

    def merge(left, right):
        arr, cnt = [], left[1] + right[1]
        la, ra = left[0], right[0]
        i, j = len(la) - 1, len(ra) - 1
        while i >= 0 and j >= 0:
            if la[i] > ra[j]:
                arr.append(la[i])
                cnt += j + 1
                i -= 1
            else:
                arr.append(ra[j])
                j -= 1
        while i >= 0:
            arr.append(la[i]); i -= 1
        while j >= 0:
            arr.append(ra[j]); j -= 1
        return arr[::-1], cnt

    def mergesort(a):
        if len(a) == 1:
            return a, 0
        mid = len(a) // 2
        return merge(mergesort(a[:mid]), mergesort(a[mid:]))

    return mergesort(data)[1]


def compute_metrics(preds: np.ndarray, gts: np.ndarray) -> dict[str, float]:
    """Compute all metrics for a batch of predictions vs ground truths."""
    preds = preds.reshape(-1)
    gts = gts.reshape(-1)
    return {
        "xAUC": round(float(xauc_score(gts, preds)), 4),
        "MAE":  round(float(np.mean(np.abs(preds - gts))), 4),
    }

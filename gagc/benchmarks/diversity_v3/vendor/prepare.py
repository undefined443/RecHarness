"""
prepare.py — V2 评估管线 (READ-ONLY, Agent 不可修改)

功能:
  1. 加载数据 (sample + vec)
  2. 调用 train.py 的 scatter() 函数获取打散结果
  3. 计算三大标准 (V2: 2+5+3=10项指标 + 通过率)
  4. 输出结构化评估结果到 eval_results.json

V2 评估标准:
  标准1: C3类目去重数 (含前序曝光商品) — 辅助评估
    - Cat3Diversity_top4  = |unique(pre_goods_cat ∪ result_cat[:4])|
    - Cat3Diversity_top10 = |unique(pre_goods_cat ∪ result_cat[:10])|

  标准2: 向量相似度 (5个子指标, 按请求维度评估, 60%通过率)
    相似度 = (max(dot(img_i,img_j), dot(text_i,text_j)) + 1) / 2, 值域 [0, 1]
    对每个坑位 i, 计算 result[i] 与前序已曝光商品 (pre_goods + result[0..i-1]) 的相似度:
    - VecSim_max:              每坑位最大相似度 → 跨坑位均值
    - VecSim_mean:             每坑位平均相似度 → 跨坑位均值
    - VecSim_weighted_mean:    每坑位曝光率加权平均相似度 → 跨坑位均值
    - VecSim_max07_mean:       每坑位 Max(0.7, sim) 的均值 → 跨坑位均值
    - VecSim_max07_weighted_mean: 每坑位 Max(0.7, sim) 的加权均值 → 跨坑位均值
    - VecSim_pass_rate: 请求所有子指标 ≤ baseline 的比例, 需 ≥ 60%

  标准3: rank×曝光概率 (仅新出商品, 不含前序)
    - RankValue_top4    = sum(result_fst_score[i] * exposure_prob[i], i=0..3)
    - RankValue_top10   = sum(result_score[i] * exposure_prob[i], i=0..9)
    - RankValue_bottom6 = sum(result_score[i] * exposure_prob[i], i=4..9)

  n_pre 分组: 按前序商品数量拆分数据, 分开计算各组指标

约束:
  - 标准1和标准3不下降的前提下，标准2通过率 ≥ 60%
  - V2输入约束: 算法仅可使用 rank, fst_rank, 向量emb (不可使用cat1/cat3)
  - 前序曝光商品计入标准1和标准3的计算

用法:
  python prepare.py                          # 调用 train.py 的 scatter() 评估
  python prepare.py --verify-baseline        # 评估线上 result_goods (验证评估逻辑)
  python prepare.py --workers 16             # 指定多进程并行数 (Linux fork COW)
"""

import json
import sys
import os
import subprocess
import platform
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# =====================================================================
#  不可修改的常量
# =====================================================================

PROJECT_DIR = Path(__file__).resolve().parent

EXPOSURE_PROBS = [1.0, 1.0, 0.418337, 0.418337, 0.167033,
                   0.167033, 0.140998, 0.140998, 0.122709, 0.122709]

VEC_DIM = 128

VECSIM_KEYS = [
    'VecSim_max', 'VecSim_mean', 'VecSim_weighted_mean',
    'VecSim_max07_mean', 'VecSim_max07_weighted_mean'
]

VECSIM_BASELINE_KEYS = [
    'std2_vecsim_max_mean', 'std2_vecsim_mean_mean', 'std2_vecsim_weighted_mean_mean',
    'std2_vecsim_max07_mean_mean', 'std2_vecsim_max07_weighted_mean_mean'
]

_IS_LINUX = platform.system() == 'Linux'


# =====================================================================
#  数据加载
# =====================================================================

def load_config():
    with open(PROJECT_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(config):
    import pyarrow.dataset as ds

    sample_path = PROJECT_DIR / config['data']['sample_path']
    vec_path = PROJECT_DIR / config['data']['vec_path']

    # 读取采样日志 (自动识别 Parquet / ORC)
    if str(sample_path).endswith('.parquet') and sample_path.is_file():
        df = pd.read_parquet(sample_path)
    else:
        fmt = "orc" if str(sample_path).endswith('.orc') or sample_path.is_dir() else "parquet"
        df = ds.dataset(str(sample_path), format=fmt).to_table().to_pandas()

    # 收集请求中出现的 goods_id
    all_goods = set()
    for goods_list in df['mix_rank_goods']:
        all_goods.update(goods_list)
    for goods_list in df['pre_goods']:
        if goods_list is not None and len(goods_list) > 0:
            all_goods.update(goods_list)
    if 'result_goods' in df.columns:
        for goods_list in df['result_goods']:
            if goods_list is not None and len(goods_list) > 0:
                all_goods.update(goods_list)

    # 读取向量数据 (自动识别 Parquet / ORC, 过滤所需的商品)
    if str(vec_path).endswith('.parquet') and vec_path.is_file():
        vec_df = pd.read_parquet(vec_path)
        vec_df = vec_df[vec_df['goods_id'].astype(str).isin(all_goods)]
    else:
        fmt = "orc" if str(vec_path).endswith('.orc') or vec_path.is_dir() else "parquet"
        vec_dataset = ds.dataset(str(vec_path), format=fmt)
        try:
            goods_ids_int = [int(g) for g in all_goods]
            scanner = vec_dataset.scanner(filter=ds.field("goods_id").isin(goods_ids_int))
            vec_df = scanner.to_table().to_pandas()
        except (ValueError, TypeError):
            vec_df = vec_dataset.to_table().to_pandas()
        vec_df = vec_df[vec_df['goods_id'].astype(str).isin(list(all_goods))]
    vec_df['goods_id'] = vec_df['goods_id'].astype(str)

    # 构建 vec_lookup: 数组格式 (内存高效, 查询快)
    goods_ids = vec_df['goods_id'].tolist()
    img_col = vec_df['img_vec'].tolist()
    text_col = vec_df['text_vec'].tolist()

    # 一次性 stack 所有向量 (比逐个 np.array 慢)
    img_vecs = np.stack([np.array(v, dtype=np.float32) for v in img_col])
    text_vecs = np.stack([np.array(v, dtype=np.float32) for v in text_col])
    goods_id_to_idx = {gid: i for i, gid in enumerate(goods_ids)}

    vec_lookup = {
        'goods_id_to_idx': goods_id_to_idx,
        'img_vecs': img_vecs,
        'text_vecs': text_vecs,
    }

    return df, vec_lookup


# =====================================================================
#  请求构建
# =====================================================================

def build_request(row):
    """从 DataFrame 行构建请求 dict"""
    goods_list = list(row['mix_rank_goods'])
    cat1_list = list(row['mix_rank_cat1'])
    cat3_list = list(row['mix_rank_cat'])
    score_list = [float(x) for x in row['mix_rank_score']]
    fst_score_list = [float(x) for x in row['mix_rank_fst_score']]

    # zip 替代逐个 append (C层实现, 更快)
    candidates = [
        {'goods_id': g, 'cat1': c1, 'cat3': c3, 'score': s, 'fst_score': fs}
        for g, c1, c3, s, fs in zip(goods_list, cat1_list, cat3_list, score_list, fst_score_list)
    ]

    pre_goods_list = list(row['pre_goods'])
    pre_cat1_list = list(row['pre_goods_cat1'])
    pre_cat3_list = list(row['pre_goods_cat'])
    pre_goods = [
        {'goods_id': g, 'cat1': c1, 'cat3': c3}
        for g, c1, c3 in zip(pre_goods_list, pre_cat1_list, pre_cat3_list)
    ]

    return {
        'search_id': row['search_id'],
        'candidates': candidates,
        'pre_goods': pre_goods,
    }


def lookup_result_info(result_goods_ids, candidates):
    """从候选列表中查找结果商品的类目和分数"""
    cand_map = {c['goods_id']: c for c in candidates}
    result_cats = []
    result_scores = []
    result_fst_scores = []
    for gid in result_goods_ids:
        if gid not in cand_map:
            raise ValueError(f"Goods {gid} not found in candidates")
        c = cand_map[gid]
        result_cats.append(c['cat3'])
        result_scores.append(c['score'])
        result_fst_scores.append(c['fst_score'])
    return result_cats, result_scores, result_fst_scores


# =====================================================================
#  V2 评估指标 (Agent 不可修改)
# =====================================================================

def compute_Cat3Diversity(pre_cats, result_cats, k):
    """标准1: C3类目去重数 (含前序曝光)"""
    pre_set = set(pre_cats)
    result_set = set(result_cats[:k])
    return len(pre_set | result_set)


def compute_VecSim_metrics(result_goods_ids, pre_goods_ids, vec_lookup):
    """V2 标准2: 向量相似度 (5个子指标)

    对每个坑位 i, 计算 result[i] 与前序已曝光商品 (pre_goods + result[0..i-1]) 的相似度:
      sim = (max(dot(img_i, img_j), dot(text_i, text_j)) + 1) / 2, 值域 [0, 1]

    5个子指标 (每个坑位计算, 跨坑位取均值):
      1. VecSim_max:              大相似度
      2. VecSim_mean:             平均相似度
      3. VecSim_weighted_mean:    曝光概率加权平均相似度
      4. VecSim_max07_mean:       Max(0.7, sim) 的均值
      5. VecSim_max07_weighted_mean: Max(0.7, sim) 的加权均值

    权重: pre_goods=1.0, result[j]=exposure_probs[j]
    """
    goods_id_to_idx = vec_lookup['goods_id_to_idx']
    img_vecs = vec_lookup['img_vecs']
    text_vecs = vec_lookup['text_vecs']

    n_pre = len(pre_goods_ids)
    n_result = len(result_goods_ids)  # should be 10

    # Get indices for all items: pre_goods + result
    all_ids = list(pre_goods_ids) + list(result_goods_ids)
    all_indices = [goods_id_to_idx[gid] for gid in all_ids]

    # Stack vectors: (n_pre + n_result, 128)
    all_img = img_vecs[all_indices]
    all_text = text_vecs[all_indices]

    # Compute full similarity matrix: (n_pre + n_result, n_pre + n_result)
    img_sim_matrix = all_img @ all_img.T
    text_sim_matrix = all_text @ all_text.T
    sim_matrix = np.maximum(img_sim_matrix, text_sim_matrix)
    sim_matrix = (sim_matrix + 1.0) / 2.0  # Map to [0, 1]

    sub1_vals = []  # max
    sub2_vals = []  # mean
    sub3_vals = []  # weighted mean
    sub4_vals = []  # mean of Max(0.7, sim)
    sub5_vals = []  # weighted mean of Max(0.7, sim)

    for pit in range(n_result):
        # Exposed items: pre_goods (indices 0..n_pre-1) + result[0..pit-1] (indices n_pre..n_pre+pit-1)
        n_exposed = n_pre + pit
        if n_exposed == 0:
            continue  # No exposed items, skip this pit

        # Similarity of result[pit] (index n_pre + pit) to exposed items (indices 0..n_exposed-1)
        result_idx = n_pre + pit
        sims = sim_matrix[result_idx, :n_exposed]  # (n_exposed,)

        # Weights: pre_goods = 1.0, result[j] = exposure_probs[j]
        weights = np.ones(n_exposed, dtype=np.float64)
        if pit > 0:
            weights[n_pre:] = EXPOSURE_PROBS[:pit]

        weight_sum = np.sum(weights)

        # Sub-metrics
        sub1_vals.append(float(np.max(sims)))
        sub2_vals.append(float(np.mean(sims)))
        sub3_vals.append(float(np.sum(sims * weights) / weight_sum))

        capped_sims = np.maximum(0.7, sims)
        sub4_vals.append(float(np.mean(capped_sims)))
        sub5_vals.append(float(np.sum(capped_sims * weights) / weight_sum))

    # Average across pits (only pits with at least 1 exposed item)
    return {
        'VecSim_max': float(np.mean(sub1_vals)),
        'VecSim_mean': float(np.mean(sub2_vals)),
        'VecSim_weighted_mean': float(np.mean(sub3_vals)),
        'VecSim_max07_mean': float(np.mean(sub4_vals)),
        'VecSim_max07_weighted_mean': float(np.mean(sub5_vals)),
    }


def compute_RankValue_top4(fst_scores):
    """标准3: 前4坑rank×曝光概率"""
    return sum(fst_scores[i] * EXPOSURE_PROBS[i] for i in range(min(4, len(fst_scores))))


def compute_RankValue_top10(scores):
    """标准3: 全10坑rank×曝光概率"""
    return sum(scores[i] * EXPOSURE_PROBS[i] for i in range(min(10, len(scores))))


def compute_RankValue_bottom6(scores):
    """标准3: 后6坑rank×曝光概率"""
    return sum(scores[i] * EXPOSURE_PROBS[i] for i in range(4, min(10, len(scores))))


# =====================================================================
#  单请求评估 (供单线程和多进程共用)
# =====================================================================

def _process_single_request(idx, row, vec_lookup, config, mode, scatter_fn=None):
    """处理单个请求, 返回 (result_dict, error_str_or_None)"""
    request = build_request(row)
    pre_goods_ids = [g['goods_id'] for g in request['pre_goods']]
    pre_cats = [g['cat3'] for g in request['pre_goods']]
    n_pre = len(pre_goods_ids)

    if mode == 'scatter':
        try:
            result_goods_ids = scatter_fn(request, vec_lookup, config)
        except Exception as e:
            return None, f"Row {idx} (search_id={request['search_id'][:20]}): scatter() failed: {e}"
        if len(result_goods_ids) != 10:
            return None, f"Row {idx}: expected 10 items, got {len(result_goods_ids)}"
        try:
            result_cats, result_scores, result_fst_scores = lookup_result_info(
                result_goods_ids, request['candidates']
            )
        except ValueError as e:
            return None, f"Row {idx}: {e}"
    elif mode == 'baseline':
        result_goods_ids = list(row['result_goods'])
        if len(result_goods_ids) != 10:
            return None, f"Row {idx}: expected 10 items, got {len(result_goods_ids)}"
        result_cats = list(row['result_cat'])
        result_scores = [float(x) for x in row['result_score']]
        result_fst_scores = [float(x) for x in row['result_fst_score']]
    else:
        return None, f"Row {idx}: unknown mode {mode}"

    goods_id_to_idx = vec_lookup['goods_id_to_idx']
    missing_vecs = [gid for gid in pre_goods_ids + result_goods_ids if gid not in goods_id_to_idx]
    if missing_vecs:
        return None, f"Row {idx}: missing vectors for {len(missing_vecs)} goods"

    # 标准1: C3类目去重
    Cat3Diversity_top4 = compute_Cat3Diversity(pre_cats, result_cats, 4)
    Cat3Diversity_top10 = compute_Cat3Diversity(pre_cats, result_cats, 10)

    # 标准2: 向量相似度 (5个子指标)
    vecsim = compute_VecSim_metrics(result_goods_ids, pre_goods_ids, vec_lookup)

    # 标准3: rank×曝光概率
    RankValue_top4 = compute_RankValue_top4(result_fst_scores)
    RankValue_top10 = compute_RankValue_top10(result_scores)
    RankValue_bottom6 = compute_RankValue_bottom6(result_scores)

    result = {
        'search_id': request['search_id'],
        'n_pre': n_pre,
        'Cat3Diversity_top4': Cat3Diversity_top4,
        'Cat3Diversity_top10': Cat3Diversity_top10,
        'VecSim_max': vecsim['VecSim_max'],
        'VecSim_mean': vecsim['VecSim_mean'],
        'VecSim_weighted_mean': vecsim['VecSim_weighted_mean'],
        'VecSim_max07_mean': vecsim['VecSim_max07_mean'],
        'VecSim_max07_weighted_mean': vecsim['VecSim_max07_weighted_mean'],
        'RankValue_top4': RankValue_top4,
        'RankValue_top10': RankValue_top10,
        'RankValue_bottom6': RankValue_bottom6,
    }

    return result, None


# =====================================================================
#  多进程支持 (Linux fork COW, 共享 vec_lookup)
# =====================================================================

_worker_data = None    # (vec_lookup, config, mode) 赋值于 fork 前, 靠 COW 共享
_worker_df = None      # DataFrame 赋值于 fork 前, 靠 COW 共享


def _process_chunk_worker(chunk_range):
    """多进程 worker v2: 使用全局 df, 支持 scatter 和 baseline 模式"""
    global _worker_data, _worker_df
    vec_lookup, config, mode = _worker_data

    scatter_fn = None
    if mode == 'scatter':
        from train import scatter as scatter_fn

    start, end = chunk_range
    results = []
    errors = []
    for idx in range(start, end):
        row = _worker_df.iloc[idx]
        result, error = _process_single_request(idx, row, vec_lookup, config, mode, scatter_fn)
        if error:
            errors.append(error)
        else:
            results.append(result)
    return results, errors


# =====================================================================
#  评估主流程
# =====================================================================

def _run_eval(df, vec_lookup, config, mode, num_workers):
    """运行评估核心逻辑, 返回 (results, errors)"""
    global _worker_data, _worker_df

    results = []
    errors = []

    use_multiprocess = (num_workers > 1 and _IS_LINUX)

    if use_multiprocess:
        import multiprocessing
        _worker_data = (vec_lookup, config, mode)
        _worker_df = df

        n = len(df)
        chunk_size = max(1, (n + num_workers - 1) // num_workers)
        chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

        ctx = multiprocessing.get_context('fork')
        print(f"  [prepare] Running {mode} with {num_workers} workers, {len(chunks)} chunks, {n} requests")
        with ctx.Pool(num_workers) as pool:
            chunk_results = pool.map(_process_chunk_worker, chunks)

        for r, e in chunk_results:
            results.extend(r)
            errors.extend(e)
    else:
        scatter_fn = None
        if mode == 'scatter':
            from train import scatter as scatter_fn
        for idx, row in df.iterrows():
            result, error = _process_single_request(idx, row, vec_lookup, config, mode, scatter_fn)
            if error:
                errors.append(error)
            else:
                results.append(result)

    return results, errors


def run_evaluation(mode='scatter', num_workers=0):
    """
    运行完整评估流程

    Args:
        mode: 'scatter' 调用 train.py 的 scatter() 函数
              'baseline' 评估线上 result_goods (验证评估逻辑)
        num_workers: 并行进程数 (0=自动, Linux用CPU数, Windows用1)

    Returns:
        (metrics_dict, results_list, errors_list, n_pre_metrics)
    """
    config = load_config()
    df, vec_lookup = load_data(config)

    if num_workers <= 0:
        cpu = os.cpu_count() or 1
        num_workers = min(cpu, 32) if _IS_LINUX else 1

    results, errors = _run_eval(df, vec_lookup, config, mode, num_workers)

    if errors:
        print(f"\n  WARNING: {len(errors)} errors encountered:", file=sys.stderr)
        for e in errors[:10]:
            print(f"    {e}", file=sys.stderr)
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more", file=sys.stderr)

    if not results:
        raise RuntimeError("No valid results. All requests failed.")

    # Run baseline for per-request RankValue pass rate (only in scatter mode)
    baseline_results = []
    if mode == 'scatter':
        print(f"  [prepare] Running baseline for per-request comparison...")
        baseline_results, _ = _run_eval(df, vec_lookup, config, 'baseline', num_workers)

    bl_map = {r['search_id']: r for r in baseline_results}

    # Per-request RankValue pass rate (scatter vs same-request baseline)
    matched_pairs = [(r, bl_map.get(r['search_id'])) for r in results]
    matched_pairs = [(s, b) for s, b in matched_pairs if b is not None]

    if matched_pairs:
        n_matched = len(matched_pairs)
        rv_t4_pass_rate = sum(1 for s, b in matched_pairs if s['RankValue_top4'] >= b['RankValue_top4']) / n_matched
        rv_b6_pass_rate = sum(1 for s, b in matched_pairs if s['RankValue_bottom6'] >= b['RankValue_bottom6']) / n_matched
    else:
        rv_t4_pass_rate = 0.0
        rv_b6_pass_rate = 0.0

    # 计算 VecSim pass rate (标准2通过率, per-request vs 线上 baseline
    if matched_pairs:
        vecsim_good_count = sum(
            1 for s, b in matched_pairs
            if all(s.get(vk, 0.0) <= b.get(vk, 1.0) for vk in VECSIM_KEYS)
        )
        vecsim_pass_rate = vecsim_good_count / len(matched_pairs)
    else:
        vecsim_pass_rate = 1.0

    # 计算整体指标
    all_metric_keys = [
        'Cat3Diversity_top4', 'Cat3Diversity_top10',
        'VecSim_max', 'VecSim_mean', 'VecSim_weighted_mean',
        'VecSim_max07_mean', 'VecSim_max07_weighted_mean',
        'RankValue_top4', 'RankValue_top10', 'RankValue_bottom6',
    ]

    metrics = {}
    for key in all_metric_keys:
        vals = [r[key] for r in results]
        metrics[key] = {
            'mean': float(np.mean(vals)),
            'median': float(np.median(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'std': float(np.std(vals)),
        }

    # VecSim pass rate (特殊指标, 不是均值是通过率)
    metrics['VecSim_pass_rate'] = {
        'mean': vecsim_pass_rate,
        'median': vecsim_pass_rate,
        'min': vecsim_pass_rate,
        'max': vecsim_pass_rate,
        'std': 0.0,
    }

    # RankValue per-request pass rate (scatter vs baseline)
    metrics['RankValue_top4_pass_rate'] = {
        'mean': rv_t4_pass_rate,
        'median': rv_t4_pass_rate,
        'min': rv_t4_pass_rate,
        'max': rv_t4_pass_rate,
        'std': 0.0,
    }
    metrics['RankValue_bottom6_pass_rate'] = {
        'mean': rv_b6_pass_rate,
        'median': rv_b6_pass_rate,
        'min': rv_b6_pass_rate,
        'max': rv_b6_pass_rate,
        'std': 0.0,
    }

    # Combined pass rate: per-request 同时通过 VecSim + RV t4 + RV b6 的比例
    if matched_pairs:
        combined_pass_count = sum(
            1 for s, b in matched_pairs
            if all(s.get(vk, 0.0) <= b.get(vk, 1.0) for vk in VECSIM_KEYS)
            and s['RankValue_top4'] >= b['RankValue_top4']
            and s['RankValue_bottom6'] >= b['RankValue_bottom6']
        )
        combined_pass_rate = combined_pass_count / len(matched_pairs)
    else:
        combined_pass_rate = 0.0

    metrics['combined_pass_rate'] = {
        'mean': combined_pass_rate,
        'median': combined_pass_rate,
        'min': combined_pass_rate,
        'max': combined_pass_rate,
        'std': 0.0,
    }

    # ===================================================================
    #  诊断指标: 列联表分析 (帮 agent 理解 combined 低的根因)
    # ===================================================================
    diag_metrics = {}
    contingency_table_str = ""
    if matched_pairs and len(matched_pairs) > 0:
        n_matched = len(matched_pairs)
        from collections import Counter as _Counter

        # Per-request pass/fail + margins for 3 groups: VecSim(all 5 sub), RV_top4, RV_b6
        vs_pass_list = []; rt4_pass_list = []; rb6_pass_list = []
        n_pass_7_list = []; per_req_fail_keys = []
        vecsim_sub_pass = {vk: [] for vk in VECSIM_KEYS}

        # Per-cell margin storage: cell_key -> {metric -> [margins]}
        cell_margins = {}
        for vp in [True, False]:
            for tp in [True, False]:
                for bp in [True, False]:
                    cell_margins[(vp, tp, bp)] = {'VS': [], 'T4': [], 'B6': []}

        for s, b in matched_pairs:
            vs_sub_passes = {vk: s.get(vk, 0.0) <= b.get(vk, 1.0) for vk in VECSIM_KEYS}
            vs_pass = all(vs_sub_passes.values())
            vs_pass_list.append(vs_pass)
            for vk in VECSIM_KEYS:
                vecsim_sub_pass[vk].append(vs_sub_passes[vk])

            rt4_pass = s['RankValue_top4'] >= b['RankValue_top4']
            rb6_pass = s['RankValue_bottom6'] >= b['RankValue_bottom6']
            rt4_pass_list.append(rt4_pass)
            rb6_pass_list.append(rb6_pass)

            n_pass = sum(vs_sub_passes.values()) + (1 if rt4_pass else 0) + (1 if rb6_pass else 0)
            n_pass_7_list.append(n_pass)

            failed = [vk for vk in VECSIM_KEYS if not vs_sub_passes[vk]]
            if not rt4_pass: failed.append('RV_top4')
            if not rb6_pass: failed.append('RV_b6')
            per_req_fail_keys.append(failed)

            # Margins per cell
            cell_key = (vs_pass, rt4_pass, rb6_pass)
            if not vs_pass:
                max_rel = 0.0
                for vk in VECSIM_KEYS:
                    if not vs_sub_passes[vk]:
                        rel = (s.get(vk, 0.0) - b.get(vk, 0.0)) / max(abs(b.get(vk, 0.0)), 1e-8) * 100
                        if rel > max_rel: max_rel = rel
                cell_margins[cell_key]['VS'].append(max_rel)
            if not rt4_pass:
                cell_margins[cell_key]['T4'].append(
                    (b['RankValue_top4'] - s['RankValue_top4']) / max(abs(b['RankValue_top4']), 1e-8) * 100)
            if not rb6_pass:
                cell_margins[cell_key]['B6'].append(
                    (b['RankValue_bottom6'] - s['RankValue_bottom6']) / max(abs(b['RankValue_bottom6']), 1e-8) * 100)

        # Scalar diagnostic metrics (for branch table + reward)
        p_vs = sum(vs_pass_list) / n_matched
        p_t4 = sum(rt4_pass_list) / n_matched
        p_b6 = sum(rb6_pass_list) / n_matched

        wasted_count = sum(1 for i in range(n_matched) if vs_pass_list[i] and not rt4_pass_list[i] and not rb6_pass_list[i])
        rv_only_count = sum(1 for i in range(n_matched) if not vs_pass_list[i] and rt4_pass_list[i] and rb6_pass_list[i])
        expected_indep = p_vs * p_t4 * p_b6
        correlation_ratio = combined_pass_rate / expected_indep if expected_indep > 1e-12 else 0.0

        max_pr = max(p_vs, p_t4, p_b6)
        min_pr = min(p_vs, p_t4, p_b6)
        balance_score = 1.0 - (max_pr - min_pr) / max_pr if max_pr > 1e-8 else 0.0

        pass_6of7_count = sum(1 for n in n_pass_7_list if n == 6)
        pass_6of7_rate = pass_6of7_count / n_matched

        six7_fail_counter = _Counter()
        for i in range(n_matched):
            if n_pass_7_list[i] == 6:
                for fk in per_req_fail_keys[i]:
                    six7_fail_counter[fk] += 1
        dominant_6of7 = six7_fail_counter.most_common(1)[0][0] if six7_fail_counter else "none"

        vecsim_fail_counter = _Counter()
        for i in range(n_matched):
            if not vs_pass_list[i]:
                for fk in per_req_fail_keys[i]:
                    if fk in VECSIM_KEYS:
                        vecsim_fail_counter[fk] += 1
        vecsim_bottleneck = vecsim_fail_counter.most_common(1)[0][0] if vecsim_fail_counter else "none"

        # All fail margins for p50
        all_vs_margins = [m for cell in cell_margins.values() for m in cell['VS']]
        all_t4_margins = [m for cell in cell_margins.values() for m in cell['T4']]
        all_b6_margins = [m for cell in cell_margins.values() for m in cell['B6']]
        rv_b6_fail_margin_p50 = float(np.median(all_b6_margins)) if all_b6_margins else 0.0
        vecsim_fail_margin_p50 = float(np.median(all_vs_margins)) if all_vs_margins else 0.0

        # Store scalar metrics
        diag_metrics = {
            'wasted_diversity_rate': {'mean': wasted_count / n_matched, 'median': wasted_count / n_matched,
                                       'min': wasted_count / n_matched, 'max': wasted_count / n_matched, 'std': 0.0},
            'rv_only_rate': {'mean': rv_only_count / n_matched, 'median': rv_only_count / n_matched,
                             'min': rv_only_count / n_matched, 'max': rv_only_count / n_matched, 'std': 0.0},
            'correlation_ratio': {'mean': correlation_ratio, 'median': correlation_ratio,
                                   'min': correlation_ratio, 'max': correlation_ratio, 'std': 0.0},
            'balance_score': {'mean': balance_score, 'median': balance_score,
                              'min': balance_score, 'max': balance_score, 'std': 0.0},
            'pass_6of7_rate': {'mean': pass_6of7_rate, 'median': pass_6of7_rate,
                               'min': pass_6of7_rate, 'max': pass_6of7_rate, 'std': 0.0},
            'dominant_6of7_failure': {'mean': dominant_6of7, 'median': dominant_6of7,
                                       'min': dominant_6of7, 'max': dominant_6of7, 'std': 0.0},
            'vecsim_bottleneck_submetric': {'mean': vecsim_bottleneck, 'median': vecsim_bottleneck,
                                              'min': vecsim_bottleneck, 'max': vecsim_bottleneck, 'std': 0.0},
            'rv_b6_fail_margin_p50': {'mean': rv_b6_fail_margin_p50, 'median': rv_b6_fail_margin_p50,
                                       'min': rv_b6_fail_margin_p50, 'max': rv_b6_fail_margin_p50, 'std': 0.0},
            'vecsim_fail_margin_p50': {'mean': vecsim_fail_margin_p50, 'median': vecsim_fail_margin_p50,
                                        'min': vecsim_fail_margin_p50, 'max': vecsim_fail_margin_p50, 'std': 0.0},
        }

        # Build contingency table string
        cell_labels = {
            (True, True, True):   ("all pass", ""),
            (True, True, False):  ("only RV_bottom6 fail", ""),
            (True, False, True):   ("only RV_top4 fail", ""),
            (True, False, False):  ("VecSim pass, RV fail", "wasted diversity"),
            (False, True, True):   ("only VecSim fail", "RV only"),
            (False, True, False):  ("VecSim+RV_bottom6 fail", ""),
            (False, False, True):  ("VecSim+RV_top4 fail", ""),
            (False, False, False): ("all fail", ""),
        }
        cell_counts = _Counter()
        for i in range(n_matched):
            cell_counts[(vs_pass_list[i], rt4_pass_list[i], rb6_pass_list[i])] += 1

        lines = ["Pass/Fail Contingency (VecSim, RV_top4, RV_bottom6):", ""]
        lines.append(f"{'Cell':<28s} {'Count':>5s} {'%':>5s}  {'VecSim gap':>12s} {'RV_top4 gap':>12s} {'RV_b6 gap':>12s}  Note")
        lines.append("-" * 100)
        for cell_key in [(True,True,True), (True,True,False), (True,False,True), (True,False,False),
                          (False,True,True), (False,True,False), (False,False,True), (False,False,False)]:
            label, note = cell_labels[cell_key]
            cnt = cell_counts.get(cell_key, 0)
            pct = cnt / n_matched
            vp, tp, bp = cell_key
            vs_str = "PASS" if vp else (f"med {np.median(cell_margins[cell_key]['VS']):.2f}%" if cell_margins[cell_key]['VS'] else "")
            t4_str = "PASS" if tp else (f"med {np.median(cell_margins[cell_key]['T4']):.2f}%" if cell_margins[cell_key]['T4'] else "")
            b6_str = "PASS" if bp else (f"med {np.median(cell_margins[cell_key]['B6']):.2f}%" if cell_margins[cell_key]['B6'] else "")
            note_str = f"  <- {note}" if note else ""
            lines.append(f"{label:<28s} {cnt:>5d} {pct:>4.1%}  {vs_str:>12s} {t4_str:>12s} {b6_str:>12s}{note_str}")

        lines.append("")
        lines.append(f"Summary: wasted={wasted_count/n_matched:.1%} rv_only={rv_only_count/n_matched:.1%} "
                      f"corr={correlation_ratio:.2f}x balance={balance_score:.2f} 6/7={pass_6of7_rate:.1%} "
                      f"bottleneck={dominant_6of7} vs_sub={vecsim_bottleneck}")
        contingency_table_str = "\n".join(lines)

    metrics.update(diag_metrics)

    # Store contingency table string as a metric so it flows through the pipeline
    if contingency_table_str:
        metrics['contingency_table'] = {
            'mean': contingency_table_str, 'median': contingency_table_str,
            'min': contingency_table_str, 'max': contingency_table_str, 'std': 0.0,
        }

    # n_pre 分组指标
    n_pre_groups = {}
    for r in results:
        n_pre = r['n_pre']
        if n_pre not in n_pre_groups:
            n_pre_groups[n_pre] = []
        n_pre_groups[n_pre].append(r)

    n_pre_metrics = {}
    for n_pre, group_results in sorted(n_pre_groups.items()):
        group_metrics = {'count': len(group_results)}
        for key in all_metric_keys:
            vals = [r[key] for r in group_results]
            group_metrics[key + '_mean'] = float(np.mean(vals))
        # Group per-request pass rates (VecSim + RV t4 + RV b6 + combined)
        group_pairs = [(r, bl_map.get(r['search_id'])) for r in group_results]
        group_pairs = [(s, b) for s, b in group_pairs if b is not None]
        if group_pairs:
            gc = len(group_pairs)
            group_metrics['VecSim_pass_rate'] = sum(
                1 for s, b in group_pairs
                if all(s.get(vk, 0.0) <= b.get(vk, 1.0) for vk in VECSIM_KEYS)
            ) / gc
            group_metrics['RankValue_top4_pass_rate'] = sum(1 for s, b in group_pairs if s['RankValue_top4'] >= b['RankValue_top4']) / gc
            group_metrics['RankValue_bottom6_pass_rate'] = sum(1 for s, b in group_pairs if s['RankValue_bottom6'] >= b['RankValue_bottom6']) / gc
            group_metrics['combined_pass_rate'] = sum(
                1 for s, b in group_pairs
                if all(s.get(vk, 0.0) <= b.get(vk, 1.0) for vk in VECSIM_KEYS)
                and s['RankValue_top4'] >= b['RankValue_top4']
                and s['RankValue_bottom6'] >= b['RankValue_bottom6']
            ) / gc
        else:
            group_metrics['VecSim_pass_rate'] = 0.0
            group_metrics['RankValue_top4_pass_rate'] = 0.0
            group_metrics['RankValue_bottom6_pass_rate'] = 0.0
            group_metrics['combined_pass_rate'] = 0.0
        n_pre_metrics[str(n_pre)] = group_metrics

    return metrics, results, errors, n_pre_metrics, contingency_table_str


def _format_n_pre_table(n_pre_metrics):
    """格式化 n_pre 分组完整指标表 (markdown 格式, 同时用于控制台输出和文件保存)"""
    lines = []
    header = (
        "| n_pre | count | "
        "Cat3Div t4 | Cat3Div t10 | "
        "VecSim max | VecSim mean | VecSim wmean | VecSim m07 | VecSim m07w | "
        "pass% | "
        "RankVal t4 | RankVal t10 | RankVal b6 | "
        "RV t4% | RV b6% |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    for n_pre, gm in sorted(n_pre_metrics.items(), key=lambda x: int(x[0])):
        row = (
            f"| {n_pre} | {gm['count']} | "
            f"{gm.get('Cat3Diversity_top4_mean', 0):.2f} | "
            f"{gm.get('Cat3Diversity_top10_mean', 0):.2f} | "
            f"{gm.get('VecSim_max_mean', 0):.6f} | "
            f"{gm.get('VecSim_mean_mean', 0):.6f} | "
            f"{gm.get('VecSim_weighted_mean_mean', 0):.6f} | "
            f"{gm.get('VecSim_max07_mean_mean', 0):.6f} | "
            f"{gm.get('VecSim_max07_weighted_mean_mean', 0):.6f} | "
            f"{gm.get('VecSim_pass_rate', 0):.2%} | "
            f"{gm.get('RankValue_top4_mean', 0):.2f} | "
            f"{gm.get('RankValue_top10_mean', 0):.2f} | "
            f"{gm.get('RankValue_bottom6_mean', 0):.2f} | "
            f"{gm.get('RankValue_top4_pass_rate', 0):.2%} | "
            f"{gm.get('RankValue_bottom6_pass_rate', 0):.2%} |"
        )
        lines.append(row)

    return lines


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V2 Evaluation pipeline (prepare.py)")
    parser.add_argument('--verify-baseline', action='store_true',
                        help='Evaluate online result_goods instead of calling scatter()')
    parser.add_argument('--output', default='eval_results.json',
                        help='Output file path (default: eval_results.json)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers (0=auto: Linux=CPU count, Windows=1)')
    args = parser.parse_args()

    mode = 'baseline' if args.verify_baseline else 'scatter'
    print(f"\n=== prepare.py — V2 Evaluation Pipeline (mode={mode}, workers={args.workers}) ===\n")

    commit_hash = ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, encoding="utf-8"
        )
        commit_hash = result.stdout.strip()
    except Exception:
        pass

    print(f"Loading data...")
    metrics, results, errors, n_pre_metrics, contingency_table_str = run_evaluation(mode=mode, num_workers=args.workers)

    report = {
        'status': 'done',
        'version': 'v2',
        'commit': commit_hash[:7] if commit_hash else 'unknown',
        'full_commit': commit_hash,
        'mode': mode,
        'num_requests': len(results),
        'num_errors': len(errors),
        'metrics': metrics,
        'contingency_table': contingency_table_str,
        'n_pre_groups': n_pre_metrics,
        'n_pre_table': '\n'.join(_format_n_pre_table(n_pre_metrics)),
        'per_request': results,
    }

    output_path = PROJECT_DIR / args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== V2 Evaluation Results ===")
    print(f"  Requests: {len(results)} (errors: {len(errors)})")
    print(f"  Commit: {commit_hash[:7] if commit_hash else 'unknown'}")
    print()
    print(f"  --- 标准1: C3类目多样性 (辅助) ---")
    for key in ['Cat3Diversity_top4', 'Cat3Diversity_top10']:
        m = metrics[key]
        print(f"  {key:30s}: mean={m['mean']:12.4f}  median={m['median']:12.4f}")
    print()
    print(f"  --- 标准2: 向量相似度 ---")
    for key in VECSIM_KEYS:
        m = metrics[key]
        print(f"  {key:30s}: mean={m['mean']:12.6f}")
    m = metrics['VecSim_pass_rate']
    print(f"  {'VecSim_pass_rate':30s}: {m['mean']:.4f} (threshold: 0.60)")
    print()
    print(f"  --- 标准3: rank价值 ---")
    for key in ['RankValue_top4', 'RankValue_top10', 'RankValue_bottom6']:
        m = metrics[key]
        print(f"  {key:30s}: mean={m['mean']:12.4f}  median={m['median']:12.4f}")
    print(f"  {'RankValue_top4_pass_rate':30s}: {metrics['RankValue_top4_pass_rate']['mean']:.2%} (per-request scatter >= baseline)")
    print(f"  {'RankValue_bottom6_pass_rate':30s}: {metrics['RankValue_bottom6_pass_rate']['mean']:.2%} (per-request scatter >= baseline)")
    print(f"  {'combined_pass_rate':30s}: {metrics['combined_pass_rate']['mean']:.2%} (VecSim+RVt4+RVb6 all pass)")
    print()
    if contingency_table_str:
        print(f"  --- 列联表分析 (反相关诊断) ---")
        for line in contingency_table_str.split('\n'):
            print(f"  {line}")
        print()
    print(f"  --- n_pre 分组 (完整指标) ---")
    n_pre_table_lines = _format_n_pre_table(n_pre_metrics)
    for line in n_pre_table_lines:
        print(line)

    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()

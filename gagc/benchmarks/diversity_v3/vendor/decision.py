"""
decision.py — diversity_v3 (打散算法 V3 DPP) 任务决策逻辑

主指标: combined_pass_rate（per-request VecSim+RVt4+RVb6 同时通过的比例）
KEEP/REVERT 逻辑:
  1. 硬门控: mean 指标不低于 online baseline 的 95%
     - Cat3Diversity_top4/top10_mean >= baseline * 0.95
     - RankValue_top4/bottom6_mean >= baseline * 0.95
  2. VecSim mean 不恶化: <= branch_best * 1.05 (lower is better, 5% 容差)
  3. pass rate 不大幅下降: >= branch_best * 0.85 (VecSim, RV_top4, RV_bottom6, 15% 容差)
  4. combined_pass_rate 不大幅下降: >= branch_best * 0.85 (最重要)
"""

MEAN_BASELINE_TOLERANCE = 0.05   # mean 指标允许低于 online baseline 5%
VECSIM_MEAN_TOLERANCE = 0.05     # VecSim mean 允许恶化 5%
PASS_RATE_TOLERANCE = 0.85       # pass rate 允许下降到 branch_best 的 85%

VECSIM_PASS_RATE_THRESHOLD = 0.60
RANKVALUE_PASS_RATE_THRESHOLD = 0.60
COMBINED_PASS_RATE_THRESHOLD = 0.60

METRIC_KEYS = [
    "Cat3Diversity_top4", "Cat3Diversity_top10",
    "VecSim_max", "VecSim_mean", "VecSim_weighted_mean",
    "VecSim_max07_mean", "VecSim_max07_weighted_mean",
    "VecSim_pass_rate",
    "RankValue_top4", "RankValue_top10", "RankValue_bottom6",
    "RankValue_top4_pass_rate", "RankValue_bottom6_pass_rate",
    "combined_pass_rate",
    "wasted_diversity_rate", "rv_only_rate", "correlation_ratio",
    "balance_score", "pass_6of7_rate",
    "dominant_6of7_failure", "vecsim_bottleneck_submetric",
    "rv_b6_fail_margin_p50", "vecsim_fail_margin_p50",
    "contingency_table",
]

METRIC_FIELDS = [k + "_mean" for k in METRIC_KEYS]

# mean 指标: 不低于 online baseline * (1 - tolerance)
MEAN_BASELINE_KEYS = [
    "Cat3Diversity_top4_mean",
    "Cat3Diversity_top10_mean",
    "RankValue_top4_mean",
    "RankValue_bottom6_mean",
]

# VecSim mean: 越低越好, 不恶化超过 tolerance
VECSIM_MEAN_KEYS = [
    "VecSim_max_mean", "VecSim_mean_mean", "VecSim_weighted_mean_mean",
    "VecSim_max07_mean_mean", "VecSim_max07_weighted_mean_mean",
]

# pass rate: 不低于 branch_best * tolerance
PASS_RATE_KEYS = [
    "VecSim_pass_rate_mean",
    "RankValue_top4_pass_rate_mean",
    "RankValue_bottom6_pass_rate_mean",
]

ONLINE_BASELINE = {
    "Cat3Diversity_top4_mean": 6.4381,
    "Cat3Diversity_top10_mean": 12.1431,
    "VecSim_max_mean": 0.622743,
    "VecSim_mean_mean": 0.544072,
    "VecSim_weighted_mean_mean": 0.546157,
    "VecSim_max07_mean_mean": 0.702037,
    "VecSim_max07_weighted_mean_mean": 0.702397,
    "VecSim_pass_rate_mean": 1.0,
    "RankValue_top4_mean": 318.8872,
    "RankValue_top10_mean": 1421.8636,
    "RankValue_bottom6_mean": 185.4970,
    "RankValue_top4_pass_rate_mean": 1.0,
    "RankValue_bottom6_pass_rate_mean": 1.0,
    "combined_pass_rate_mean": 1.0,
}


def extract_metrics(result: dict) -> dict:
    """从原始评估结果提取指标字典。
    eval_results.json 嵌套结构: result["metrics"]["Cat3Diversity_top4"]["mean"]
    展平为: {"Cat3Diversity_top4_mean": 6.51, ...}
    """
    metrics = {}
    raw = result.get("metrics", result)
    for key in METRIC_KEYS:
        val = raw.get(key, {})
        if isinstance(val, dict):
            mv = val.get("mean", 0.0)
        else:
            mv = val
        if mv is None:
            mv = 0.0
        if isinstance(mv, str):
            metrics[key + "_mean"] = mv
        else:
            metrics[key + "_mean"] = float(mv)
    return metrics


def compute_primary_metric(metrics: dict) -> float:
    """主指标值（用于 branch best 比较）"""
    return metrics.get("combined_pass_rate_mean", 0.0)


def get_primary_metric_name() -> str:
    return "combined_pass_rate_mean"


def get_primary_metric_value(metrics: dict) -> float:
    return metrics.get("combined_pass_rate_mean", 0.0)


def get_display_metric_name() -> str:
    """返回 tree topology 中显示的指标名称"""
    return "VS_pass, RV_t4/b6_pass, combined_pass"


def format_node_tag(node: dict) -> str | None:
    """返回节点在 tree topology 中的显示标签。None 表示使用 loop_controller 默认格式。
    - mean 指标跌 >5% baseline → REVERT(mean<95%)，不显示 pass rate
    - 否则 → 显示 VS/RV/Comb 三个 pass rate（3位小数）
    """
    status = node.get("status", "?")
    if status == "crashed":
        return "crashed"

    # pending / no metrics yet
    cpr = node.get("combined_pass_rate_mean", 0.0)
    if cpr <= 0 and node.get("VecSim_pass_rate_mean", 0.0) <= 0:
        return status

    # 检查 Cat3Diversity top4/top10 是否跌 >5% baseline
    bl = ONLINE_BASELINE
    for key in ["Cat3Diversity_top4_mean", "Cat3Diversity_top10_mean"]:
        val = node.get(key, 0.0)
        if val < bl.get(key, 0.0) * (1.0 - MEAN_BASELINE_TOLERANCE):
            return "REVERT"

    # mean OK，显示 pass rate 指标
    vs = node.get("VecSim_pass_rate_mean", 0.0)
    rvt4 = node.get("RankValue_top4_pass_rate_mean", 0.0)
    rvb6 = node.get("RankValue_bottom6_pass_rate_mean", 0.0)
    cpr = node.get("combined_pass_rate_mean", 0.0)

    if cpr <= 0 and vs <= 0:
        return status

    return f"{status} VS={vs:.3f} RV={rvt4:.3f}/{rvb6:.3f} Comb={cpr:.3f}"


def filter_global_best_candidates(nodes: list, root_id: str | None = None) -> list:
    """过滤全局最优候选节点。diversity_v2: 仅 done + is_best + 排除 root。"""
    return [n for n in nodes
            if n.get("status") == "done" and n.get("is_best") and n["id"] != root_id]


def format_branch_table(branches: list, active_branch: str, online_baseline: dict | None = None) -> list:
    """格式化分支摘要表格。展示 mean + pass rate, 各自有 ✓/✗。"""
    bl = online_baseline or {}
    lines = [""]
    lines.append("| Branch | Cat3Div t4/t10 | VecSim pass% | RV t4/b6 mean | RV t4/b6 pass% | Comb pass% | Wasted% | Corr | Bal | 6/7% | Bottleneck | Pass | Exp | Last |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    if bl:
        lines.append(
            f"| **online baseline** | "
            f"{bl.get('Cat3Diversity_top4_mean', 0):.1f}/{bl.get('Cat3Diversity_top10_mean', 0):.1f} | "
            f"{bl.get('VecSim_pass_rate_mean', 0):.2%} | "
            f"{bl.get('RankValue_top4_mean', 0):.0f}/{bl.get('RankValue_bottom6_mean', 0):.0f} | "
            f"— | "
            f"{bl.get('combined_pass_rate_mean', 0):.2%} | "
            f"— | — | — |"
        )
    for b in sorted(branches, key=lambda x: (x.get("best_primary", 0)), reverse=True):
        if b["name"] == "baseline":
            continue
        marker = " (active)" if b["name"] == active_branch else ""
        m = b.get("metrics", {})
        if not m and b.get("has_pending"):
            lines.append(f"| {b['name']}{marker} | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | — | {b['num_experiments']} | R{b['last_round']} |")
            continue
        if not m:
            lines.append(f"| {b['name']}{marker} | — | — | — | — | — | — | — | — | — | — | — | 0 | — |")
            continue
        c3t4 = "✓" if m.get("Cat3Diversity_top4_mean", 0) >= bl.get("Cat3Diversity_top4_mean", 0) else "✗"
        c3t10 = "✓" if m.get("Cat3Diversity_top10_mean", 0) >= bl.get("Cat3Diversity_top10_mean", 0) else "✗"
        rvt4_mean = m.get("RankValue_top4_mean", 0)
        rvb6_mean = m.get("RankValue_bottom6_mean", 0)
        rvt4_m = "✓" if rvt4_mean >= bl.get("RankValue_top4_mean", 0) else "✗"
        rvb6_m = "✓" if rvb6_mean >= bl.get("RankValue_bottom6_mean", 0) else "✗"
        rvt4_pass = m.get("RankValue_top4_pass_rate_mean", 0)
        rvb6_pass = m.get("RankValue_bottom6_pass_rate_mean", 0)
        rvt4_p = "✓" if rvt4_pass >= RANKVALUE_PASS_RATE_THRESHOLD else "✗"
        rvb6_p = "✓" if rvb6_pass >= RANKVALUE_PASS_RATE_THRESHOLD else "✗"
        vspr = "✓" if m.get("VecSim_pass_rate_mean", 0) >= VECSIM_PASS_RATE_THRESHOLD else "✗"
        cpr = m.get("combined_pass_rate_mean", 0)
        cpr_mark = "✓" if cpr >= COMBINED_PASS_RATE_THRESHOLD else "✗"
        cat3_str = f"{m.get('Cat3Diversity_top4_mean', 0):.1f}{c3t4}/{m.get('Cat3Diversity_top10_mean', 0):.1f}{c3t10}"
        vecsim_str = f"{m.get('VecSim_pass_rate_mean', 0):.2%}{vspr}"
        rv_mean_str = f"{rvt4_mean:.0f}{rvt4_m}/{rvb6_mean:.0f}{rvb6_m}"
        rv_pass_str = f"{rvt4_pass:.1%}{rvt4_p}/{rvb6_pass:.1%}{rvb6_p}"
        comb_str = f"{cpr:.1%}{cpr_mark}"
        wasted = m.get("wasted_diversity_rate_mean", 0.0)
        corr_val = m.get("correlation_ratio_mean", 0.0)
        bal = m.get("balance_score_mean", 0.0)
        p6 = m.get("pass_6of7_rate_mean", 0.0)
        dom6 = m.get("dominant_6of7_failure_mean", "?")
        wasted_str = f"{wasted:.0%}"
        corr_str = f"{corr_val:.2f}x"
        bal_str = f"{bal:.2f}"
        p6_str = f"{p6:.0%}"
        pc = sum([c3t4 == "✓", c3t10 == "✓", rvt4_m == "✓", rvb6_m == "✓", rvt4_p == "✓", rvb6_p == "✓", vspr == "✓", cpr_mark == "✓"])
        lines.append(f"| {b['name']}{marker} | {cat3_str} | {vecsim_str} | {rv_mean_str} | {rv_pass_str} | {comb_str} | {wasted_str} | {corr_str} | {bal_str} | {p6_str} | {dom6} | {pc}/8 | {b['num_experiments']} | R{b['last_round']} |")
    return lines


def compute_pass_count(metrics: dict, online_baseline: dict | None = None, branch_best: dict | None = None) -> int:
    """计算 8 项终极目标通过数（用于显示，vs 60% 目标阈值）。
    mean 指标 vs online_baseline, pass_rate 指标 vs 60%, combined_pass_rate vs 60%。
    """
    bl = online_baseline or ONLINE_BASELINE
    checks = [
        metrics.get("Cat3Diversity_top4_mean", 0) >= bl.get("Cat3Diversity_top4_mean", 0),
        metrics.get("Cat3Diversity_top10_mean", 0) >= bl.get("Cat3Diversity_top10_mean", 0),
        metrics.get("RankValue_top4_mean", 0) >= bl.get("RankValue_top4_mean", 0),
        metrics.get("RankValue_bottom6_mean", 0) >= bl.get("RankValue_bottom6_mean", 0),
        metrics.get("RankValue_top4_pass_rate_mean", 0) >= RANKVALUE_PASS_RATE_THRESHOLD,
        metrics.get("RankValue_bottom6_pass_rate_mean", 0) >= RANKVALUE_PASS_RATE_THRESHOLD,
        metrics.get("VecSim_pass_rate_mean", 0) >= VECSIM_PASS_RATE_THRESHOLD,
        metrics.get("combined_pass_rate_mean", 0) >= COMBINED_PASS_RATE_THRESHOLD,
    ]
    return sum(checks)


def decide_keep(new_metrics: dict, branch_best: dict | None, state: dict) -> tuple[bool, str]:
    """KEEP/REVERT 决策:
    1. 硬门控: mean 指标不低于 online baseline * 0.95
    2. VecSim mean 不恶化: <= branch_best * 1.05
    3. pass rate 不大幅下降: >= branch_best * 0.85
    4. combined_pass_rate 不大幅下降: >= branch_best * 0.85
    """
    if not branch_best:
        return True, "first experiment in branch"

    bl = ONLINE_BASELINE

    # 1. 硬门控: mean 指标不低于 online baseline * (1 - tolerance)
    for key in MEAN_BASELINE_KEYS:
        new_val = new_metrics.get(key, 0.0)
        bl_val = bl.get(key, 0.0)
        threshold = bl_val * (1.0 - MEAN_BASELINE_TOLERANCE)
        if new_val < threshold:
            return False, f"{key} {new_val:.4f} < baseline*{1-MEAN_BASELINE_TOLERANCE:.0%} ({threshold:.4f})"

    # 2. VecSim mean 不恶化 (lower is better)
    for key in VECSIM_MEAN_KEYS:
        new_val = new_metrics.get(key, 0.0)
        best_val = branch_best.get(key, 0.0)
        threshold = best_val * (1.0 + VECSIM_MEAN_TOLERANCE)
        if new_val > threshold:
            return False, f"{key} worsened ({best_val:.6f} -> {new_val:.6f}, threshold {threshold:.6f})"

    # 3. pass rate 不大幅下降
    for key in PASS_RATE_KEYS:
        new_val = new_metrics.get(key, 0.0)
        best_val = branch_best.get(key, 0.0)
        threshold = best_val * PASS_RATE_TOLERANCE
        if new_val < threshold:
            return False, f"{key} decreased ({best_val:.4f} -> {new_val:.4f}, threshold {threshold:.4f})"

    # 4. combined_pass_rate 不大幅下降 (最重要)
    new_cpr = new_metrics.get("combined_pass_rate_mean", 0.0)
    best_cpr = branch_best.get("combined_pass_rate_mean", 0.0)
    cpr_threshold = best_cpr * PASS_RATE_TOLERANCE
    if new_cpr < cpr_threshold:
        return False, f"combined_pass_rate decreased ({best_cpr:.4f} -> {new_cpr:.4f}, threshold {cpr_threshold:.4f})"

    return True, "kept: mean within 5% of baseline, VecSim stable, pass rates stable, combined stable"


def is_new_best(new_metrics: dict, branch_best: dict | None) -> bool:
    """判断是否为新最优: primary = combined_pass_rate"""
    if not branch_best:
        return True
    return new_metrics.get("combined_pass_rate_mean", 0.0) > branch_best.get("combined_pass_rate_mean", 0.0)


def generate_result_analysis(new_metrics: dict, best_metrics: dict, status: str) -> str:
    """生成结果分析文本"""
    if status.startswith("crash"):
        return "CRASHED. Must revert and try a different approach."

    parts = []
    bl = ONLINE_BASELINE

    # mean 指标 vs online baseline (5% tolerance)
    for key in MEAN_BASELINE_KEYS:
        new_val = new_metrics.get(key, 0.0)
        bl_val = bl.get(key, 0.0)
        ratio = new_val / max(bl_val, 1e-8)
        gate = "PASS" if new_val >= bl_val * (1.0 - MEAN_BASELINE_TOLERANCE) else "FAIL"
        parts.append(f"{key} {new_val:.4f} (baseline {bl_val:.4f}, ratio {ratio:.2%}, gate: {gate})")

    # VecSim mean vs branch_best (5% tolerance, lower is better)
    for key in VECSIM_MEAN_KEYS:
        new_val = new_metrics.get(key, 0.0)
        old_val = best_metrics.get(key, 0.0) if best_metrics else 0.0
        gate = "PASS" if not best_metrics or new_val <= old_val * (1.0 + VECSIM_MEAN_TOLERANCE) else "FAIL"
        parts.append(f"{key} {new_val:.6f} (best {old_val:.6f}, gate: {gate})")

    # pass rate vs branch_best (15% tolerance)
    for key in PASS_RATE_KEYS:
        new_val = new_metrics.get(key, 0.0)
        old_val = best_metrics.get(key, 0.0) if best_metrics else 0.0
        gate = "PASS" if not best_metrics or new_val >= old_val * PASS_RATE_TOLERANCE else "FAIL"
        parts.append(f"{key} {new_val:.4f} (best {old_val:.4f}, gate: {gate})")

    # combined_pass_rate (最重要)
    cpr = new_metrics.get("combined_pass_rate_mean", 0.0)
    best_cpr = best_metrics.get("combined_pass_rate_mean", 0.0) if best_metrics else 0.0
    gate = "PASS" if not best_metrics or cpr >= best_cpr * PASS_RATE_TOLERANCE else "FAIL"
    parts.append(f"combined_pass_rate {cpr:.4f} (best {best_cpr:.4f}, gate: {gate}) [PRIMARY]")

    # 列联表分析 (反相关诊断)
    ct = new_metrics.get("contingency_table_mean", "")
    if ct and isinstance(ct, str) and len(ct) > 10:
        parts.append(f"\n{ct}")

    if best_metrics:
        _, reason = decide_keep(new_metrics, best_metrics, {})
        parts.append(f"Verdict: {reason}")
    else:
        parts.append("Verdict: KEEP — first experiment (baseline)")

    return "; ".join(parts)


def format_metrics_line(exp: dict) -> str:
    """格式化单个实验的指标为一行（用于 prompt 显示）"""
    c3t4 = exp.get("Cat3Diversity_top4_mean")
    c3t10 = exp.get("Cat3Diversity_top10_mean")
    vspr = exp.get("VecSim_pass_rate_mean")
    rvt4_mean = exp.get("RankValue_top4_mean")
    rvb6_mean = exp.get("RankValue_bottom6_mean")
    rvt4_pass = exp.get("RankValue_top4_pass_rate_mean")
    rvb6_pass = exp.get("RankValue_bottom6_pass_rate_mean")
    cpr = exp.get("combined_pass_rate_mean")

    # 诊断指标
    wasted = exp.get("wasted_diversity_rate_mean")
    corr = exp.get("correlation_ratio_mean")
    balance = exp.get("balance_score_mean")
    p6 = exp.get("pass_6of7_rate_mean")
    dom6 = exp.get("dominant_6of7_failure_mean")
    vs_bottleneck = exp.get("vecsim_bottleneck_submetric_mean")
    rvb6_margin = exp.get("rv_b6_fail_margin_p50_mean")
    vs_margin = exp.get("vecsim_fail_margin_p50_mean")

    def _fmt(v, prec=4):
        return f"{v:.{prec}f}" if v is not None and isinstance(v, (int, float)) else (str(v) if v is not None else "N/A")

    return (f"Cat3Div={_fmt(c3t4)}/{_fmt(c3t10)}, "
            f"VecSim_pass={_fmt(vspr, 4)}, "
            f"RankVal={_fmt(rvt4_mean)}/{_fmt(rvb6_mean)}, "
            f"RankVal_pass={_fmt(rvt4_pass, 4)}/{_fmt(rvb6_pass, 4)}, "
            f"Comb_pass={_fmt(cpr, 4)}, "
            f"Wasted={_fmt(wasted, 4)}, Corr={_fmt(corr, 2)}x, Bal={_fmt(balance, 4)}, "
            f"6/7={_fmt(p6, 4)}, Bottleneck={dom6}, VS_sub={vs_bottleneck}, "
            f"RVb6_gap={_fmt(rvb6_margin, 2)}%, VS_gap={_fmt(vs_margin, 3)}%")


def format_best_metrics(state: dict) -> str:
    """格式化最佳指标（用于 prompt 显示）"""
    return (
        f"- Best Cat3Diversity: top4={state.get('best_Cat3Diversity_top4_mean', 0.0):.4f}, "
        f"top10={state.get('best_Cat3Diversity_top10_mean', 0.0):.4f}\n"
        f"- Best VecSim: pass_rate={state.get('best_VecSim_pass_rate_mean', 0.0):.4f}, "
        f"max={state.get('best_VecSim_max_mean', 0.0):.6f}, "
        f"mean={state.get('best_VecSim_mean_mean', 0.0):.6f}, "
        f"weighted_mean={state.get('best_VecSim_weighted_mean_mean', 0.0):.6f}, "
        f"max07_mean={state.get('best_VecSim_max07_mean_mean', 0.0):.6f}, "
        f"max07_weighted_mean={state.get('best_VecSim_max07_weighted_mean_mean', 0.0):.6f}\n"
        f"- Best RankValue: top4={state.get('best_RankValue_top4_mean', 0.0):.4f}, "
        f"bottom6={state.get('best_RankValue_bottom6_mean', 0.0):.4f}, "
        f"top4_pass={state.get('best_RankValue_top4_pass_rate_mean', 0.0):.4f}, "
        f"bottom6_pass={state.get('best_RankValue_bottom6_pass_rate_mean', 0.0):.4f}\n"
        f"- Best CombinedPass: pass_rate={state.get('best_combined_pass_rate_mean', 0.0):.4f}"
    )


def get_outcome_label(exp: dict) -> str:
    """从实验结果分析中提取 outcome 标签"""
    ra = exp.get("result_analysis", "")
    if not ra:
        return "NEUTRAL"
    if "KEEP" in ra or "passed" in ra:
        return "KEPT"
    if "REVERT" in ra or "decreased" in ra:
        return "REVERTED"
    return "NEUTRAL"


def format_experiment_history_entry(result: dict) -> dict:
    """返回需要存入 experiment_history 的额外指标字段"""
    entry = {}
    for field in METRIC_FIELDS:
        if field in result:
            entry[field] = result[field]
    return entry


def format_extra_details(exp: dict) -> list[str]:
    """返回实验日志中的额外详情行（列联表诊断）"""
    details = []
    ct = exp.get("contingency_table_mean")
    if ct and isinstance(ct, str) and len(ct) > 10:
        for line in ct.split('\n'):
            details.append(f"  [Diag] {line}")
    return details


def _constraint_checks(new_metrics: dict, branch_best: dict | None) -> list[bool]:
    """8 项约束检查（用于显示）。
    1-4: mean >= online baseline * 0.95
    5-7: pass rate >= branch_best * 0.85
    8: combined_pass_rate >= branch_best * 0.85
    """
    bl = ONLINE_BASELINE
    if not branch_best:
        return [True] * 8
    checks = []
    # 1-4: mean vs baseline (5% tolerance)
    for key in MEAN_BASELINE_KEYS:
        checks.append(new_metrics.get(key, 0) >= bl.get(key, 0) * (1.0 - MEAN_BASELINE_TOLERANCE))
    # 5-7: pass rate vs branch_best (15% tolerance)
    for key in PASS_RATE_KEYS:
        checks.append(new_metrics.get(key, 0) >= branch_best.get(key, 0) * PASS_RATE_TOLERANCE)
    # 8: combined_pass_rate vs branch_best (15% tolerance)
    checks.append(new_metrics.get("combined_pass_rate_mean", 0) >= branch_best.get("combined_pass_rate_mean", 0) * PASS_RATE_TOLERANCE)
    return checks


def compute_outcome_reward(new_metrics: dict, branch_best: dict | None, outcome: str) -> float:
    """连续 outcome reward [-1, 1]。

    combined_pass_rate 作为核心 gate:
      - 3 个 individual pass rate (RV_top4, RV_bottom6, VecSim) 先各自归一化到 [0,1]
      - combined_pass_rate 归一化后作为乘数 (gate):
        pass_component = indiv_mean * combined_score
      - combined 低 → individual 高分也被拉低 (防止只刷单项指标)
      - combined = 0 → pass_component = 0

    4 个 mean (目标接近 online baseline):
      - ratio = val / online_baseline
      - 0.95~1.0 容忍区 → 0.9~1.0
      - <0.95 线性下降

    权重: pass_component 60%, mean_component 40%
    REVERT 时强制取负。
    """
    if outcome == "CRASH":
        return -1.0
    if not branch_best:
        return 0.5

    import numpy as np

    # ── 3 个 individual pass rate (归一化到 [0,1] at 60%) ──
    indiv_pass_rates = [
        new_metrics.get("RankValue_top4_pass_rate_mean", 0.0),
        new_metrics.get("RankValue_bottom6_pass_rate_mean", 0.0),
        new_metrics.get("VecSim_pass_rate_mean", 0.0),
    ]
    indiv_scores = [min(1.0, r / 0.60) for r in indiv_pass_rates]
    indiv_mean = float(np.mean(indiv_scores))

    # ── combined_pass_rate 作为 gate (核心指标) ──
    combined_rate = new_metrics.get("combined_pass_rate_mean", 0.0)
    combined_score = min(1.0, combined_rate / 0.60)

    # combined gates individual: 低 combined → 整体 pass_component 被拉低
    pass_component = indiv_mean * combined_score

    # ── 4 个 mean (目标接近 online baseline) ──
    mean_baselines = {
        "Cat3Diversity_top4_mean": 6.4381,
        "Cat3Diversity_top10_mean": 12.1431,
        "RankValue_top4_mean": 318.8872,
        "RankValue_bottom6_mean": 185.4970,
    }
    mean_scores = []
    for metric, baseline in mean_baselines.items():
        ratio = new_metrics.get(metric, 0.0) / max(baseline, 1e-8)
        if ratio >= 0.95:
            mean_scores.append(min(1.0, 0.9 + 2.0 * (ratio - 0.95)))
        else:
            mean_scores.append(max(0.0, 0.9 * ratio / 0.95))
    mean_component = float(np.mean(mean_scores))

    # ── 组合: pass 60%, mean 40% ──
    reward = (mean_component * 0.4 + pass_component * 0.6) * 2 - 1

    if outcome == "REVERT":
        reward = -abs(reward)

    return float(np.clip(reward, -1.0, 1.0))


def should_eval_last_turn_prm() -> bool:
    """Enable plan-quality PRM evaluation for the last assistant turn.

    The last turn is the experiment plan output (\\boxed{...}). We evaluate
    its reasoning quality (hypothesis clarity, change specificity, analysis
    depth) independently of the experiment outcome, using
    PRMEvaluator.evaluate_plan_quality(). This provides a process-quality
    signal orthogonal to the outcome reward (trajectory_score).
    """
    return True

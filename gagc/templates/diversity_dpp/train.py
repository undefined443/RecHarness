"""
train.py — DPP Multi-Window Diversity Algorithm (FIXED, Agent 不可修改)

所有可调参数在 config.yaml 的 scatter 部分，Agent 只需修改 config.yaml。
train.py 自动读取 config.yaml 参数，无需修改代码。

核心算法: DPP (Determinantal Point Process) 多窗口多样性算分器
  - 从 Java 版 DPPMultiWinDiversityScoreCalculator / DPPMultiWinUpdater 翻译而来
  - 使用 Cholesky 分解进行增量更新, 支持滑动窗口吉文斯旋转降阶
  - 多窗口融合: sum / weighted_sum / power_product

V2 核心约束 (不变):
  - 仅可以使用 rank (score), fst_rank (fst_score), 和向量 emb (img_vec, text_vec) 作为输入
  - 不可以使用 cat1, cat3 类目信息作为算法输入 (仅用于辅助评估)
  - 不允许仅调整10个输出商品的rank绝对值排序

核心接口:
  scatter(request, vec_lookup, config) -> list[str]
    返回10个goods_id (字符串), 按坑位0~9排序

DPP 算法核心流程:
  1. 构建商品列表: 前序曝光(pre_goods) + 已选(selected) + 候选(candidates)
  2. 计算质量分 Q (从rankscore派生) 和相似度 S (从向量emb计算)
  3. 构建 Kernel 矩阵 L = diag(Q) * S * diag(Q)
  4. Cholesky 分解, 增量更新残差能量
  5. 多样性权重 = 残差能量 / 初始能量 (d_ratio) 或 残差能量 / rawScore (d_value)
  6. rerankScore = rankScore * diversityWeight, 贪心选择 TOP1

逐坑贪心排序:
  for pit in range(10):
      pit 0-3: 使用 fst_score, FIRST_POS 配置
      pit 4-9: 使用 score, DEFAULT 配置
      pit 0 和 pit 4 时重新初始化 DPP 引擎
"""

import numpy as np

ZERO = 1e-12


def _transform_sim(sim_transform_type, sim, min_sim, exp_alpha, exp_bias):
    """相似度敏感度映射

    Java 对应: DPPEngine.transformSim / transformSimByOneMinusExpNeg / transformSimBySqrtOneMinusExpNeg

    Transform types:
      - None / "default": 不变换, sim 直接使用
      - "ONE_MINUS_EXP_NEG": 1 - e^(-alpha * max(0, sim - S)) + bias, 压缩高相似度区间
      - "SQRT_ONE_MINUS_EXP_NEG": sqrt(1 - e^(-alpha * max(0, sim - S))) + bias, 更激进地压缩
    """
    if sim_transform_type is None or sim_transform_type == "default":
        return sim
    clip_sim = np.maximum(sim - min_sim, 0.0)
    if sim_transform_type == "ONE_MINUS_EXP_NEG":
        return np.maximum(0.0, np.minimum(1.0 - np.exp(-exp_alpha * clip_sim) + exp_bias, 1.0))
    elif sim_transform_type == "SQRT_ONE_MINUS_EXP_NEG":
        inner = np.maximum(0.0, 1.0 - np.exp(-exp_alpha * clip_sim))
        return np.maximum(0.0, np.minimum(np.sqrt(inner) + exp_bias, 1.0))
    return sim


class DPPEngine:
    """单窗口 DPP 引擎

    Java 对应: DPPMultiWinUpdater.DPPEngine

    核心: Cholesky 分解增量更新 + 滑动窗口吉文斯旋转降阶

    参数:
      win_config: dict, 窗口配置 (slidingWindowSize, enableSlidingWindowGivensRotation)
      fst_config: dict, FstDefConfig (Q处理, Sim变换, rerank方法, qPower/dwPower)
      enable_max_ei: bool, 是否限制打压能量不超过自身剩余能量
      engine_idx: int, 引擎序号
    """

    def __init__(self, win_config, fst_config, enable_max_ei, engine_idx):
        self.engine_idx = engine_idx
        self.sliding_window_size = int(win_config.get('slidingWindowSize', 30))
        self.enable_givens = bool(win_config.get('enableSlidingWindowGivensRotation', True))

        self.preprocess_q_method = fst_config.get('preprocessQMethod', 'default')
        self.expo_default_q_method = fst_config.get('expoDefaultQMethod', 'default')
        self.expo_default_q = float(fst_config.get('expoDefaultQ', 1.0))
        self.sim_transform_type = fst_config.get('simTransformType')
        self.min_sim = float(fst_config.get('minSim', 0.0))
        self.exp_alpha = float(fst_config.get('expAlpha', 1.0))
        self.exp_bias = float(fst_config.get('expBias', 0.0))
        self.rerank_method = fst_config.get('rerankMethod', 'd_ratio')
        self.q_power = float(fst_config.get('qPower', 1.0))
        self.dw_power = float(fst_config.get('dwPower', 1.0))
        self.enable_max_ei = bool(enable_max_ei)

    def init(self, all_img, all_text, all_raw_scores, expo_num, selected_num, output_size):
        """初始化 DPP 引擎

        Java 对应: DPPMultiWinUpdater.init + DPPEngine constructor + initDPPEnergy

        构建商品列表: expo(前序曝光) + selected(已选未曝光) + candidates(候选)
        计算质量分 Q, 初始化残差能量, 根据历史已曝光&已选进行初始打压

        Args:
            all_img: (total_num, 128) 所有商品的图像向量
            all_text: (total_num, 128) 所有商品的文本向量
            all_raw_scores: (total_num,) 所有商品的原始打分
            expo_num: 前序曝光商品数量
            selected_num: 已选(未曝光)商品数量
            output_size: 本阶段要选择的商品数量 (stage1=4, stage2=6)
        """
        self.total_num = len(all_raw_scores)
        self.expo_num = expo_num
        self.selected_num = selected_num
        self.candi_num = self.total_num - expo_num - selected_num
        self.expo_and_selected_num = expo_num + selected_num
        self.max_select_steps = self.expo_and_selected_num + output_size

        self.all_img = all_img
        self.all_text = all_text
        self.raw_scores = np.maximum(all_raw_scores, ZERO).copy()

        # 质量分预处理
        # Java 对应: DPPEngine.preProcessRankScore + getRawScores
        self.quality_scores = self.raw_scores.copy()
        if self.preprocess_q_method == "q_power":
            self.quality_scores = np.power(self.quality_scores, self.q_power)

        # 历史已曝光商品的 Q 值处理
        # Java 对应: DPPEngine.getRawScores -> expoDefaultQMethod
        # Java 流程: 先设置所有商品的 rawScore (expo 用 expoDefaultQMethod, 其余用 scoreGetter)
        #            然后统一对 rawScore 应用 preprocessQMethod (q_power)
        # 因此 expoDefaultQ 也需要经过 q_power 处理
        if self.expo_num > 0:
            if self.expo_default_q_method == "topN_candi":
                # Java: sortedScores 包含 selected + candi 的 rawScore
                # Python: quality_scores 已经过 q_power 处理, 排序结果等价
                sel_and_candi_q = self.quality_scores[self.expo_num:]
                if len(sel_and_candi_q) > 0:
                    sorted_q = np.sort(sel_and_candi_q)[::-1]
                    limit = min(self.expo_num, len(sorted_q))
                    self.quality_scores[:limit] = sorted_q[:limit]
                    if self.expo_num > limit:
                        fill_val = sorted_q[-1] if len(sorted_q) > 0 else self.expo_default_q
                        self.quality_scores[limit:self.expo_num] = fill_val
            else:
                # Java: expoDefaultQ 也要经过 q_power 处理
                expo_q = self.expo_default_q
                if self.preprocess_q_method == "q_power":
                    expo_q = np.power(expo_q, self.q_power)
                self.quality_scores[:self.expo_num] = expo_q

        # 初始化残差能量 residualEnergySq = Q^2
        self.residual_energy_sq = self.quality_scores ** 2

        # Cholesky 因子矩阵: [total_num][maxSelectSteps]
        self.cholesky_factors = np.zeros((self.total_num, self.max_select_steps))
        self.active_cholesky = np.zeros((self.max_select_steps, self.max_select_steps))
        self.window_left_index = 0

        # 已选商品 mask
        self.is_selected = np.zeros(self.total_num, dtype=bool)
        self.is_selected[:self.expo_and_selected_num] = True

        # 初始能量打压: 根据历史已曝光&已选商品更新残差能量
        self._update_initial_energy()

        # 初始化 DPP 值
        self.dpp_values = np.zeros(self.total_num)
        self.dpp_values[:self.expo_and_selected_num] = 0.0
        if self.candi_num > 0:
            self.dpp_values[self.expo_and_selected_num:] = self.residual_energy_sq[self.expo_and_selected_num:]

    def _update_initial_energy(self):
        """初始能量打压: 逐个处理历史已曝光&已选商品

        Java 对应: DPPEngine.updateInitialEnergy
        """
        for step in range(self.expo_and_selected_num):
            if self.enable_givens:
                self._update_active_cholesky(step, step)
            self._update_remaining_energy(step, step, step + 1)
            if self.enable_givens and self.sliding_window_size > 0 and (step + 1) > self.sliding_window_size:
                self._sliding_window_givens_rotation(step)

    def _update_remaining_energy(self, best_idx, current_row, start_idx):
        """核心更新逻辑: 更新候选商品的残差能量

        Java 对应: DPPEngine.updateRemainingEnergy

        对每个候选商品 i:
          1. 计算相似度 sim = (max(img_dot, text_dot) + 1) / 2
          2. 相似度变换 transformSim
          3. Kernel 矩阵项 L_ji = Q_best * sim * Q_i
          4. 增量向量 V: sum_v = v_best[m] * v_i[m] (滑动窗口内)
          5. 更新能量: e_i = (L_ji - sum_v) / sqrt(energy_best)
          6. residualEnergySq[i] -= e_i^2

        Args:
            best_idx: 当前被选中的商品索引
            current_row: 当前执行到 V 矩阵的哪一行
            start_idx: 候选商品索引起始位置
        """
        max_energy = self.residual_energy_sq[best_idx]
        if max_energy <= ZERO:
            return
        sqrt_max_energy = np.sqrt(max_energy)
        v_best = self.cholesky_factors[best_idx]

        # 有效候选: 排除已选 & 零能量
        indices = np.arange(start_idx, self.total_num)
        valid_mask = ~self.is_selected[indices]
        valid_mask &= (indices != best_idx)
        valid_mask &= (self.residual_energy_sq[indices] > 0)
        valid_indices = indices[valid_mask]

        if len(valid_indices) == 0:
            return

        # 1. 计算相似度 (向量化)
        sel_img = self.all_img[best_idx]
        sel_text = self.all_text[best_idx]
        img_sim = self.all_img[valid_indices] @ sel_img
        text_sim = self.all_text[valid_indices] @ sel_text
        sim = (np.maximum(img_sim, text_sim) + 1.0) / 2.0

        # 2. 相似度变换
        sim = _transform_sim(self.sim_transform_type, sim, self.min_sim, self.exp_alpha, self.exp_bias)

        # 3. Kernel 矩阵项
        q_best = self.quality_scores[best_idx]
        q_valid = self.quality_scores[valid_indices]
        matrix_lji = q_best * sim * q_valid

        # 4. 增量向量 V 的内积和 (向量化)
        if self.enable_givens:
            start_row = self.window_left_index
        else:
            start_row = 0

        if current_row > start_row:
            sum_v = self.cholesky_factors[valid_indices][:, start_row:current_row] @ v_best[start_row:current_row]
        else:
            sum_v = np.zeros(len(valid_indices))

        # 5. 更新能量
        e_i = (matrix_lji - sum_v) / sqrt_max_energy

        # 异常数值处理: 受打压能量不高于自身剩余能量
        if self.enable_max_ei:
            e_i_sq = e_i * e_i
            overflow = e_i_sq > self.residual_energy_sq[valid_indices]
            if np.any(overflow):
                max_allowed = np.sqrt(np.maximum(0.0, self.residual_energy_sq[valid_indices[overflow]]))
                e_i[overflow] = np.sign(e_i[overflow]) * max_allowed

        # 6. 存储 & 更新残差能量
        self.cholesky_factors[valid_indices, current_row] = e_i
        self.residual_energy_sq[valid_indices] -= e_i * e_i
        self.residual_energy_sq[valid_indices] = np.maximum(self.residual_energy_sq[valid_indices], 0.0)

    def _update_active_cholesky(self, best_idx, current_row):
        """更新活跃 Cholesky 矩阵

        Java 对应: DPPEngine.updateActiveCholesky
        """
        best_energy = self.residual_energy_sq[best_idx]
        sqrt_best = np.sqrt(best_energy) if best_energy > ZERO else ZERO
        v_best = self.cholesky_factors[best_idx]
        self.active_cholesky[current_row, self.window_left_index:current_row] = v_best[self.window_left_index:current_row]
        self.active_cholesky[current_row, current_row] = sqrt_best

    def _sliding_window_givens_rotation(self, current_row):
        """滑动窗口吉文斯旋转降阶

        Java 对应: DPPEngine.slidingWindowGivensRotation

        当选择商品超出窗口时, 利用吉文斯旋转进行退耦降阶:
        1. 对窗口内剩余商品依次执行正交旋转
        2. 对所有商品在 choleskyFactors 中的投影维度执行同步正交旋转
        3. 旋转退耦后, 加回已剔除商品的投影能量
        """
        self.window_left_index += 1
        oldest = self.window_left_index - 1

        for ind in range(self.window_left_index, current_row + 1):
            v_ind_ind = self.active_cholesky[ind, ind]
            v_ind_prev = self.active_cholesky[ind, oldest]
            t = np.sqrt(v_ind_ind * v_ind_ind + v_ind_prev * v_ind_prev)

            if t <= ZERO:
                c, s = 1.0, 0.0
            else:
                c = v_ind_ind / t
                s = v_ind_prev / t

            self.active_cholesky[ind, ind] = t
            self.active_cholesky[ind, oldest] = 0.0

            # activeCholesky 矩阵列旋转
            if ind + 1 <= current_row:
                v_r_ind = self.active_cholesky[ind + 1:current_row + 1, ind].copy()
                v_r_prev = self.active_cholesky[ind + 1:current_row + 1, oldest].copy()
                self.active_cholesky[ind + 1:current_row + 1, ind] = c * v_r_ind + s * v_r_prev
                self.active_cholesky[ind + 1:current_row + 1, oldest] = c * v_r_prev - s * v_r_ind

            # 对所有商品在 choleskyFactors 中的投影维度执行同步正交旋转 (向量化)
            cis_ind = self.cholesky_factors[:, ind].copy()
            cis_prev = self.cholesky_factors[:, oldest].copy()
            self.cholesky_factors[:, ind] = c * cis_ind + s * cis_prev
            self.cholesky_factors[:, oldest] = c * cis_prev - s * cis_ind

        # 旋转退耦后, 加回已剔除商品的投影能量
        vals = self.cholesky_factors[:, oldest]
        self.residual_energy_sq = np.minimum(
            self.residual_energy_sq + vals * vals,
            self.quality_scores ** 2
        )

    def select_and_update(self, best_idx, current_row):
        """选中商品后更新 DPP 矩阵状态

        Java 对应: DPPEngine.selectAndUpdate

        Args:
            best_idx: 当前被选中的商品索引
            current_row: 当前执行到 V 矩阵的哪一行
        """
        if best_idx < 0 or current_row >= self.cholesky_factors.shape[1]:
            return
        self.is_selected[best_idx] = True

        if self.enable_givens:
            self._update_active_cholesky(best_idx, current_row)

        self._update_remaining_energy(best_idx, current_row, self.expo_and_selected_num)

        if self.enable_givens and self.sliding_window_size > 0 and (current_row + 1) > self.sliding_window_size:
            self._sliding_window_givens_rotation(current_row)

        # 更新 DPP 值
        self.dpp_values[:self.expo_and_selected_num] = 0.0
        candi_indices = np.arange(self.expo_and_selected_num, self.total_num)
        if len(candi_indices) > 0:
            not_sel = ~self.is_selected[candi_indices]
            self.dpp_values[candi_indices] = np.where(
                not_sel, self.residual_energy_sq[candi_indices], 0.0
            )

    def get_diversity_weights(self):
        """获取所有商品的多样性权重

        Java 对应: DPPEngine.getDiversityWeight

        rerankMethod:
          - "d_ratio": diversityWeight = (dppValue / qValue^2)^dwPower
            = (残差能量 / 初始能量)^dwPower = (1 - F(sim))^dwPower
            dwPower 越大打散越强
          - "d_value": diversityWeight = dppValue / rawScore
            = qValue^2 * (1 - F(sim)) / rawScore
            qPower 越大打散越弱 (通过 preprocessQMethod="q_power" 生效)

        Returns:
            np.ndarray (total_num,): 多样性权重, 候选商品有值, 已选商品为0
        """
        weights = np.zeros(self.total_num)
        candi_start = self.expo_and_selected_num
        if self.candi_num <= 0:
            return weights

        q_vals = self.quality_scores[candi_start:]
        raw_vals = self.raw_scores[candi_start:]
        dpp_vals = self.dpp_values[candi_start:]

        if self.rerank_method == "d_ratio":
            q_sq = q_vals * q_vals
            valid = q_sq > ZERO
            ratio = np.where(valid, dpp_vals / np.where(valid, q_sq, 1.0), 0.0)
            ratio = np.maximum(ratio, 0.0)
            weights[candi_start:] = np.power(ratio, self.dw_power)
        elif self.rerank_method == "d_value":
            valid = raw_vals > ZERO
            weights[candi_start:] = np.where(valid, dpp_vals / np.where(valid, raw_vals, 1.0), 0.0)
        else:
            q_sq = q_vals * q_vals
            valid = q_sq > ZERO
            weights[candi_start:] = np.where(valid, dpp_vals / np.where(valid, q_sq, 1.0), 0.0)

        return weights


def _fuse_weights(win_weights_list, method, multi_win_weights):
    """融合多窗口权重

    Java 对应: DPPMultiWinUpdater.fuseWeights

    Args:
        win_weights_list: list[np.ndarray], 各窗口的权重
        method: str, 融合方法 (sum / weighted_sum / power_product)
        multi_win_weights: list[float], 各窗口的融合权重

    Returns:
        np.ndarray: 融合后的权重
    """
    if not win_weights_list:
        return np.zeros(0)

    n = len(win_weights_list)
    total_len = len(win_weights_list[0])

    # 补齐 multi_win_weights 长度, 与 Java 的 Math.min 等价但更安全
    if len(multi_win_weights) < n:
        multi_win_weights = list(multi_win_weights) + [1.0] * (n - len(multi_win_weights))

    if method == "weighted_sum":
        w_sum = sum(multi_win_weights[:n]) + ZERO
        result = np.zeros(total_len)
        for i in range(n):
            result += win_weights_list[i] * multi_win_weights[i] / w_sum
        return result
    elif method == "power_product":
        result = np.ones(total_len)
        for i in range(n):
            result *= np.power(np.maximum(win_weights_list[i], ZERO), multi_win_weights[i])
        return result
    else:  # "sum" (default)
        result = np.zeros(total_len)
        for w in win_weights_list:
            result += w
        return result / n


def _init_dpp_stage(pit, is_first_pos, pre_items, expo_num,
                    selected_indices, cand_img, cand_text,
                    fst_scores, scores, mask, raw_score_arr,
                    dpp_config_list, multi_win_num, enable_max_ei):
    """初始化 DPP 阶段

    Java 对应: DPPMultiWinUpdater.init + initDPPEngines

    在 pit 0 (PSA Stage 1) 和 pit 4 (PSA Stage 2) 时调用:
      - 构建商品列表: expo(前序曝光) + selected(已选) + candidates(候选)
      - 初始化多窗口 DPP 引擎

    Returns:
        dict: DPP 状态 (engines, avail_indices, candi_offset, total_num, ...)
    """
    # 已选商品 (Stage 2 时为 Stage 1 选中的商品)
    # Java: 已选商品的 rawScore 使用当前阶段的 scoreGetter 计算
    #   Stage 1 (isPSAFirstStage=true): firstPosScoreGetter → fst_score
    #   Stage 2 (isPSAFirstStage=false): defaultScoreGetter → score
    if pit == 4 and selected_indices:
        sel_img = cand_img[selected_indices]
        sel_text = cand_text[selected_indices]
        sel_scores = raw_score_arr[selected_indices]
        selected_num = len(selected_indices)
    else:
        sel_img = np.zeros((0, 128), dtype=np.float32)
        sel_text = np.zeros((0, 128), dtype=np.float32)
        sel_scores = np.zeros(0, dtype=np.float64)
        selected_num = 0

    # 可用候选商品
    avail_mask = ~mask
    avail_indices = np.where(avail_mask)[0]
    candi_img = cand_img[avail_indices]
    candi_text = cand_text[avail_indices]
    candi_scores = raw_score_arr[avail_indices]

    # 前序曝光商品
    if expo_num > 0:
        expo_img = np.array([p[0] for p in pre_items], dtype=np.float32)
        expo_text = np.array([p[1] for p in pre_items], dtype=np.float32)
    else:
        expo_img = np.zeros((0, 128), dtype=np.float32)
        expo_text = np.zeros((0, 128), dtype=np.float32)

    # 构建全部商品列表: expo + selected + candidates
    total_num = expo_num + selected_num + len(avail_indices)
    if total_num > 0:
        all_img = np.concatenate([expo_img, sel_img, candi_img], axis=0)
        all_text = np.concatenate([expo_text, sel_text, candi_text], axis=0)
        all_scores = np.concatenate([
            np.zeros(expo_num, dtype=np.float64),
            sel_scores,
            candi_scores
        ])
    else:
        all_img = np.zeros((0, 128), dtype=np.float32)
        all_text = np.zeros((0, 128), dtype=np.float32)
        all_scores = np.zeros(0, dtype=np.float64)

    output_size = 4 if is_first_pos else 6

    # 初始化多窗口 DPP 引擎
    num_engines = min(multi_win_num, len(dpp_config_list)) if dpp_config_list else 0
    engines = []
    for win_idx in range(num_engines):
        win_config = dpp_config_list[win_idx]

        # 获取 FstDefConfig (FIRST_POS 或 DEFAULT)
        fst_def_map = win_config.get('fstDefConfigMap', {})
        fst_key = "FIRST_POS" if is_first_pos else "DEFAULT"
        fst_config = fst_def_map.get(fst_key, fst_def_map.get('DEFAULT', {}))

        # 如果 fstDefConfigMap 为空, 使用窗口级别的扁平配置
        if not fst_config:
            fst_config = {
                'preprocessQMethod': win_config.get('preprocessQMethod', 'default'),
                'expoDefaultQMethod': win_config.get('expoDefaultQMethod', 'default'),
                'expoDefaultQ': win_config.get('expoDefaultQ', 1.0),
                'simTransformType': win_config.get('simTransformType'),
                'minSim': win_config.get('minSim', 0.0),
                'expAlpha': win_config.get('expAlpha', 1.0),
                'expBias': win_config.get('expBias', 0.0),
                'rerankMethod': win_config.get('rerankMethod', 'd_ratio'),
                'qPower': win_config.get('qPower', 1.0),
                'dwPower': win_config.get('dwPower', 1.0),
            }

        engine = DPPEngine(win_config, fst_config, enable_max_ei, win_idx)
        engine.init(all_img, all_text, all_scores, expo_num, selected_num, output_size)
        engines.append(engine)

    return {
        'engines': engines,
        'avail_indices': avail_indices,
        'candi_offset': expo_num + selected_num,
        'total_num': total_num,
        'stage_start_pit': pit,
        'expo_num': expo_num,
        'selected_num': selected_num,
    }


def scatter(request, vec_lookup, config):
    """
    DPP 多窗口多样性打散算法

    策略:
      使用 DPP (Determinantal Point Process) 进行逐坑位贪心选择。
      多窗口 DPP 融合不同窗口大小的多样性信号。

      核心流程:
        1. 构建商品列表: 前序曝光(pre_goods) + 已选(selected) + 候选(candidates)
        2. 计算质量分 Q (从rankscore派生) 和相似度 S (从向量emb计算)
        3. Cholesky 分解增量更新残差能量
        4. 多样性权重 = 残差能量 / 初始能量 (d_ratio) 或 残差能量 / rawScore (d_value)
        5. rerankScore = rankScore * diversityWeight, 贪心选择 TOP1

      PSA 两阶段:
        - Stage 1 (pit 0-3): FIRST_POS 配置, 使用 fst_score
        - Stage 2 (pit 4-9): DEFAULT 配置, 使用 score, 重新初始化 DPP 引擎
    """
    candidates = request['candidates']
    pre_goods = request['pre_goods']

    n = len(candidates)
    goods_ids = [c['goods_id'] for c in candidates]
    fst_scores = np.array([c['fst_score'] for c in candidates], dtype=np.float64)
    scores = np.array([c['score'] for c in candidates], dtype=np.float64)

    # DPP 配置
    dpp_cfg = config.get('scatter', {})
    multi_win_num = int(dpp_cfg.get('multiWinNum', 1))
    weights_fusion_method = dpp_cfg.get('weightsFusionMethod', 'sum')
    multi_win_weights = dpp_cfg.get('multiWinWeights', [1.0] * multi_win_num)
    enable_max_ei = dpp_cfg.get('enableMaxEi', False)
    dpp_config_list = dpp_cfg.get('dppConfigList', [])

    # 获取向量
    img_vecs = vec_lookup['img_vecs']
    text_vecs = vec_lookup['text_vecs']
    goods_id_to_idx = vec_lookup['goods_id_to_idx']

    # 候选商品向量
    cand_vec_idxs = np.array(
        [goods_id_to_idx.get(gid, -1) for gid in goods_ids],
        dtype=np.int64
    )
    valid_vec_mask = cand_vec_idxs >= 0
    cand_img = np.zeros((n, 128), dtype=np.float32)
    cand_text = np.zeros((n, 128), dtype=np.float32)
    if valid_vec_mask.any():
        valid_idxs = cand_vec_idxs[valid_vec_mask]
        cand_img[valid_vec_mask] = img_vecs[valid_idxs]
        cand_text[valid_vec_mask] = text_vecs[valid_idxs]

    # 前序曝光商品向量
    pre_items = []
    for g in pre_goods:
        idx = goods_id_to_idx.get(g['goods_id'], -1)
        if idx >= 0:
            pre_items.append((img_vecs[idx], text_vecs[idx]))
    expo_num = len(pre_items)

    # 逐坑贪心选择
    selected = []
    selected_indices = []
    mask = np.zeros(n, dtype=bool)

    # DPP 状态 (在 pit 0 和 pit 4 重新初始化)
    dpp_state = {}

    for pit in range(10):
        is_first_pos = pit < 4
        raw_score_arr = fst_scores if is_first_pos else scores

        # PSA 阶段切换时重新初始化 DPP
        if pit == 0 or pit == 4:
            dpp_state = _init_dpp_stage(
                pit, is_first_pos, pre_items, expo_num,
                selected_indices, cand_img, cand_text,
                fst_scores, scores, mask, raw_score_arr,
                dpp_config_list, multi_win_num, enable_max_ei
            )

        # 获取多样性权重
        engines = dpp_state['engines']
        avail_indices = dpp_state['avail_indices']
        candi_offset = dpp_state['candi_offset']
        total_num = dpp_state['total_num']
        stage_selected_num = dpp_state['selected_num']

        if engines:
            # 融合多窗口权重
            win_weights_list = [engine.get_diversity_weights() for engine in engines]
            fused_weights = _fuse_weights(win_weights_list, weights_fusion_method, multi_win_weights)

            # 候选商品的权重
            candi_weights = fused_weights[candi_offset:]

            # 无向量商品的兜底: 使用有向量商品的平均权重
            avail_valid_mask = valid_vec_mask[avail_indices]
            if not avail_valid_mask.all():
                valid_w = candi_weights[avail_valid_mask]
                default_scatter = float(np.mean(valid_w)) if len(valid_w) > 0 else 1.0
                candi_weights = np.where(avail_valid_mask, candi_weights, default_scatter)

            # rerankScore = rankScore * diversityWeight
            candi_scores = raw_score_arr[avail_indices]
            adjusted_score = candi_scores * candi_weights

            best_local = int(np.argmax(adjusted_score))
        else:
            candi_scores = raw_score_arr[avail_indices]
            best_local = int(np.argmax(candi_scores))

        best_cand_idx = int(avail_indices[best_local])
        selected.append(goods_ids[best_cand_idx])
        selected_indices.append(best_cand_idx)
        mask[best_cand_idx] = True

        # 更新 DPP 矩阵
        if engines:
            best_all_idx = candi_offset + best_local
            current_row = expo_num + stage_selected_num + (pit - dpp_state['stage_start_pit'])
            for engine in engines:
                engine.select_and_update(best_all_idx, current_row)

    return selected

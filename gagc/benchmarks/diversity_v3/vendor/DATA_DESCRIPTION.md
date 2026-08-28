# 数据说明 (Data Description) — V2

## 0. 原始数据存储

### 服务器信息

- **服务器**: `sbd2-pikachu-24` (SSH: `shiboying@sbd2-pikachu-24`)
- **数据根目录**: `/cfs5/shiboying/datasets/diversity_mechanism/`
- **Python**: `/usr/local/bin/python3` (3.10.14, 需 `sudo` 安装包)

### 原始数据一览

| 数据集 | 原始路径 | 格式 | 总大小 | 文件结构 |
|--------|---------|------|--------|---------|
| 采样日志 | `/cfs5/shiboying/datasets/diversity_mechanism/sample_0721` | ORC | 13G | `hr=00/000000_0` ~ `hr=23/000000_0` (24小时分区) |
| 商品向量 | `/cfs5/shiboying/datasets/diversity_mechanism/vec_0721` | ORC | 9.4G | `000000_0` ~ `000019_0` (20个文件) |

### 原始 Hive 表 Schema

#### 采样日志表 `arec.arec_scatter_auto_search_sample_hr`

```sql
CREATE TABLE `arec.arec_scatter_auto_search_sample_hr`(
  `search_id` string COMMENT '请求id', 
  `uid` bigint COMMENT 'uid', 
  `pdd_log_id` string COMMENT '请求id', 
  `mix_rank_goods` array<string> COMMENT '候选goods', 
  `mix_rank_cat1` array<string> COMMENT '候选cat1', 
  `mix_rank_cat` array<string> COMMENT '候选cat', 
  `mix_rank_score` array<string> COMMENT '候选rankscore', 
  `mix_rank_fst_score` array<string> COMMENT '候选首坑rankscore', 
  `result_goods` array<string> COMMENT '结果goods', 
  `result_cat1` array<string> COMMENT '结果cat1', 
  `result_cat` array<string> COMMENT '结果cat', 
  `result_score` array<string> COMMENT '结果rankscore', 
  `result_fst_score` array<string> COMMENT '结果首坑rankscore', 
  `pre_goods` array<string> COMMENT '前序goods', 
  `pre_goods_cat1` array<string> COMMENT '前序cat1', 
  `pre_goods_cat` array<string> COMMENT '前序cat')
COMMENT 'arec.arec_scatter_auto_search_sample_hr'
PARTITIONED BY ( 
  `pt` string COMMENT 'pt', 
  `hr` string COMMENT 'hr')
```

- 全量数据: 24小时 × ~13,304条/小时 ≈ 32万条请求
- 每小时一个ORC文件 (`hr=XX/000000_0`)，约290M
- 分区字段: `pt` (日期), `hr` (小时)

#### 商品向量表 `arec.diversity_auto_goods_vec`

```sql
CREATE TABLE `arec.diversity_auto_goods_vec`(
  `goods_id` bigint COMMENT 'goods id', 
  `img_vec` array<float> COMMENT 'img_vec', 
  `text_vec` array<float> COMMENT 'text_vec')
COMMENT 'auto diversity 向量'
PARTITIONED BY ( 
  `pt` string COMMENT '')
```

- 全量数据: 20个ORC文件 × ~759,281行/文件 ≈ 1500万商品向量
- 每个文件约484M
- 向量为单位向量（L2归一化），128维 float32

### 本地开发数据

从 hr=00 取前500条请求 + 对应向量过滤，存放于 git 仓库：

| 文件 | 格式 | 大小 | 说明 |
|------|------|------|------|
| `data/sample_500.parquet` | Parquet | 22M | 500条请求的完整日志 |
| `data/vec_filtered.parquet` | Parquet | 140M | 298,549条商品向量（覆盖率100%） |

---

## 1. 采样日志 (`sample_500.parquet`)

### 概览

- **行数**: 500（每行 = 一次搜索请求）
- **列数**: 16
- **每行包含三部分数据**: 精排候选、线上打散结果、前序曝光商品

### 字段详情

#### 请求标识

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `search_id` | string | 请求ID | `uFEnekOd_U2SHyX7ovm2u2RNOgF...` |
| `uid` | int64 | 用户ID | `1005692305` |
| `pdd_log_id` | string | 日志ID | `199606c2-2e1d-4914-9606-c22e1d791489` |

#### 精排候选（mix_rank_*，约1000个商品）

| 字段 | 类型 | 元素类型 | 说明 |
|------|------|---------|------|
| `mix_rank_goods` | array | string | 候选商品ID列表 |
| `mix_rank_cat1` | array | string | 候选商品C1类目（一级类目） |
| `mix_rank_cat` | array | string | 候选商品C3类目（叶子类目） |
| `mix_rank_score` | array | string | 候选商品rankscore（数值字符串） |
| `mix_rank_fst_score` | array | string | 候选商品首坑rankscore（数值字符串） |

- 候选数量: min=543, max=1415, mean=928
- 所有数组等长，按索引一一对应: `goods[i] ↔ cat1[i] ↔ cat[i] ↔ score[i] ↔ fst_score[i]`
- score/fst_score 存储为字符串，需 `float()` 转换
- mix_rank_goods 的顺序由上游日志决定，不保证按任何分数排序

#### 线上打散结果（result_*，10个商品）

| 字段 | 类型 | 元素类型 | 说明 |
|------|------|---------|------|
| `result_goods` | array | string | 结果商品ID列表 |
| `result_cat1` | array | string | 结果商品C1类目 |
| `result_cat` | array | string | 结果商品C3类目 |
| `result_score` | array | string | 结果商品rankscore |
| `result_fst_score` | array | string | 结果商品首坑rankscore |

- 结果数量: 498条为10个商品, 2条为6个商品
- result_goods 是 mix_rank_goods 的子集（已验证）
- result_score 是从 mix_rank_score 中取出的对应值（非重新计算）
- 顺序为最终展示顺序（坑位0~9）

#### 前序曝光商品（pre_*，0~9个商品）

| 字段 | 类型 | 元素类型 | 说明 |
|------|------|---------|------|
| `pre_goods` | array | string | 前序已曝光商品ID列表 |
| `pre_goods_cat1` | array | string | 前序商品C1类目 |
| `pre_goods_cat` | array | string | 前序商品C3类目 |

- 前序商品数量: min=0, max=9, mean=3.2
- 非空占比: 228/500 (45.6%)
- 前序商品是用户在当前请求之前已经看过的商品，打散时需避免类目重复

---

## 2. 商品向量 (`vec_filtered.parquet`)

### 概览

- **行数**: 298,549（每个商品一行，无重复）
- **列数**: 3
- **覆盖率**: 100%（采样日志中出现的所有商品均有向量）

### 字段详情

| 字段 | 类型 | 说明 |
|------|------|------|
| `goods_id` | int64 | 商品ID |
| `img_vec` | array<float32> | 图像向量，128维 |
| `text_vec` | array<float32> | 文本向量，128维 |

### 向量用途

- 用于计算 **标准2（向量相似度）** 的商品相似度
- 向量为单位向量（L2归一化），点积即cosine相似度，范围 [-1, 1]
- 相似度 = `(max(dot(img_vec_i, img_vec_j), dot(text_vec_i, text_vec_j)) + 1) / 2`，映射到 [0, 1]

---

## 3. V2 评估标准

> 对输出的10个商品进行评估
> V2 输入约束: 算法仅可使用 rank, fst_rank, 向量emb 作为输入 (不可使用 cat1/cat3)
> 不允许仅调整10个输出商品的rank绝对值排序 (除非所有请求评估均满足更好)
> 按前序 n_pre 个数，拆分数据，分开评估
> 按请求维度看，标准1、3更好，标准2只要求60%的请求"好"即可

### 标准1 曝光C3类目维度 (辅助评估，不可作为算法输入)
- 包含前序已经曝光的与新出商品前4坑位的C3类目去重数
- 包含前序已经曝光的与新出商品前10坑位的C3类目去重数

### 标准2 曝光向量相似度 (60%通过率)
相似度采用 `(max(图片向量dot, 文本向量dot) + 1) / 2`，值域 [0, 1]

对每个坑位 i (0-9)，计算 result[i] 与前序已曝光商品 (pre_goods + result[0..i-1]) 的相似度：
1. 每一个坑位计算包含前序已经曝光商品的相似度的最大值
2. 每一个坑位计算包含前序已经曝光商品的相似度的平均值
3. 每一个坑位计算包含前序已经曝光商品的相似度用曝光概率做加权的平均值
4. 每一个坑位计算包含前序已经曝光商品的 Max(0.7, 相似度) 的平均值
5. 每一个坑位计算包含前序已经曝光商品的 Max(0.7, 相似度) 用曝光概率做加权的平均值

每个子指标跨坑位取均值，得到请求级指标。
请求"好"的定义: 该请求的所有5个子指标 ≤ baseline。
VecSim_pass_rate = "好"请求数 / 总请求数, 需 ≥ 60%。

曝光概率权重: pre_goods 权重=1.0, result[j] 权重=exposure_probs[j]

### 标准3 商品rank维度
- 新出4个商品的首坑rank乘以曝光概率的和 (RankValue_top4, 用 fst_score)
- 新出10个商品的rank乘以曝光概率的和 (RankValue_top10, 用 score)
- 新出后6个商品的rank乘以曝光概率的和 (RankValue_bottom6, 用 score)

### 约束
- 标准1和标准3不下降的前提下，标准2通过率 ≥ 60%
- V2输入约束: 算法仅可使用 rank, fst_rank, 向量emb (不可使用 cat1/cat3)
- 不允许仅调整10个输出商品的rank绝对值排序
- 例外：如果对所有请求评估均满足更好，则允许破坏该规则

---

## 4. 评估标准与数据字段映射

| 标准 | 计算内容 | 使用字段 |
|------|---------|---------|
| **标准1** C3类目去重 | 前4坑/前10坑的C3类目去重数（含前序曝光） | `result_cat` + `pre_goods_cat` (仅评估, 非算法输入) |
| **标准2** 向量相似度 | 5个子指标, 每坑位计算与前序已曝光商品的相似度 | `result_goods` + `pre_goods` → 查 `vec_filtered` |
| **标准3** rank加权 | 首坑rank×曝光概率(前4) + rank×曝光概率(前10) + rank×曝光概率(后6) | 前4用 `result_fst_score`，前10/后6用 `result_score` |

### 标准详解

#### 标准1: C3类目维度 (辅助评估)
```
Cat3Diversity_top4 = |unique(pre_goods_cat ∪ result_cat[:4])|
Cat3Diversity_top10 = |unique(pre_goods_cat ∪ result_cat[:10])|
```
- 越高越好（类目越多样）
- 前序已曝光的类目也算入去重集合
- **V2: 仅辅助评估, 不允许使用C3作为算法输入**

#### 标准2: 向量相似度维度 (60%通过率)
```
# 向量均为单位向量（L2归一化），点积即cosine相似度，范围[-1, 1]
sim_img = dot_product(img_vec[g_i], img_vec[g_j])
sim_text = dot_product(text_vec[g_i], text_vec[g_j])
similarity = (max(sim_img, sim_text) + 1) / 2  # 映射到 [0, 1]

# 对每个坑位 i (0-9), 计算 result[i] 与前序已曝光商品 (pre_goods + result[0..i-1]) 的相似度:
# 子指标1: max(similarity)  — 最大相似度
# 子指标2: mean(similarity)  — 平均相似度
# 子指标3: weighted_mean(similarity, weights)  — 曝光概率加权平均
# 子指标4: mean(Max(0.7, similarity))  — Max(0.7, sim) 的均值
# 子指标5: weighted_mean(Max(0.7, similarity), weights)  — Max(0.7, sim) 的加权均值

# 权重: pre_goods=1.0, result[j]=exposure_probs[j]
# 每个子指标跨坑位取均值 → 请求级指标
# 请求"好" = 所有5个子指标 ≤ baseline
# VecSim_pass_rate = 好请求数 / 总请求数, 需 ≥ 60%
```
- 越低越好（商品向量越分散）
- 前序曝光商品的向量也参与计算

#### 标准3: 商品rank维度
```
# 前4个商品用首坑rankscore
RankValue_top4 = sum(result_fst_score[i] * exposure_prob[i] for i in range(4))
# 全10个商品用rankscore
RankValue_top10 = sum(result_score[i] * exposure_prob[i] for i in range(10))
# 后6个商品用rankscore
RankValue_bottom6 = sum(result_score[i] * exposure_prob[i] for i in range(4, 10))
```
- 越高越好（高rank商品排在前面）

#### 坑位曝光概率

来源: 线上曝光率统计（双列布局，每对坑位曝光率相同）

| 坑位 | 曝光率 |
|------|--------|
| 0 | 1.0 |
| 1 | 1.0 |
| 2 | 0.418337 |
| 3 | 0.418337 |
| 4 | 0.167033 |
| 5 | 0.167033 |
| 6 | 0.140998 |
| 7 | 0.140998 |
| 8 | 0.122709 |
| 9 | 0.122709 |

```python
EXPOSURE_PROBS = [1.0, 1.0, 0.418337, 0.418337, 0.167033,
                  0.167033, 0.140998, 0.140998, 0.122709, 0.122709]
```

---

## 5. 打散算法设计约束

### V2 输入约束 (CRITICAL)

- **仅可以使用**: `rank` (score), `fst_rank` (fst_score), 和向量 `emb` (img_vec, text_vec) 作为打散算法输入
- **不可以使用**: `cat1`, `cat3` 类目信息作为算法输入（仅用于辅助评估）
- **不允许**: 仅调整10个输出商品的rank绝对值排序（除非所有请求评估均满足更好）

### 逐坑贪心排序大原则

采用**逐坑排序**的方式生成最终10个商品的序列：

```
输入:
  - 前序品 (pre_goods): 已曝光商品列表，含向量、类目信息
  - 候选集合 (mix_rank_*): ~1000个候选商品，每个商品有:
    · rankingscore首 (fst_score): 用于排前4坑的rankscore
    · rankingscore末 (score): 用于排4-10坑的rankscore
    · embedding (img_vec, text_vec)
    · 类目信息 (cat1, cat3) — V2: 不可作为算法输入

流程:
  for pit in range(10):
      1. 确定当前坑位用哪个rankscore:
         - pit 0~3: 使用 rankingscore首 (fst_score)
         - pit 4~9: 使用 rankingscore末 (score)
      
      2. 基于原始rankscore + 多样性因素 → 生成新的rankscore
         (V2: 多样性因素只能基于向量相似度，不能使用类目信息)
      
      3. 基于新rankscore贪心选择TOP1 → 放入当前坑位
      
      4. 已选择的品加入"已曝光品"集合，继续排下一个坑位
   
  直到排完前10坑为止
```

### 关键注意事项：双rankscore体系

| 坑位 | 排序时使用的rankscore | 说明 |
|------|---------------------|------|
| 0~3 (前4坑) | `rankingscore首` (fst_score) | 前4坑要尽量吸引用户点击，建模目标不同 |
| 4~9 (后6坑) | `rankingscore末` (score) | 后6坑使用整体价值rankscore |

**最终评估时**:
- 标准3的 `RankValue_top4` 用 `rankingscore首` (fst_score) × 曝光概率
- 标准3的 `RankValue_top10` 用 `rankingscore末` (score) × 曝光概率
- 标准3的 `RankValue_bottom6` 用 `rankingscore末` (score) × 曝光概率
- **最终想要的价值排序按 `rankingscore末` 统计计算**，即10个item的list的整体价值

---

## 6. 线上Baseline指标

V2 标准1 和 标准3 与 V1 相同。标准2 为新增指标，需运行 `python prepare.py --verify-baseline` 获取。

### 标准1 & 标准3 (与 V1 相同)

| 指标 | mean |
|------|------|
| Cat3Diversity_top4 (C3去重) | 6.511 |
| Cat3Diversity_top10 (C3去重) | 12.266 |
| RankValue_top4 (fst_rank×曝光) | 1506.333 |
| RankValue_top10 (rank×曝光) | 5607.246 |
| RankValue_bottom6 (rank×曝光) | 862.514 |

### 标准2 (V2新增 — 需运行 --verify-baseline 获取)

| 指标 | 说明 |
|------|------|
| VecSim_max | 每坑位最大相似度的跨坑位均值 |
| VecSim_mean | 每坑位平均相似度的跨坑位均值 |
| VecSim_weighted_mean | 每坑位加权平均相似度的跨坑位均值 |
| VecSim_max07_mean | 每坑位 Max(0.7, sim) 均值的跨坑位均值 |
| VecSim_max07_weighted_mean | 每坑位 Max(0.7, sim) 加权均值的跨坑位均值 |
| VecSim_pass_rate | 请求中所有5个子指标 ≤ baseline 的比例, 需 ≥ 60% |

---

## 7. 待确认问题

无。所有V2评估标准已明确。

# Section 7 Experiments

TinyStories ~17M 参数 Transformer LM。实验分两轮：**g27 低资源**（41M tokens）与 **g52/g68 全量**（327M tokens，作业默认配置）。

Learning curves：[wandb cs336-section7](https://wandb.ai/steven144-nanjing-university/cs336-section7)（含 step 与 wall-clock time）。

---

## 作业 Deliverables

| Problem | 分值 | 手册要求（摘要） | 交付物 | 状态 |
| ------- | --- | ---------------- | ------ | ---- |
| **experiment_log** | 3 | logging code + experiment log | `train.py` + 本文档 | ✅ |
| **learning_rate (a)** | 3 | LR curves + search strategy + val≤1.45 | 全量 7 LR + **6e-3**；最佳 `ts_bs64_lr6em3` val=**1.327** | ✅ |
| **learning_rate (b)** | ↑ | 含 divergent run 的 LR curves + 分析 | `ts_lr_diverge_5em2`（val 1.61，显著退化）；另见无 RMSNorm@3e-3 NaN | ✅ |
| **batch_size_experiment** | 1 | batch curves + 讨论 | 低资源 ✅；全量 bs16/64/128 + LR scaling ✅ | ✅ |
| **generate** | 1 | ≥256 tokens + fluency + 2 因素 | `generate_full/` 7 组 decoding | ✅ |
| **layer_norm_ablation** | 1 | 无 RMSNorm curve + 评论 | 低资源 ✅；全量 lr=3e-4/3e-3 ✅ | ✅ |
| **pre_norm_ablation** | 1 | post-norm vs pre-norm | 低资源 ✅；全量 lr=3e-4/3e-3 ✅ | ✅ |
| **no_pos_emb** | 1 | RoPE vs NoPE | 低资源 ✅；全量 lr=3e-4/3e-3 ✅ | ✅ |
| **swiglu_ablation** | 1 | SwiGLU vs SiLU + 讨论 | 低资源 ✅；全量 lr=3e-4/3e-3 ✅ | ✅ |
| **main_experiment (OWT)** | 2 | OWT curve + 对比 + 生成 | 训练 + 生成（`generate_owt/`） | ✅ |

---

## Infrastructure

| 组件 | 路径 |
| ---- | ---- |
| 训练 | `train.py` |
| 生成 | `generate_text.py` |
| 启动器 | `run_section7.sh`, `run_section7_wave2.sh`, `run_section7_g52_wave2.sh` |
| Checkpoints / logs | `experiments/section7/`（gitignored） |
| 交付用图表 | `figures/section7/` |
| wandb | [cs336-section7](https://wandb.ai/steven144-nanjing-university/cs336-section7) |
| 命名工具 | `scripts/section7_names.sh`, `scripts/organize_wandb_section7.py` |

### WandB 命名

Project 固定 **`cs336-section7`**。本地 checkpoint 目录名不变（如 `ts_lr3em3`）；WandB 用 **group + 可读 run name** 分层。

**Group：** `{dataset}/{tier}/{category}`，其中 `tier` = `full`（327M）/ `low`（41M）。

| category | 含义 | group 示例 |
|----------|------|------------|
| `lr_sweep` | LR 扫描 | `ts/full/lr_sweep` |
| `lr_diverge` | Edge of stability | `ts/full/lr_diverge` |
| `batch` | Batch size（固定 lr=3e-4） | `ts/low/batch` |
| `batch_lr` | Batch + LR scaling | `ts/full/batch_lr` |
| `ablation` | 架构消融 | `ts/full/ablation` |
| `baseline` | 重复 baseline | `ts/full/baseline` |
| `main` | OWT 主实验 | `owt/full/main` |

**Run name：**

| 类型 | 格式 | 示例 |
|------|------|------|
| LR / batch | `bs{B}_lr{lr}` | `bs64_lr3e-3` |
| Ablation | `{variant}__bs{B}_lr{lr}` | `no_rope__bs64_lr3e-4` |
| OWT | `bs{B}_lr{lr}__{k}k` | `bs64_lr3e-3__20k` |

**旧名 → 新名（常见）：**

| 旧名 | 新 group | 新 run name |
|------|----------|-------------|
| `ts_lr3em3`（全量） | `ts/full/lr_sweep` | `bs64_lr3e-3` |
| `ts_lr3em3`（低资源） | `ts/low/lr_sweep` | `bs16_lr3e-3` |
| `ts_lr_diverge_2em2` | `ts/full/lr_diverge` | `bs64_lr2e-2` |
| `ts_bs128_lr6em3` | `ts/full/batch_lr` | `bs128_lr6e-3` |
| `ts_ablate_no_rope_lr3em3` | `ts/full/ablation` | `no_rope__bs64_lr3e-3` |
| `owt_bs64_lr3em3_20000` | `owt/full/main` | `bs64_lr3e-3__20k` |

> `em` 后缀（`3em3` = `3e-3`）仅用于 checkpoint 目录；WandB 显示统一为 `3e-3`。

```bash
.venv/bin/python scripts/organize_wandb_section7.py          # 预览
.venv/bin/python scripts/organize_wandb_section7.py --apply  # 同步云端
```

---

## 配置对照

| 参数 | 作业默认 / g52 全量 | g27 低资源 |
| ---- | ------------------- | ---------- |
| 机器 | g52 GPU 4–7 等 | g27 GPU 0–3（与 vLLM 共享） |
| tokens | **327,680,000** | 40,960,000 |
| batch size | **64** | 16 |
| max iters | **20,000** | 10,000 |
| context length | 256 | 256 |
| warmup | **2,000** | 1,000 |
| val loss 目标 | ≤ **1.45** | ≤ 2.00 |
| 峰值显存 | ~8.9 GB | ~2.3 GB |

---

## 实验进度

| ID | Problem | g27 低资源 | g52/g68 全量 |
| -- | ------- | ---------- | ------------ |
| 7.1 | experiment_log | ✅ wandb | ✅ 同 project |
| 7.2a | learning_rate sweep | ✅ 7 runs | ✅ 7 runs + **6e-3** 补跑 |
| 7.2b | edge of stability | ✅ 2 runs | ✅ 4 runs；`5em2` 计为发散 ✅ |
| 7.2 | batch_size | ✅ bs=1/4/16 | ✅ bs16/64/128 + LR scaling |
| 7.2 | generate | ✅ 低资源 checkpoint | ✅ 全量 `ts_lr3em3` + 7 组 decoding |
| 7.3 | ablations | ✅ 4 runs | ✅ 8 runs（lr=3e-4/3e-3 各 4） |
| 7.4 | OWT | ❌ | ✅ 训练 + 生成 |

**机器分工：**

| 机器 | 任务 | 状态 |
| ---- | ---- | ---- |
| g27 | 低资源 LR / batch / ablation | ✅ |
| g52 | 全量 LR sweep、diverge、batch-lr | ✅ |
| g68 | batch-lr bs128、ablation lr=3e-4、OWT | ✅ |
| g26 | ablation lr=3e-3（×3） | ✅ |
| g32 | ablation silu lr=3e-3 | ✅ |

注意：全量与低资源 run **同名**时，`experiments/section7/ts_lr*` 的 checkpoint 已被全量覆盖；低资源数值保留在本文档与 wandb 历史中。

---

## 7.2a Learning Rate Sweep

**搜索策略**：对数均匀扫描 `1e-5 → 1e-2`（7 点），cosine decay 至 `min_lr=3e-5`。固定 batch=64、20k steps。主扫后在 batch-LR 实验中补跑 **6e-3**（`ts_bs64_lr6em3`，同配置，纳入 LR curve）。

### g52 全量（batch=64, 20k steps）

| Run | LR | Final valid loss | Wall time | 状态 |
| --- | -- | ---------------- | --------- | ---- |
| **ts_bs64_lr6em3** | **6e-3** | **1.327** | ~63 min | ✅ **全量最佳**（补跑） |
| **ts_lr3em3** | 3e-3 | 1.331 | ~62 min | ✅ LR sweep |
| ts_lr1em2 | 1e-2 | 1.332 | ~62 min | ✅ 未发散 |
| ts_lr1em3 | 1e-3 | 1.371 | ~62 min | ✅ |
| **ts_lr3em4** | 3e-4（默认） | **1.469** | ~72 min | ✅ 达标（≤1.45） |
| ts_lr1em4 | 1e-4 | 1.653 | ~72 min | ✅ |
| ts_lr3em5 | 3e-5 | 1.899 | ~72 min | ✅ |
| ts_lr1em5 | 1e-5 | 2.065 | ~72 min | ✅ |

Wave 1（GPU 4–7）：1e-5 ~ 3e-4；Wave 2（GPU 4–6）：1e-3 ~ 1e-2；**6e-3** 在 batch-LR 阶段补跑（g52 GPU 6）。

**结论：**

- 最优 LR 为 **6e-3**（val=**1.327**），略优于 3e-3（1.331）和 1e-2（1.332）；最优点在 3e-3 ~ 1e-2 之间。
- Loss 随 LR 呈近似 U 型：1e-5（2.07）→ 6e-3（1.33）→ 1e-2（1.33）；继续增大至 2e-2 / 5e-2 见 7.2b。
- 默认 lr=3e-4（1.469）已满足 ≤1.45；更好 checkpoint 用 `ts_bs64_lr6em3` 或 `ts_lr3em3`。

![LR sweep + edge of stability](figures/section7/lr_sweep_full.png)

> 蓝线 = LR sweep + 6e-3；橙虚线 = 7.2b diverge 2e-2/5e-2。脚本：`scripts/plot_lr_sweep.py`

### g27 低资源（batch=16, 10k steps, ~10 min/run）

| Run | LR | Final valid loss | Wall time | 备注 |
| --- | -- | ---------------- | --------- | ---- |
| **ts_lr3em3** | **3e-3** | **1.690** | ~925s | 低资源最佳 |
| ts_lr1em3 | 1e-3 | 1.697 | ~517s | |
| ts_lr_diverge_5em3 | 5e-3 | 1.713 | ~527s | 未发散 |
| ts_lr1em2 | 1e-2 | 1.766 | ~927s | 未发散 |
| ts_lr3em4 | 3e-4 | 1.832 | ~594s | baseline |
| ts_lr1em4 | 1e-4 | 2.128 | ~594s | |
| ts_lr3em5 | 3e-5 | 2.469 | ~594s | |
| ts_lr1em5 | 1e-5 | 2.669 | ~594s | |

### 低资源 vs 全量（相同 LR）

| LR | 低资源 | 全量 | Δ |
| -- | ------ | ---- | - |
| 1e-2 | 1.766 | 1.332 | −0.43 |
| 3e-3 | 1.690 | **1.331** | −0.36 |
| 1e-3 | 1.697 | 1.371 | −0.33 |
| 3e-4 | 1.832 | 1.469 | −0.36 |
| 1e-4 | 2.128 | 1.653 | −0.48 |
| 3e-5 | 2.469 | 1.899 | −0.57 |
| 1e-5 | 2.669 | 2.065 | −0.60 |

全量（8× tokens + 4× batch）对所有 LR 均大幅提升。低资源最优 3e-3 在全量下仍接近最优（全量最佳为补跑的 6e-3）。

---

## 7.2b Edge of Stability

与 7.2a 同配置（全量 bs=64、20k steps），在 sweep 外继续增大 LR。**曲线见上方 LR sweep 图（橙虚线）。**

| Run | 配置 | LR | Final valid loss | Wall time | 是否发散 |
| --- | ---- | -- | ---------------- | --------- | -------- |
| ts_lr_diverge_5em3 | 低资源 | 5e-3 | 1.713 | ~9 min | 否 |
| ts_lr_diverge_1em2 | 低资源 | 1e-2 | 1.766 | ~15 min | 否 |
| ts_lr1em2 | **全量** | 1e-2 | 1.332 | ~62 min | 否 |
| ts_lr_diverge_2em2 | **全量** | 2e-2 | **1.358** | ~73 min | 否 |
| ts_lr_diverge_5em2 | **全量** | 5e-2 | **1.608** | ~73 min | **是**（未 NaN，但相对最优 +0.28，视为发散） |
| ts_ablate_no_rmsnorm_lr3em3 | 全量 ablation | 3e-3 | **NaN** | ~68 min | **是**（训练崩溃） |

**全量 LR → val：** 1e-5（2.07）→ … → 6e-3（**1.327**）→ 1e-2（1.33）→ 2e-2（1.36）→ 5e-2（**1.61，发散**）

- 低资源：最优 3e-3（1.690）优于 5e-3 / 1e-2；1e-2 有过拟合迹象（train 1.64 vs valid 1.77）。
- 全量：最优在 **6e-3**（1.327）；2e-2（1.358）仍接近最优；**5e-2（1.608）明显越过 edge of stability**——未 NaN，但 loss 相对最优恶化约 +0.28，训练已实质发散/退化。
- 最优点（6e-3）并不贴着 stability edge：edge 约在 2e-2~5e-2，中间仍有较大 margin。
- 更极端的发散：无 RMSNorm + lr=3e-3 → **NaN**，说明 normalization 对高 LR 稳定性至关重要。

---

## 7.2 Batch Size Experiment

### 低资源（~41M tokens, lr=3e-4）

| Run | Batch | Iters | Final valid loss | Wall time |
| --- | ----- | ----- | ---------------- | --------- |
| ts_bs16 | 16 | 10,000 | 1.832 | ~517s |
| ts_bs4 | 4 | 39,500 | 1.925 | ~972s |
| ts_bs1 | 1 | 159,500 | 2.285 | ~2828s |

### 全量（327M tokens）

| Run | Batch | LR | Final valid loss | Wall time | 状态 |
| --- | ----- | -- | ---------------- | --------- | ---- |
| `ts_lr3em4` / `ts_bs64` | 64 | 3e-4 | 1.469 | ~73 min | ✅ baseline |
| **`ts_bs64_lr6em3`** | 64 | **6e-3** | **1.327** | ~63 min | ✅ |
| **`ts_bs16`** | 16 | 3e-4 | **1.473** | ~101 min | ✅ 80k steps |
| `ts_bs128` | 128 | 3e-4 | 1.512 | ~63 min | ✅ |
| **`ts_bs128_lr6em3`** | 128 | **6e-3** | **1.311** | ~61 min | ✅ 最佳 |
| `ts_bs128_lr1em2` | 128 | 1e-2 | 1.312 | ~61 min | ✅ |
| `ts_bs128_lr3em3` | 128 | 3e-3 | 1.324 | ~61 min | ✅ |
| `ts_bs128_lr2em2` | 128 | 2e-2 | 1.325 | ~61 min | ✅ |

bs=1/4 全量因步数过多已取消；低资源版可引用。

**讨论：**

- 固定 token budget、lr=3e-4：bs16 **1.473** ≈ bs64 **1.469** > bs128 **1.512**；小 batch 并非更差。
- bs128@3e-4 差于 bs64，**线性 LR scaling**（×2 batch → ×2 LR）有效：bs128@6e-3 **1.311** 优于 bs64@3e-4。
- 固定 token budget 下 bs128 wall time 仅比 bs64 快 ~14%（非 2×），因每 step 计算量更大。
- bs16 需 80k steps、~101 min，最慢；bs128@6e-3 质量最佳且 ~61 min。

---

## 7.2 Generate Text

**Prompt**：`Once upon a time`  
**Checkpoint（全量）**：`ts_lr3em3/checkpoint_final.pt`（val=**1.331**）  
**运行**：g52 GPU 6，2026-07-24  
**Code**：`generate_text.py` + `cs336_basics/decoding.py`

### Decoding 参数对比

| 配置 | temperature | top_p | fluency | 备注 |
| ---- | ----------- | ----- | ------- | ---- |
| t0_greedy | 0 | — | ★★★★ | 最连贯，情节完整（Lily + Mr. Bear） |
| t05_p095 | 0.5 | 0.95 | ★★★★ | 与 greedy 类似，略有多样性 |
| t08_p095 | 0.8 | 0.95 | ★★★★ | 故事完整（Sue + shell），对话自然 |
| t08_p09 | 0.8 | 0.9 | ★★★☆ | 略保守，"weigh the leaves" 语义略怪 |
| t08_p099 | 0.8 | 0.99 | ★★★★ | 与 t08_p095 接近 |
| t10_p095 | 1.0 | 0.95 | ★★★☆ | 出现重复短语 |
| t12_p095 | 1.2 | 0.95 | ★★☆ | 语义崩坏 |

**推荐**：temperature=0~0.8，top_p=0.9~0.95。全量明显优于低资源。

**影响输出质量的两个因素：**

1. **Temperature**：越低越 deterministic；≥1.2 时小模型易 nonsense。
2. **Top-p**：在 t=0.8 下 0.9 vs 0.99 差异不大；主效应来自 temperature。

### 全量最佳样例（t=0 greedy）

```
Once upon a time, there was a little girl named Lily. She had a big, soft teddy bear named Mr. Bear. Lily loved Mr. Bear very much. They played together every day.
One day, Lily and Mr. Bear went to the park. They played on the swings and the slide. Lily was very happy. But then, a big wind came and blew Mr. Bear away. Lily was very sad.
Lily looked for Mr. Bear everywhere. She asked her friends to help her find him. They all looked for Mr. Bear. At last, they found him under a big tree. Lily was so happy to have her bear back. She hugged Mr. Bear and said, "Thank you for helping me find my bear!"
<|endoftext|>
```

完整输出：`experiments/section7/generate_full/`  
低资源对照：`experiments/section7/generated_lr3em3.txt`

---

## 7.3 Ablations

### 低资源（lr=3e-4, batch=16, 10k steps；baseline=1.832）

| Run | 改动 | Final valid loss | Δ |
| --- | ---- | ---------------- | - |
| ts_ablate_post_norm | pre-norm → post-norm | 1.824 | −0.008 |
| ts_ablate_no_rmsnorm | 移除 RMSNorm | 1.840 | +0.008 |
| ts_ablate_silu_ffn | SwiGLU → SiLU | 1.888 | +0.056 |
| ts_ablate_no_rope | 移除 RoPE | 1.926 | +0.094 |

### 全量 lr=3e-4（baseline val=**1.469**）

| Run | 改动 | Final valid loss | Δ |
| --- | ---- | ---------------- | - |
| ts_ablate_post_norm | post-norm | **1.464** | −0.005 |
| ts_ablate_no_rmsnorm | 无 RMSNorm | **1.483** | +0.014 |
| ts_ablate_silu_ffn | SiLU FFN | **1.492** | +0.023 |
| ts_ablate_no_rope | NoPE | **1.547** | +0.078 |

### 全量 lr=3e-3（baseline val=**1.331**）

| Run | 改动 | Final valid loss | Δ | 备注 |
| --- | ---- | ---------------- | - | ---- |
| ts_ablate_silu_ffn | SiLU FFN | **1.341** | +0.010 | ✅ |
| ts_ablate_post_norm | post-norm | **1.358** | +0.027 | ✅ |
| ts_ablate_no_rope | NoPE | **1.395** | +0.064 | ✅ |
| ts_ablate_no_rmsnorm | 无 RMSNorm | **NaN** | — | 训练崩溃 ⚠️ |

**结论：**

- **RoPE** 影响最大（全量 +0.06~0.08）；NoPE 仍收敛但明显更差。
- **SwiGLU** 有稳定收益（+0.01~0.02）；参数量匹配下 SiLU 略差。
- **RMSNorm**：lr=3e-4 下移除仅 +0.014；lr=3e-3 下 **NaN**——高 LR 训练的关键。
- **post-norm vs pre-norm**：差异很小（≤0.03），低资源甚至 post-norm 略好。

---

## 7.4 OpenWebText

**Run：** `owt_bs64_lr3em3_20000`（g68 GPU 0，2026-07-24，~222 min）  
**wandb：** [owt_bs64_lr3em3_20000](https://wandb.ai/steven144-nanjing-university/cs336-section7/runs/zn79v918)  
**checkpoint：** `experiments/section7/owt_bs64_lr3em3_20000/checkpoint_final.pt`  
**生成样例：** `experiments/section7/generate_owt/`

| 参数 | TinyStories（`ts_lr3em3`） | OWT |
|------|---------------------------|-----|
| 词表 | 10,000 | 32,000 |
| tokens | 327M | 327M |
| batch / lr | 64 / 3e-3 | 64 / 3e-3 |
| final valid loss | **1.331** | **3.985** @ step 19500 |
| wall time | ~62 min | ~222 min（13340s） |

### Learning curve（valid loss 采样）

| step | OWT | TinyStories |
|------|-----|-------------|
| 0 | 10.39 | 9.25 |
| 500 | 5.71 | 2.53 |
| 4500 | 4.54 | 1.68 |
| 19500 | **3.99** | **1.33** |

### Loss 差异解释

- OWT 词表 32k，随机 baseline CE ≈ log(32000) ≈ **10.4**（实测 step 0 = 10.39）；TinyStories 词表 10k，step 0 = 9.25。OWT 起点更高且下降更慢。
- OWT 长尾、主题/语法更多样；17M + 327M tokens 仍严重 underfit（train ~4.0）。
- 同 compute 下 OWT ~4.0 vs TS ~1.33，主要来自 **任务难度 + 词表**，而非训练 bug。

### 生成与 fluency（prompt: `Once upon a time`）

| 配置 | fluency | 现象 |
|------|---------|------|
| t=0 greedy | ★☆☆ | 短句后陷入重复："the world is not a place to live" 循环 |
| t=0.8, p=0.95 | ★★☆ | 局部像新闻/对话，但人名、时间线、因果混乱 |
| t=1.0, p=0.95 | ★☆☆ | 主题漂移、乱码式 token 组合 |

**为何同 compute 质量更差：**

1. **数据复杂度：** OWT 需 world knowledge + 长程依赖；TinyStories 是受限童话语域。
2. **Underfitting：** val≈4.0 仍接近弱预测；TS val≈1.33 已拟合较好。
3. **Prompt 不匹配：** `Once upon a time` 偏童话，OWT 分布更偏新闻/论坛，greedy 易落入高频句式重复。

~~`owt_bs64_lr3em3_80000`~~ 已在 step ~300 kill（预估 ~19 h > 6 h 上限）。

**生成命令：**

```bash
CKPT=experiments/section7/owt_bs64_lr3em3_20000/checkpoint_final.pt
CUDA_VISIBLE_DEVICES=0 .venv/bin/python generate_text.py \
  --checkpoint $CKPT \
  --vocab-file data/owt_vocab.json --merges-file data/owt_merges.txt \
  --vocab-size 32000 --temperature 0 --top-p 0.95 \
  --output experiments/section7/generate_owt/generated_t0_greedy.txt
```

---

## Checkpoint 索引

| 用途 | 路径 |
| ---- | ---- |
| **全量最佳**（lr=6e-3, val=1.327） | `experiments/section7/ts_bs64_lr6em3/checkpoint_final.pt` |
| LR sweep 最佳（lr=3e-3, val=1.331） | `experiments/section7/ts_lr3em3/checkpoint_final.pt` |
| 达标默认（lr=3e-4, val=1.469） | `experiments/section7/ts_lr3em4/checkpoint_final.pt` |
| batch-LR 最佳（bs128@6e-3, val=1.311） | `experiments/section7/ts_bs128_lr6em3/checkpoint_final.pt` |
| OWT 模型 + 生成 | `owt_bs64_lr3em3_20000/checkpoint_final.pt` + `generate_owt/` |
| 生成样例（低资源） | `experiments/section7/generated_lr3em3.txt` |
| 生成样例（全量） | `experiments/section7/generate_full/` |

---

## 复现命令

```bash
# 低资源
LOW_RESOURCE=1 ./run_section7.sh lr-sweep 0 0 1 2 3

# 全量（需 ~10GB 空闲显存）
./run_section7.sh lr-sweep 0 4 5 6 7

# 文本生成（全量）
CKPT=experiments/section7/ts_lr3em3/checkpoint_final.pt
CUDA_VISIBLE_DEVICES=6 .venv/bin/python generate_text.py \
  --checkpoint $CKPT --temperature 0 --top-p 0.95 \
  --output experiments/section7/generate_full/generated_t0_greedy.txt

# OWT（327M tokens = 20k steps）
./run_section7.sh owt <gpu> --batch-size 64 --max-iters 20000 --lr 3e-3

# Batch-LR scaling
./run_section7.sh batch-lr 128 4:6e-3 5:3e-3 6:1e-2

# Ablation（lr=3e-3）
./run_section7.sh ablations --lr 3e-3 0 1 2 3
```

---

## 已知问题

- `generate_text.py` 最初用 `torch.optim.AdamW` 加载 checkpoint 会失败（自定义 AdamW state 格式不同），已改为只加载 model weights。
- `run_section7_wave2.sh` 的 diverge 段有重复启动逻辑，不影响 checkpoint 有效性。
- g82/g83/g85（glibc 2.17）无法运行当前 `.venv`；g26 为 glibc 2.35，环境正常。

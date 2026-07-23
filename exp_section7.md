# Section 7 Experiments

TinyStories ~17M 参数 Transformer LM。实验分两轮：**g27 低资源**（41M tokens）和 **g52/g68 全量**（327M tokens，作业默认配置）。

---

## 作业 Deliverables 清单

本节对照手册要求，列出每项 deliverable、对应产物与完成状态。Learning curves 均在 [wandb cs336-section7](https://wandb.ai/steven144-nanjing-university/cs336-section7)（含 step 与 wall-clock time）。


| Problem                   | 分值  | 手册 Deliverable（摘要）                        | 我们的交付物                                 | 状态    |
| ------------------------- | --- | ----------------------------------------- | -------------------------------------- | ----- |
| **experiment_log**        | 3   | logging code + experiment log 文档          | `train.py` + 本文档                       | ✅     |
| **learning_rate (a)**     | 3   | LR curves + search strategy + val≤1.45 模型 | 全量 7 LR ✅；最佳 `ts_lr3em3` val=**1.331** | ✅     |
| **learning_rate (b)**     | ↑   | 含 divergent run 的 LR curves + 分析          | 至 1e-2 未发散；2e-2/5e-2 🔄 g52       | ⚠️    |
| **batch_size_experiment** | 1   | batch curves + 讨论                         | 低资源 bs=1/4/16；g68 bs=16/128 进行中        | ⚠️    |
| **generate**              | 1   | ≥256 tokens dump + fluency + 2 因素         | `generate_full/` 7 组 decoding         | ✅     |
| **layer_norm_ablation**   | 1   | 无 RMSNorm curve + 评论                      | 低资源 ✅；g68 全量 🔄                        | ⚠️    |
| **pre_norm_ablation**     | 1   | post-norm vs pre-norm curve               | 低资源 ✅；g68 全量 🔄                        | ⚠️    |
| **no_pos_emb**            | 1   | RoPE vs NoPE curve                        | 低资源 ✅；g68 全量 🔄                        | ⚠️    |
| **swiglu_ablation**       | 1   | SwiGLU vs SiLU curve + 讨论                 | 低资源 ✅；g68 全量 🔄                        | ⚠️    |
| **main_experiment (OWT)** | 2   | OWT curve + 对比 + 生成分析                     | 未开始                                    | ❌     |




### 各 Problem 详细说明



#### Problem (experiment_log) — 实验日志基础设施

**手册 Deliverable（原文）：**

> Deliverable：用于实验的 logging infrastructure code，以及本节后续作业问题所需的 experiment log（记录你尝试过的所有事情的文档）。

**手册要求：**

- 创建 experiment tracking infrastructure，能相对 gradient steps 和 wall-clock time 跟踪 loss curves
- 提交 logging code + experiment log 文档

**交付：**

- Code：`train.py` 集成 wandb；`eval_interval` 周期性算 valid loss；log 含 `step`、`elapsed`（秒）
- Log：本文档 + wandb project `cs336-section7`

---



#### Problem (learning_rate) — 调 learning rate

**(a) Hyperparameter sweep**

**手册 Deliverable（原文）：**

> Deliverable：多个 learning rates 对应的 learning curves。解释你的 hyperparameter search strategy。  
> Deliverable：一个在 TinyStories 上 validation loss（per-token）不超过 1.45 的模型。

**手册要求：**

- 多个 LR 的 learning curves；说明 search strategy
- 一个 TinyStories 上 **validation loss（per-token）≤ 1.45** 的模型

**交付：**

- Search strategy：对数均匀扫描 `1e-5 … 1e-2`；固定 batch；cosine decay 在末 step 到 min_lr
- **达标模型（全量）：**


| 项目               | 值                                                    |
| ---------------- | ---------------------------------------------------- |
| Run（最佳）          | `ts_lr3em3`                                          |
| LR               | 3e-3                                                 |
| Final valid loss | **1.331**                                            |
| Checkpoint       | `experiments/section7/ts_lr3em3/checkpoint_final.pt` |
| 配置               | batch=64, 20k steps, 327M tokens, ~64 min            |



| 项目               | 值                                                    |
| ---------------- | ---------------------------------------------------- |
| Run（默认 lr）       | `ts_lr3em4`                                          |
| LR               | 3e-4                                                 |
| Final valid loss | **1.469**                                            |
| Checkpoint       | `experiments/section7/ts_lr3em4/checkpoint_final.pt` |


- 低资源最佳 lr=3e-3（loss 1.690），未达 1.45，但满足 low-resource tip（≤2.0）

**(b) Edge of stability**

**手册 Deliverable（原文）：**

> Deliverable：逐渐增大 learning rate 的 learning curves，其中至少包含一个 divergent run，并分析这与 convergence rates 的关系。

**手册要求：**

- 逐渐增大 LR 的 curves，**至少一个 divergent run**
- 分析 divergent 点与最佳 LR 的关系

**交付：**

- 低资源：lr=5e-3、1e-2 均未发散
- 全量：lr=1e-2（`ts_lr1em2`, val=1.332）仍稳定，未出现 loss 爆炸或 NaN
- 全量最佳 lr=3e-3（1.331）优于 1e-2（1.332），最优点不在 stability edge
- **缺口：** 手册要求至少一个 divergent run；sweep 至全量 1e-2 仍稳定；**已启动** 2e-2/5e-2（g52 GPU 4/5）

---



#### Problem (batch_size_experiment) — Batch size

**手册 Deliverable（原文）：**

> Deliverable：不同 batch sizes 的 runs 对应的 learning curves。必要时应重新优化 learning rates。  
> Deliverable：用几句话讨论你关于 batch sizes 及其对训练影响的发现。

**手册要求：**

- batch 从 1 到 GPU 上限；至少包含 64、128 等
- 不同 batch 的 learning curves；必要时重调 LR
- 几句话讨论发现

**交付：**

- 低资源（41M tokens）：bs=1, 4, 16；结论见 7.2
- 全量：bs=64 即 `ts_lr3em4`（val=1.469）；g68 正在跑 bs=16（56%）、bs=128（50%）
- bs=1/4 全量因步数过多（32万~128万 steps）已取消；低资源版仍可引用
- **缺口：** 全量 bs=128 结果待完成；未对每个 batch 单独重调 LR

---



#### Problem (generate) — 生成文本

**手册 Deliverable（原文）：**

> Deliverable：至少 256 tokens 的文本 dump（或直到第一个 `<|endoftext|>` token），并简要评论该输出的 fluency，以及至少两个影响输出好坏的因素。

**手册要求：**

- 用 decoder + trained checkpoint 生成文本
- ≥256 tokens（或至 `<|endoftext|>`）
- fluency 简评 + 至少两个影响因素

**交付：**

- 全量 checkpoint：`ts_lr3em3/checkpoint_final.pt`（val=1.331，g52 GPU 6）
- 文件：`experiments/section7/generate_full/generated_*.txt`（7 组 decoding 参数）
- 低资源对照：`experiments/section7/generated_lr3em3.txt`
- Code：`generate_text.py` + `cs336_basics/decoding.py`
- 评论：见 7.2 Generate Text

---



#### Problem (layer_norm_ablation) — 移除 RMSNorm

**手册 Deliverable（原文）：**

> Deliverable：移除 RMSNorms 后训练的 learning curve，以及最佳 learning rate 对应的 learning curve。  
> Deliverable：几句话评论 RMSNorm 的影响。

**手册要求：**

- 移除 RMSNorm 后的 learning curve
- 几句话评论 RMSNorm 影响（可否用更低 LR 稳定？）

**交付：**

- Run：`ts_ablate_no_rmsnorm`（lr=3e-4）
- 低资源 valid loss 1.840 vs baseline 1.832；训练未崩溃
- g68 全量版进行中

---



#### Problem (pre_norm_ablation) — Post-norm

**手册 Deliverable（原文）：**

> Deliverable：post-norm Transformer 的 learning curve，并与 pre-norm 对比。

**手册要求：**

- post-norm 的 learning curve，与 pre-norm 对比

**交付：**

- Run：`ts_ablate_post_norm`
- 低资源：post-norm 1.824 vs pre-norm baseline 1.832（略好，差距小）
- g68 全量版进行中

---



#### Problem (no_pos_emb) — NoPE

**手册 Deliverable（原文）：**

> Deliverable：比较 RoPE 和 NoPE 表现的 learning curve。

**手册要求：**

- RoPE vs NoPE 的 learning curve 对比

**交付：**

- Run：`ts_ablate_no_rope`（NoPE）vs baseline（RoPE）
- 低资源：NoPE 1.926，RoPE baseline 1.832；RoPE 影响最大
- g68 全量版进行中

---



#### Problem (swiglu_ablation) — SwiGLU vs SiLU

**手册 Deliverable（原文）：**

> Deliverable：在 parameter counts 大致匹配的情况下，比较 SwiGLU 和 SiLU feed-forward networks 性能的 learning curve。  
> Deliverable：几句话讨论你的发现。

**手册要求：**

- 参数量大致匹配下 SwiGLU vs SiLU 的 learning curves
- 几句话讨论

**交付：**

- Run：`ts_ablate_silu_ffn`（SiLU, d_ff=2048）vs baseline SwiGLU
- 低资源：SiLU 1.888 vs SwiGLU 1.832；gating 有收益
- g68 全量版进行中

---



#### Problem (main_experiment) — OpenWebText

**手册 Deliverable（原文）：**

> Deliverable：你的 language model 在 OpenWebText 上的 learning curve。描述与 TinyStories 的 losses 差异；我们应如何解释这些 losses？  
> Deliverable：OpenWebText LM 生成的文本，格式与 TinyStories outputs 相同。该文本 fluency 如何？为什么即使使用与 TinyStories 相同的模型和 compute budget，输出质量仍更差？

**手册要求：**

- OWT 上同架构、同 iterations 的 learning curve
- 与 TinyStories losses 差异及解释
- OWT 生成文本 + fluency 分析（为何同 compute 质量更差）

**交付：** 未开始。数据已 tokenize：`data/tokenized/owt_train.npy`，可用 `./run_section7.sh owt [gpu]` 启动。

---



### Checkpoint 索引（交作业常用）


| 用途                             | 路径                                                   |
| ------------------------------ | ---------------------------------------------------- |
| **全量最佳模型**（lr=3e-3, val=1.331） | `experiments/section7/ts_lr3em3/checkpoint_final.pt` |
| 达标全量模型（lr=3e-4, val=1.469）     | `experiments/section7/ts_lr3em4/checkpoint_final.pt` |
| 生成样例（低资源）                      | `experiments/section7/generated_lr3em3.txt`          |
| 生成样例（全量，7 组 decoding）          | `experiments/section7/generate_full/`                |


---



## Infrastructure


| 组件                 | 路径                                                                             |
| ------------------ | ------------------------------------------------------------------------------ |
| 训练                 | `train.py`                                                                     |
| 生成                 | `generate_text.py`                                                             |
| 启动器                | `run_section7.sh`, `run_section7_wave2.sh`, `run_section7_g52_wave2.sh`        |
| Checkpoints / logs | `experiments/section7/`                                                        |
| wandb              | [cs336-section7](https://wandb.ai/steven144-nanjing-university/cs336-section7) |




## 实验进度


| ID   | Problem             | g27 低资源          | g52/g68 全量                     |
| ---- | ------------------- | ---------------- | ------------------------------ |
| 7.1  | experiment_log      | ✅ wandb          | ✅ 同 project                    |
| 7.2a | learning_rate sweep | ✅ 7 runs         | ✅ 7 runs（g52 wave1+2）          |
| 7.2b | edge of stability   | ✅ 2 runs         | 🔄 diverge 2e-2/5e-2 g52        |
| 7.2  | batch_size          | ✅ bs=1/4/16      | 🔄 bs=16/128 g68（~50–56%）      |
| 7.2  | generate            | ✅ 低资源 checkpoint | ✅ 全量 `ts_lr3em3` + 7 组 decoding |
| 7.3  | ablations           | ✅ 4 runs         | 🔄 4 runs g68（50–75%）          |
| 7.4  | OWT                 | ❌                | ❌                              |


---



## 配置对照


| 参数             | 作业默认 / g52 全量   | g27 低资源                |
| -------------- | --------------- | ---------------------- |
| 机器             | g52 GPU 4–7     | g27 GPU 0–3（与 vLLM 共享） |
| tokens         | **327,680,000** | 40,960,000             |
| batch size     | **64**          | 16                     |
| max iters      | **20,000**      | 10,000                 |
| context length | 256             | 256                    |
| warmup         | **2,000**       | 1,000                  |
| val loss 目标    | ≤ **1.45**      | ≤ 2.00                 |
| 峰值显存           | ~8.9 GB         | ~2.3 GB                |


---



## 7.2a Learning Rate Sweep

**搜索策略**：对数均匀扫描 `1e-5 → 1e-2`，cosine decay 在训练结束时降至 min_lr=3e-5。扫 LR 时固定 batch size，不重扫 batch。

### g52 全量（batch=64, 20k steps）


| Run           | LR       | Final valid loss | Wall time          | 状态          |
| ------------- | -------- | ---------------- | ------------------ | ----------- |
| **ts_lr3em3** | **3e-3** | **1.331**        | ~~3734s (~~62 min) | ✅ **全量最佳**  |
| ts_lr1em2     | 1e-2     | 1.332            | ~3739s             | ✅ 未发散       |
| ts_lr1em3     | 1e-3     | 1.371            | ~3728s             | ✅           |
| **ts_lr3em4** | 3e-4（默认） | **1.469**        | ~~4302s (~~72 min) | ✅ 达标（≤1.45） |
| ts_lr1em4     | 1e-4     | 1.653            | ~4302s             | ✅           |
| ts_lr3em5     | 3e-5     | 1.899            | ~4302s             | ✅           |
| ts_lr1em5     | 1e-5     | 2.065            | ~4302s             | ✅           |


Wave 1（GPU 4–7）：1e-5 ~ 3e-4；Wave 2（GPU 4–6）：1e-3 ~ 1e-2。baseline 重复 run 已取消（`ts_lr3em4` 已足够）。

**结论：**

- 最优 LR 为 **3e-3**（val=1.331），优于默认 3e-4（1.469）约 0.14。
- Loss 随 LR 呈近似 U 型：1e-5（2.07）→ 3e-3（1.33）→ 1e-2（1.33）；1e-2 未劣化但也未发散。
- 默认 lr=3e-4 已满足作业 ≤1.45；若追求更好 checkpoint 用 `ts_lr3em3`。



### g27 低资源（batch=16, 10k steps, ~10 min/run）


| Run                | LR       | Final valid loss | Wall time | 备注       |
| ------------------ | -------- | ---------------- | --------- | -------- |
| **ts_lr3em3**      | **3e-3** | **1.690**        | ~925s     | 低资源最佳    |
| ts_lr1em3          | 1e-3     | 1.697            | ~517s     |          |
| ts_lr_diverge_5em3 | 5e-3     | 1.713            | ~527s     | 未发散      |
| ts_lr1em2          | 1e-2     | 1.766            | ~927s     | 未发散      |
| ts_lr3em4          | 3e-4     | 1.832            | ~594s     | baseline |
| ts_lr1em4          | 1e-4     | 2.128            | ~594s     |          |
| ts_lr3em5          | 3e-5     | 2.469            | ~594s     |          |
| ts_lr1em5          | 1e-5     | 2.669            | ~594s     |          |




### 低资源 vs 全量对比（相同 LR）


| LR   | 低资源 valid loss | 全量 valid loss | Δ     |
| ---- | -------------- | ------------- | ----- |
| 1e-2 | 1.766          | 1.332         | −0.43 |
| 3e-3 | 1.690          | **1.331**     | −0.36 |
| 1e-3 | 1.697          | 1.371         | −0.33 |
| 3e-4 | 1.832          | 1.469         | −0.36 |
| 1e-4 | 2.128          | 1.653         | −0.48 |
| 3e-5 | 2.469          | 1.899         | −0.57 |
| 1e-5 | 2.669          | 2.065         | −0.60 |


全量训练（8× tokens + 4× batch）对所有 LR 均有大幅提升。低资源最优 LR（3e-3）在全量下仍为最优或接近最优。

---



## 7.2b Edge of Stability


| Run                | 配置     | LR   | Final valid loss | 是否发散 |
| ------------------ | ------ | ---- | ---------------- | ---- |
| ts_lr_diverge_5em3 | 低资源    | 5e-3 | 1.713            | 否    |
| ts_lr_diverge_1em2 | 低资源    | 1e-2 | 1.766            | 否    |
| ts_lr1em2          | **全量** | 1e-2 | 1.332            | 否    |
| ts_lr_diverge_2em2 | **全量** | 2e-2 | —                | 🔄   |
| ts_lr_diverge_5em2 | **全量** | 5e-2 | —                | 🔄   |


- 低资源：最优 3e-3（1.690）优于 5e-3 和 1e-2；1e-2 有过拟合迹象（train 1.64 vs valid 1.77）。
- 全量：1e-2 仍稳定且 val≈3e-3，未观测到 divergent behavior。
- **待补 / 进行中：** 2e-2 / 5e-2 全量 diverge run（g52 GPU 4/5，`ts_lr_diverge_2em2` / `ts_lr_diverge_5em2`）

---



## 7.2 Batch Size Experiment

仅低资源配置（~41M tokens, lr=3e-4）：


| Run     | Batch | Iters   | Final valid loss | Wall time |
| ------- | ----- | ------- | ---------------- | --------- |
| ts_bs16 | 16    | 10,000  | 1.832            | ~517s     |
| ts_bs4  | 4     | 39,500  | 1.925            | ~972s     |
| ts_bs1  | 1     | 159,500 | 2.285            | ~2828s    |


batch 越大越好；全量 bs=64 baseline val=1.469。g68 全量 bs=16/128 进行中，完成后更新下表。

---



## 7.2 Generate Text

**Prompt**：`Once upon a time`  
**Checkpoint（全量）**：`ts_lr3em3/checkpoint_final.pt`（327M tokens，val=**1.331**）  
**运行**：g52 GPU 6，2026-07-24

### Decoding 参数对比（全量模型）


| 配置 | temperature | top_p | fluency | 备注 |
| ---- | ----------- | ----- | ------- | ---- |
| t0_greedy | 0 | — | ★★★★ | 最连贯，情节完整（Lily + Mr. Bear），无语法错误 |
| t05_p095 | 0.5 | 0.95 | ★★★★ | 与 greedy 类似，略有多样性 |
| t08_p095 | 0.8 | 0.95 | ★★★★ | 故事完整（Sue + shell），对话自然 |
| t08_p09 | 0.8 | 0.9 | ★★★☆ | 略保守，"weigh the leaves" 语义略怪 |
| t08_p099 | 0.8 | 0.99 | ★★★★ | 与 t08_p095 接近，情节合理 |
| t10_p095 | 1.0 | 0.95 | ★★★☆ | 出现 "strike the stick with their sticks" 等重复 |
| t12_p095 | 1.2 | 0.95 | ★★☆ | 语义崩坏（"I like your fears"、"replaced flow"） |


**推荐配置**：temperature=0~0.8, top_p=0.9~0.95。全量模型明显优于低资源（逻辑更连贯、重复更少）。

**影响输出质量的两个因素**：

1. **Temperature**：越低越 deterministic，连贯性越好；≥1.2 时小模型易出现 nonsense token 组合。
2. **Top-p (nucleus)**：在 temperature=0.8 下，0.9 vs 0.99 差异不大；主要影响来自 temperature 本身。

### 全量最佳样例（t0 greedy, temperature=0）

```
Once upon a time, there was a little girl named Lily. She had a big, soft teddy bear named Mr. Bear. Lily loved Mr. Bear very much. They played together every day.
One day, Lily and Mr. Bear went to the park. They played on the swings and the slide. Lily was very happy. But then, a big wind came and blew Mr. Bear away. Lily was very sad.
Lily looked for Mr. Bear everywhere. She asked her friends to help her find him. They all looked for Mr. Bear. At last, they found him under a big tree. Lily was so happy to have her bear back. She hugged Mr. Bear and said, "Thank you for helping me find my bear!"
<|endoftext|>
```

完整输出目录：`experiments/section7/generate_full/`  
低资源对照：`experiments/section7/generated_lr3em3.txt`（temperature=0.8, top_p=0.95，有 cake mix 重复）

---



## 7.3 Ablations

仅低资源配置（lr=3e-4, batch=16, 10k steps）：


| Run                  | 改动                   | Final valid loss | vs baseline (1.832) |
| -------------------- | -------------------- | ---------------- | ------------------- |
| ts_ablate_post_norm  | pre-norm → post-norm | 1.824            | −0.008              |
| ts_ablate_no_rmsnorm | 移除 RMSNorm           | 1.840            | +0.008              |
| ts_ablate_silu_ffn   | SwiGLU → SiLU        | 1.888            | +0.056              |
| ts_ablate_no_rope    | 移除 RoPE              | 1.926            | +0.094              |


- **RoPE** 影响最大；**SwiGLU** gating 有收益；**RMSNorm** 移除后仍稳定；**post-norm** 略优于 pre-norm（差异小）。

---



## 7.4 OpenWebText

未开始。

---



## g68 全量 Ablation + Batch Size（2026-07-23 23:57 进度）

固定 327M tokens、lr=3e-4。RoPE baseline 全量 val=**1.469**（`ts_lr3em4`）。

**Ablation（GPU 0–3 并行，~62 min/run）**


| Run                  | 进度             | 状态  |
| -------------------- | -------------- | --- |
| ts_ablate_no_rmsnorm | ~10k/20k (50%) | 🔄  |
| ts_ablate_post_norm  | ~10k/20k (50%) | 🔄  |
| ts_ablate_no_rope    | ~15k/20k (75%) | 🔄  |
| ts_ablate_silu_ffn   | ~15k/20k (75%) | 🔄  |


**Batch Size（GPU 6–7；bs=1/4 已取消）**


| Run      | Batch | Iters  | 进度             | 状态                       |
| -------- | ----- | ------ | -------------- | ------------------------ |
| ts_bs16  | 16    | 80,000 | ~45k/80k (56%) | 🔄 ~50 min 剩余            |
| ts_bs128 | 128   | 10,000 | ~5k/10k (50%)  | 🔄 ~20 min 剩余            |
| ts_bs64  | 64    | 20,000 | —              | ✅ 即 `ts_lr3em4` baseline |


> bs=1/4 全量步数过多（32万~128万 steps）已取消；低资源版 bs=1/4 见 7.2 节。

---



## 机器分工摘要


| 机器  | 任务                            | 状态     |
| --- | ----------------------------- | ------ |
| g27 | 低资源 LR / batch / ablation     | ✅ 完成   |
| g52 | 全量 LR sweep wave1+2           | ✅ 完成   |
| g52 | diverge 2e-2 / 5e-2             | 🔄 GPU 4/5 |
| g68 | 全量 ablation + batch bs=16/128 | 🔄 进行中 |
| g34 | OWT / diverge（建议）             | ❌ 未启动  |


注意：全量 run 与低资源 run **同名**，`experiments/section7/ts_lr`* 的 checkpoint 已被全量结果覆盖（低资源数值保留在本文档和 wandb 历史 run 中）。

---



## 复现命令

```bash
# 低资源
LOW_RESOURCE=1 ./run_section7.sh lr-sweep 0 0 1 2 3

# 全量（需 ~10GB 空闲显存）
./run_section7.sh lr-sweep 0 4 5 6 7

# 文本生成（全量最佳，g52）
CKPT=experiments/section7/ts_lr3em3/checkpoint_final.pt
CUDA_VISIBLE_DEVICES=6 .venv/bin/python generate_text.py \
  --checkpoint $CKPT --temperature 0 --top-p 0.95 \
  --output experiments/section7/generate_full/generated_t0_greedy.txt
# 其他 decoding 组合见 generate_full/ 目录

# OWT（g34 等空闲 GPU）
./run_section7.sh owt 0

# Diverge 补跑（2e-2, 5e-2；start=2 跳过已跑过的 5e-3/1e-2）
./run_section7.sh lr-diverge 2 4 5
```



## 已知问题

- `generate_text.py` 最初用 `torch.optim.AdamW` 加载 checkpoint 会失败（自定义 AdamW state 格式不同），已改为只加载 model weights。
- `run_section7_wave2.sh` 的 diverge 段有重复启动逻辑，不影响 checkpoint 有效性。
- g82/g83/g26 glibc 2.17，无法运行当前 `.venv`。


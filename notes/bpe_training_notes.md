# BPE 训练实验笔记

---

## 1. GPU 空闲情况

`train_bpe_tinystories.py` 是 **CPU 任务**，不占用 GPU。

---

## 2. TinyStories BPE 训练：`train_bpe_tinystories.py`

### 脚本配置

- 输入：`data/TinyStoriesV2-GPT4-train.txt`（2.1 GB）
- `vocab_size=10000`
- `special_tokens=["<|endoftext|>"]`
- 输出：`data/tinystories_vocab.json`、`data/tinystories_merges.txt`

### 实测耗时（优化前）

| 阶段 | 耗时 | 占比 |
|------|------|------|
| 总训练 | ~415–750 s（7–12 分钟） | 100% |
| Pre-tokenization | ~647 s | **~86%** |
| Merge | ~103 s | ~14% |

最长 token：` accomplishment`（15 bytes）

---

## 3. Profile（手册 Problem train_bpe_tinystories b）

使用 `cProfile` 对完整 TinyStories 训练做 profile：

```bash
uv run python -m cProfile -o profiles/bpe_tinystories.prof -s cumtime train_bpe_tinystories.py
uv run profile_bpe.py --full          # 含 flame graph / callgraph
uv run profile_bpe.py --full --snakeviz --port 8080
```

### Profile 结论

**Pre-tokenization（`_count_pretokens`）是最耗时部分**，约占 86%，其中 regex 扫描 + `match.group()` + `str.encode()` 是主要开销。Merge 约占 14%，瓶颈是每轮 `max(pair_counts, ...)` 遍历全部 pair。

### Deliverable 参考（b）

> Pre-tokenization 是最耗时的部分（约 86%），regex 扫描 corpus 并对每个 match 做 `group()` + `encode()` 是主要开销。Merge 约占 14%，瓶颈在于每轮用 `max()` 遍历全部 pair counts 找最高频 pair。

---

## 4. Pre-tokenization 为什么慢？

1. **一次性读入 2.1 GB 文本**（`f.read()`）
2. **两遍 regex 扫描**：special token split + GPT-2 `PAT` 的 `finditer`
3. **每个 match 的 Python 开销**：`group()` + `encode()`，约 5.36 亿次调用
4. **单进程**——未利用多核

手册建议：用 `pretokenization_example.py` 的 `find_chunk_boundaries` + `multiprocessing` 并行 pre-tokenization。

---

## 5. 多进程 Pre-tokenization 优化

### 改动

1. **`pretokenization_example.py`**：注释掉底部无法 import 的 usage 示例
2. **`cs336_basics/bpe.py`**：
   - 抽出 `_pretokenize_text()`、`_pretokenize_chunk()`
   - 文件 > 1 MiB 时，用 `find_chunk_boundaries` 按 `<|endoftext|>` 分块，64 进程并行
   - 小文件（测试集）仍走串行，避免 multiprocessing 开销

### 性能对比（TinyStories 全量）

| 配置 | 耗时 |
|------|------|
| 优化前（单进程） | ~415–750 s |
| 8 进程 | ~121 s |
| **64 进程** | **~69 s（< 2 分钟）** |

测试：`uv run pytest tests/test_train_bpe.py` — 3 passed

---

## 6. OpenWebText BPE 训练（Problem train_bpe_expts_owt）

### 脚本

`train_bpe_owt.py`：

- 输入：`data/owt_train.txt`（12 GB）
- `vocab_size=32000`
- 输出：`data/owt_vocab.json`、`data/owt_merges.txt`

```bash
uv run python train_bpe_owt.py
```

### 结果

| 项目 | 数值 |
|------|------|
| 训练时间 | **8 小时 27 分**（30449 s） |
| 峰值内存 | **9.6 GB** |
| Vocab size | 32,000 |
| Merges | 31,743 |

### Deliverable（a）最长 token

- **64 bytes**：`ÃÂÃÂÃÂ...`（UTF-8 双重编码 mojibake）
- 在 OWT 这种噪声 web 语料上合理

### Deliverable（b）与 TinyStories 对比

| | TinyStories (10K) | OpenWebText (32K) |
|---|---|---|
| 最长 token | 15 bytes，` accomplishment` | 64 bytes，编码乱码片段 |
| 特点 | 简单儿童故事用词 | 更多低频 byte 组合、URL、噪声 |

---

## 7. OWT 训练 8.5 小时正常吗？

**正常。** 多进程只加速 pre-tokenization，OWT 的瓶颈在 merge。

### 时间分解

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Pre-tokenization（64 进程） | **~56 s** | **< 0.2%** |
| Merge（32000 轮，单进程） | **~8.4 h** | **> 99.8%** |

### 为什么 merge 这么慢？

| | TinyStories | OpenWebText |
|---|---|---|
| 数据量 | 2.1 GB | 12 GB |
| Merge 轮数 | 10,000 | 32,000 |
| Unique pretokens | ~几十万 | **660 万** |
| 每轮 merge（粗估） | ~5–7 ms | **~950 ms** |

手册允许 OWT：≤ 12 小时、≤ 100 GB RAM——**8.5 小时在范围内**。

Merge 阶段在 Python 中**不可并行**（手册原文）。要进一步加速，需优化 merge 逻辑（如用 heap 维护最高频 pair，而非每轮 `max()` 扫全表）。

---

## 8. 相关文件

| 文件 | 说明 |
|------|------|
| `cs336_basics/bpe.py` | BPE 训练实现（含多进程 pre-tokenization） |
| `cs336_basics/pretokenization_example.py` | 分块 starter code |
| `train_bpe_tinystories.py` | TinyStories 训练脚本 |
| `train_bpe_owt.py` | OpenWebText 训练脚本 |
| `profile_bpe.py` | Profile + 可视化工具 |
| `profiles/bpe_tinystories.prof` | TinyStories cProfile 数据 |
| `profiles/bpe_tinystories_flame.svg` | 火焰图 |
| `data/tinystories_vocab.json` | TinyStories vocab |
| `data/owt_vocab.json` | OWT vocab |

---

## 9. 常用命令

```bash
# 测试
uv run pytest tests/test_train_bpe.py

# TinyStories 训练
uv run python train_bpe_tinystories.py

# OWT 训练
uv run python train_bpe_owt.py

# Profile
uv run profile_bpe.py --full
uv run profile_bpe.py --full --snakeviz --port 8080
# 远程：ssh -L 8080:localhost:8080 g27
```

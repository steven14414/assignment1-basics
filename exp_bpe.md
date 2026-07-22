# Problem (tokenizer_experiments): Experiments with tokenizers

## 时间估计

| 步骤 | 预计耗时 |
|------|----------|
| (a)(b)(c) 10 篇采样 + throughput | **~30s** |
| (d) TinyStories valid (22MB) | ~30s |
| (d) TinyStories train (2.1GB) | ~45 min |
| (d) OWT valid (277MB) | ~6 min |
| (d) OWT train (12GB) | **~4–5 h** |
| **(d) 合计** | **~5–6 h** |

运行：

```bash
uv run exp_bpe_experiments.py   # (a)(b)(c)
uv run tokenize_datasets.py     # (d)，输出到 data/tokenized/*.npy
```

---

## Deliverables

### (a)

从 TinyStories / OpenWebText valid 各取 10 篇 document（以 `<|endoftext|>` 分隔），同域 compression ratio 约为 **4.01** 与 **4.51 bytes/token**（TinyStories 10K / OpenWebText 32K vocab）。

### (b)

用 TinyStories tokenizer 编码 OpenWebText sample 时，compression ratio 降至约 **3.37 bytes/token**（同域 4.51），因 vocab 更小且训练语料为儿童故事，对 web 文本 subword 覆盖不足，token 数变多、压缩变差；反向（OWT tok → TS）为 **3.87**，大 vocab 在简单文本上仍尚可。

### (c)

TinyStories valid 上 `encode` throughput 约 **706 KB/s**；按此估计 tokenize Pile（825GB）约需 **13.5 天**（单进程 Python 实现）。

### (d)

已用 `encode_iterable` 将 TinyStories / OWT 的 train、valid 写成 `data/tokenized/*.npy`（`dtype=uint16`）。**uint16 足够**：两个 vocab 最大 ID 分别为 10K、32K，均小于 65535，且比 int32/int64 省一半磁盘与内存，便于后续 `np.memmap` 训练。

---

## 实测日志 (a)(b)(c)

```
uv run exp_bpe_experiments.py
=== (a) compression ratio on 10-document samples ===
TinyStories tokenizer on TinyStories: 4.0097 bytes/token
OpenWebText tokenizer on OpenWebText: 4.5050 bytes/token

=== (b) cross-domain compression ratio ===
TinyStories tokenizer on OpenWebText: 3.3692 bytes/token
OpenWebText tokenizer on TinyStories: 3.8672 bytes/token

=== (c) throughput (TinyStories valid, encode) ===
throughput: 706,088 bytes/s
Pile 825GB estimate: 13.5 days (324.6 hours)
```

## 全量 valid 参考（非 10 篇采样）

```
TinyStories valid + TS tok:  4.12 bytes/token, 842 KB/s
OWT valid + OWT tok:         4.37 bytes/token, 797 KB/s
TS tok on OWT valid:         3.15 bytes/token
OWT tok on TS valid:         4.01 bytes/token
```

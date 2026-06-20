# CS336 Assignment 1 (basics)：构建一个 Transformer 语言模型

版本 26.0.3  
CS336 Staff  
2026 年春季

> 本文档是 `cs336_assignment1_basics.pdf` 的中文翻译。代码标识符、函数签名、命令、数学符号、数据集名、论文名和测试名尽量保留原文，便于和作业仓库及英文 PDF 对照。

## 1 作业概览

在本作业中，你将从零开始构建训练一个标准 Transformer language model (LM) 所需的全部组件，并训练一些模型。

你将实现：

1. Byte-pair encoding (BPE) tokenizer（第 2 节）
2. Transformer language model (LM)（第 3 节）
3. Cross-entropy loss function 和 AdamW optimizer（第 4 节）
4. Training loop，并支持序列化与加载 model 和 optimizer 状态（第 5 节）

你将运行：

1. 在 TinyStories 数据集上训练 BPE tokenizer。
2. 用训练好的 tokenizer 对数据集编码，将其转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer LM。
4. 使用训练好的 Transformer LM 进行样本文本生成，并评估 perplexity。
5. 在 OpenWebText 上训练模型，并把你达到的 perplexity 提交到 leaderboard。

你可以使用的内容：

我们希望你从零开始构建每个组件。特别地，除了下面这些内容外，你不能使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的任何定义：

- `torch.nn.Parameter`
- `torch.nn` 中的容器类（例如 `Module`、`ModuleList`、`Sequential` 等）。完整列表见 `pytorch.org/docs/stable/nn.html#containers`。
- `torch.optim.Optimizer` 基类

你可以使用 PyTorch 中的其他任何定义。如果你想使用某个函数或类但不确定是否允许，可以在 Slack 上提问。拿不准时，请思考使用它是否会破坏本作业“from-scratch”的精神。

**关于 AI 工具的声明**

AI 可以完全自主地解决作业中的许多部分。这会让你更难深入参与课程材料，也更难从中学习。

允许使用 AI 工具回答高层概念问题，或提供低层编程文档，例如函数签名和库 API。然而，不允许使用 AI 工具实现任何作业的任何部分。这包括 coding agents（例如 Cursor Agents、Codex、Claude Code）和 AI autocomplete（例如 Cursor Tab、GitHub Copilot）。使用 AI agent 时，请确保它使用提供的 `AGENTS.md` 文件。使用聊天机器人时，也应包含 prompt。

我们强烈建议你在完成作业时关闭 IDE 中的 AI autocomplete（例如 Cursor Tab、GitHub Copilot）；非 AI autocomplete（例如补全函数名）当然没有问题。往届学生反馈说，关闭 AI autocomplete 让他们更容易深入理解材料。

完整 AI Policy 请见课程文档。

**代码长什么样**

作业代码以及这份说明文档都可以在 GitHub 获取：

```text
github.com/stanford-cs336/assignment1-basics
```

请 `git clone` 该仓库。如果有更新，我们会通知你，你可以 `git pull` 获取最新版本。

1. `cs336_basics/*`：这里是你写代码的地方。注意这里没有代码；你可以完全从零开始。
2. `adapters.py`：你的代码必须具备一组功能。对每个功能（例如 scaled dot product attention），通过简单调用你的代码来填充对应实现（例如 `run_scaled_dot_product_attention`）。注意：你对 `adapters.py` 的修改不应包含任何实质性逻辑；它只是 glue code。
3. `test_*.py`：这里包含你必须通过的所有测试（例如 `test_scaled_dot_product_attention`），这些测试会调用 `adapters.py` 中定义的 hooks。不要编辑测试文件。

**如何提交**

提交前，运行 `make_submission.sh` 构建 submission zip 文件。如果你有大型数据文件或 checkpoint 不想包含在提交 zip 中，请确保把它们加入脚本的排除列表。

你将向 Gradescope 提交以下文件：

- `writeup.pdf`：回答所有 written questions。请排版你的回答。
- `code.zip`：包含你写的全部代码。

要提交到 leaderboard，请向下面仓库提交 PR：

```text
github.com/stanford-cs336/assignment1-basics-leaderboard
```

详细提交说明见 leaderboard 仓库中的 `README.md`。

**从哪里获取数据集**

本作业会使用两个预处理好的数据集：TinyStories [R. Eldan et al., 2023] 和 OpenWebText [A. Gokaslan et al., 2019]。二者都是单个大型纯文本文件。

如果你作为课程学生完成作业，可以在 compute guide 中找到下载数据集的说明。如果你是在家自学，可以使用 `README.md` 中的命令下载这些文件。

**Low-Resource Tip: Init**

在整个课程的作业 handout 中，我们会给出一些建议，帮助你在较少 GPU 资源甚至没有 GPU 资源的情况下完成作业的某些部分。例如，我们有时会建议缩小数据集或模型规模，或解释如何在 Mac integrated GPU 或 CPU 上运行训练代码。你会在蓝色框中看到这些 “low-resource tips”。即使你是拥有课程机器访问权限的 Stanford 注册学生，这些建议也可能帮助你更快迭代、节省时间，因此建议阅读。

**Low-Resource Tip: Assignment 1 on Apple Silicon or CPU**

使用 staff solution code，我们可以在配备 36 GB RAM 的 Apple M4 Max 芯片上训练一个 LM，使其生成相当流畅的文本：使用 Metal GPU (MPS) 不到 5 分钟，使用 CPU 约 30 分钟。如果这些词对你意义不大，也不用担心；你只需知道，如果你有一台较新的笔记本，并且实现正确且高效，你就能训练一个小型 LM，使其生成简单儿童故事且流畅度不错。

作业后面会解释如果你使用 CPU 或 MPS，需要做哪些改动。

## 2 Byte-Pair Encoding (BPE) Tokenizer

在作业第一部分，我们将训练并实现一个 byte-level byte-pair encoding (BPE) tokenizer [R. Sennrich et al., 2016; C. Wang et al., 2019]。具体来说，我们会把任意 Unicode 字符串表示为字节序列，并在这个字节序列上训练 BPE tokenizer。之后，我们会用这个 tokenizer 将文本（字符串）编码为语言建模所用的 tokens（整数序列）。

### 2.1 Unicode 标准

Unicode 是一种文本编码标准，它把字符映射到整数 code point。截至 Unicode 17.0（2025 年 9 月发布），该标准在 172 种文字系统中定义了 159,801 个字符。例如，字符 `"s"` 的 code point 是 115（通常记作 `U+0073`，其中 `U+` 是约定前缀，`0073` 是 115 的十六进制表示），字符 `"牛"` 的 code point 是 29275。在 Python 中，可以用 `ord()` 函数把单个 Unicode 字符转换为其整数表示；`chr()` 函数则把整数 Unicode code point 转换为包含对应字符的字符串。

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

**Problem (unicode1): Understanding Unicode (1 point)**

(a) `chr(0)` 返回什么 Unicode 字符？  
Deliverable：一句话回答。

(b) 这个字符的字符串表示（`__repr__()`）和打印表示有什么不同？  
Deliverable：一句话回答。

(c) 当这个字符出现在文本中时会发生什么？你可以在 Python interpreter 中试试下面这些，看看结果是否符合预期：

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

Deliverable：一句话回答。

### 2.2 Unicode Encodings

虽然 Unicode 标准定义了从字符到 code point（整数）的映射，但直接在 Unicode code points 上训练 tokenizer 并不实际，因为 vocabulary 会过大（约 150K 项）且稀疏（许多字符很少出现）。因此，我们会使用 Unicode encoding，它把 Unicode 字符转换为字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网的主导编码（超过 98% 的网页使用它）。

要把 Unicode 字符串编码为 UTF-8，可以使用 Python 的 `encode()` 函数。要访问 Python `bytes` 对象底层的字节值，可以对它迭代（例如调用 `list()`）。最后，可以使用 `decode()` 函数把 UTF-8 byte string 解码回 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # Get the byte values for the encoded string (integers from 0 to 255).
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # One byte does not necessarily correspond to one Unicode character!
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

通过把 Unicode code points 转换为字节序列（例如通过 UTF-8 编码），我们本质上把 code point 序列（21-bit integers，159,801 个有效值）转换为字节值序列（0 到 255 范围内的整数）。长度为 256 的 byte vocabulary 更容易处理。使用 byte-level tokenization 时，我们无需担心 out-of-vocabulary tokens，因为任何输入文本都可以表示为 0 到 255 之间整数构成的序列。

**Problem (unicode2): Unicode Encodings (3 points)**

(a) 相比 UTF-16 或 UTF-32，为什么更偏好在 UTF-8 编码字节上训练 tokenizer？比较不同输入字符串在这些编码下的输出可能会有帮助。  
Deliverable：一到两句话回答。

(b) 考虑下面这个（错误的）函数，它本意是把 UTF-8 byte string 解码为 Unicode string。为什么这个函数不正确？请给出一个输入 byte string，使其产生错误结果。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

Deliverable：给出一个会让 `decode_utf8_bytes_to_str_wrong` 产生错误输出的输入 byte string，并用一句话解释该函数为什么不正确。

(c) 给出一个无法解码为任何 Unicode 字符的 two-byte sequence。  
Deliverable：一个例子，并用一句话解释。

### 2.3 Subword Tokenization

虽然 byte-level tokenization 可以缓解 word-level tokenizer 面临的 out-of-vocabulary 问题，但把文本 tokenization 成字节会产生极长的输入序列。这会减慢模型训练，因为一个包含 10 个词的句子在 word-level language model 中可能只有 10 个 tokens，但在 character-level model 中可能有 50 个或更多 tokens（取决于词长）。处理这些更长序列会让模型每一步需要更多计算。此外，在字节序列上做语言建模也更困难，因为更长的输入序列在数据中形成了更长程依赖。

Subword tokenization 是 word-level tokenizer 和 byte-level tokenizer 之间的折中。注意，byte-level tokenizer 的 vocabulary 有 256 个条目（byte values 为 0 到 255）。Subword tokenizer 用更大的 vocabulary size 换取对输入 byte sequence 更好的压缩。例如，如果 byte sequence `b'the'` 在原始训练文本中经常出现，把它作为 vocabulary 中的一个条目，就能把这个 3-token 序列缩减为一个 token。

我们如何选择要加入 vocabulary 的这些 subword units？R. Sennrich et al. [3] 提出使用 byte-pair encoding（BPE；P. Gage [5]），这是一种压缩算法，它迭代地用一个新的未使用 index 替换（“merge”）最频繁出现的一对 bytes。注意，该算法会向 vocabulary 添加 subword tokens，以最大化输入序列的压缩程度；如果某个词在输入文本中出现足够多次，它将被表示为单个 subword unit。

通过 BPE 构建 vocabulary 的 subword tokenizers 通常称为 BPE tokenizers。本作业中，我们会实现一个 byte-level BPE tokenizer，其中 vocabulary items 是 bytes 或合并后的 byte sequences。这样在处理 out-of-vocabulary 和保持可管理的输入序列长度之间取得了两全。构建 BPE tokenizer vocabulary 的过程称为“训练”BPE tokenizer。

### 2.4 BPE Tokenizer Training

BPE tokenizer training procedure 包含三个主要步骤。

**Vocabulary initialization**

Tokenizer vocabulary 是从 bytestring token 到 integer ID 的一一映射。由于我们训练的是 byte-level BPE tokenizer，初始 vocabulary 就是所有 bytes 的集合。因为可能的 byte values 有 256 个，所以初始 vocabulary size 为 256。

**Pre-tokenization**

一旦有了 vocabulary，原则上你可以统计文本中相邻 bytes 出现的频率，并从最频繁的一对 bytes 开始合并。然而这在计算上很昂贵，因为每次 merge 都需要完整扫描 corpus。此外，直接在整个 corpus 上合并 bytes 可能产生只因标点不同而不同的 tokens（例如 `dog!` 和 `dog.`）。这些 tokens 会获得完全不同的 token IDs，尽管它们很可能语义相似（因为只差标点）。

为了避免这一点，我们对 corpus 进行 pre-tokenization。你可以把它理解为一种粗粒度 tokenization，用来帮助统计字符对出现频率。例如，单词 `'text'` 可能是一个出现 10 次的 pre-token。在统计字符 `'t'` 和 `'e'` 相邻出现频率时，我们看到单词 `'text'` 中 `'t'` 和 `'e'` 相邻，就可以把计数增加 10，而无需逐字扫描整个 corpus。由于我们训练的是 byte-level BPE model，每个 pre-token 表示为 UTF-8 bytes 序列。

R. Sennrich et al. [3] 的原始 BPE 实现仅通过 whitespace split（即 `s.split(" ")`）进行 pre-tokenization。基于 SentencePiece 的 tokenizer 中仍可见这种方法（例如 Llama 1 和 2 tokenizer）。

大多数现代 tokenizers 使用 regex-based pre-tokenizer，这是来自 GPT-2 [A. Radford et al. [6]] 的做法。我们会使用原始 regex 的一个稍微更美观的形式，来自：

```text
github.com/openai/tiktoken/pull/234/files
```

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

交互式地用这个 pre-tokenizer 分割一些文本，可能有助于理解它的行为：

```python
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

不过在代码中使用它时，应使用 `re.finditer`，避免在构造 pre-token 到 count 的映射时把所有 pre-tokenized words 存入内存。

**Compute BPE merges**

现在我们已经把输入文本转换为 pre-tokens，并把每个 pre-token 表示为 UTF-8 bytes 序列，就可以计算 BPE merges（即训练 BPE tokenizer）。高层来看，BPE 算法会迭代地统计每一对 bytes，找出频率最高的一对（“A”, “B”）。然后把这对最频繁 pair（“A”, “B”）的每一次出现都合并，即替换为一个新 token “AB”。这个新 merged token 会加入 vocabulary；因此 BPE training 后的最终 vocabulary size 等于初始 vocabulary size（本作业中为 256）加上训练期间执行的 BPE merge operations 数量。为了 BPE training 的效率，我们不考虑跨越 pre-token boundary 的 pairs。计算 merges 时，如果 pair frequency 出现并列，要确定性地选择 lexicographically greater pair。例如，如果 pairs `("A", "B")`、`("A", "C")`、`("B", "ZZ")` 和 `("BA", "A")` 都有最高频率，我们会 merge `("BA", "A")`：

```python
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

**Special tokens**

一些字符串（例如 `<|endoftext|>`）常用于编码 metadata（例如文档边界）。编码文本时，我们通常希望把某些字符串当作 “special tokens”，它们永远不应被拆成多个 tokens（即总是保留为单个 token）。例如，end-of-sequence string `<|endoftext|>` 应始终保留为单个 token（即单个 integer ID），这样我们才能知道何时停止从 language model 生成文本。这些 special tokens 必须加入 vocabulary，因此它们有对应的固定 token ID。

R. Sennrich et al. [3] 的 Algorithm 1 包含一种低效的 BPE tokenizer training 实现（基本遵循上述步骤）。作为第一个练习，实现并测试这个函数有助于检查你是否理解。

> 注意：原始 BPE 公式指定包含一个 end-of-word token。本作业训练 byte-level BPE models 时不添加 end-of-word token，因为所有 bytes（包括 whitespace 和 punctuation）都包含在模型 vocabulary 中。由于我们显式表示空格和标点，学习到的 BPE merges 会自然反映这些 word boundaries。

**Example (bpe_example): BPE training example**

下面是 R. Sennrich et al. [3] 中的一个风格化例子。考虑由以下文本组成的 corpus：

```text
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

并且 vocabulary 有一个 special token `<|endoftext|>`。

Vocabulary：我们用 special token `<|endoftext|>` 和 256 个 byte values 初始化 vocabulary。

Pre-tokenization：为简单起见并专注于 merge procedure，在这个例子中假设 pre-tokenization 只是按 whitespace split。Pre-tokenize 并计数后得到 frequency table：

```python
{low: 5, lower: 2, widest: 3, newest: 6}
```

把它表示成 `dict[tuple[bytes, ...], int]` 很方便，例如 `{(l,o,w): 5, ...}`。注意，即使单个 byte 在 Python 中也是 `bytes` object。Python 中没有表示单个 byte 的 `byte` 类型，就像 Python 中没有表示单个字符的 `char` 类型一样。

Merges：首先查看每个连续 byte pair，并对它们出现的词频求和，得到 `{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。Pairs `('e', 's')` 和 `('s', 't')` 并列，因此取 lexicographically greater pair，即 `('s', 't')`。然后合并 pre-tokens，得到 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}`。

第二轮中，`(e, st)` 是最常见 pair（count 为 9），会合并为 `{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}`。继续下去，最终得到的 merges 序列是 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。

如果取 6 次 merges，则有 `['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，vocabulary elements 会是 `[<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne]`。在这个 vocabulary 和 merges 集合下，单词 `newest` 会 tokenize 为 `[ne, west]`。

### 2.5 BPE Tokenizer Training 实验

让我们在 TinyStories 数据集上训练 byte-level BPE tokenizer。数据集查找/下载说明见第 1 节。开始前，建议先看看 TinyStories 数据集，了解其中包含什么内容。

**Parallelizing pre-tokenization**

你会发现一个主要瓶颈是 pre-tokenization 步骤。可以用内置库 `multiprocessing` 并行化代码来加速 pre-tokenization。具体来说，我们建议在并行 pre-tokenization 实现中对 corpus 进行分块，同时确保 chunk boundaries 出现在 special token 的开头。你可以直接使用下面链接中的 starter code 获取 chunk boundaries，然后用这些边界把工作分发给不同进程：

```text
https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py
```

这种分块总是有效的，因为我们从不希望跨 document boundaries merge。对本作业而言，你总可以这样分割。不要担心收到一个不含 `<|endoftext|>` 的超大 corpus 这种边界情况。

**Removing special tokens before pre-tokenization**

在用 regex pattern 运行 pre-tokenization（使用 `re.finditer`）前，应从 corpus（或并行实现中的 chunk）中剥离所有 special tokens。确保你按 special tokens 分割，使得不会发生跨越它们所分隔文本的 merge。例如，如果 corpus（或 chunk）类似 `[Doc 1]<|endoftext|>[Doc 2]`，你应按 special token `<|endoftext|>` 分割，并分别 pre-tokenize `[Doc 1]` 和 `[Doc 2]`，从而不会发生跨文档边界的 merge。换句话说，special tokens 在训练期间定义硬分段边界，但它们本身不应贡献 merge counts。可以使用 `re.split`，以 `"|".join(special_tokens)` 作为 delimiter（要小心使用 `re.escape`，因为 special tokens 中可能出现 `|`）。测试 `test_train_bpe_special_tokens` 会测试这一点。

**Optimizing the merging step**

上述风格化例子的朴素 BPE training 实现很慢，因为每次 merge 时都会遍历所有 byte pairs 来找最频繁的 pair。然而，每次 merge 后只有与 merged pair 重叠的 pair counts 会改变。因此，可以通过索引所有 pairs 的 counts 并增量更新这些 counts，而不是显式遍历每一对 bytes 重新统计 pair frequencies，来提高 BPE training speed。这个 caching procedure 可以显著提速，不过我们注意到 BPE training 的 merging 部分在 Python 中不可并行化。

**Low-Resource Tip: Profiling**

你应该使用 `cProfile` 或 `py-spy` 等 profiling tools 找出实现中的瓶颈，并重点优化这些部分。

**Low-Resource Tip: "Downscaling"**

不要一上来就在完整 TinyStories 数据集上训练 tokenizer；建议先在小数据子集（一个 “debug dataset”）上训练。例如，可以先在 TinyStories validation set 上训练 tokenizer，它有 22K documents，而不是 2.12M。这展示了一个通用策略：尽可能 downscale 来加速开发，例如使用更小数据集、更小模型等。选择 debug dataset size 或 hyperparameter config 需要仔细考虑：debug set 应足够大，能呈现和完整配置相同的瓶颈（这样你的优化才会泛化），但又不能大到运行很久。

**Problem (train_bpe): BPE Tokenizer Training (15 points)**

Deliverable：写一个函数，给定输入文本文件路径，训练一个（byte-level）BPE tokenizer。你的 BPE training function 至少应处理以下输入参数：

Input：

- `input_path: str`：包含 BPE tokenizer training data 的文本文件路径。
- `vocab_size: int`：正整数，定义最大最终 vocabulary size（包括初始 byte vocabulary、merge 产生的 vocabulary items，以及任何 special tokens）。
- `special_tokens: list[str]`：要加入 vocabulary 的字符串列表。训练期间，把它们视为硬边界，防止跨越它们的 span 进行 merges，但在计算 merge statistics 时不包含它们。

你的 BPE training function 应返回得到的 vocabulary 和 merges：

Output：

- `vocab: dict[int, bytes]`：tokenizer vocabulary，从 int（vocabulary 中的 token ID）到 bytes（token bytes）的映射。
- `merges: list[tuple[bytes, bytes]]`：训练产生的 BPE merges 列表。每个列表项是 bytes tuple `(<token1>, <token2>)`，表示 `<token1>` 与 `<token2>` 被合并。Merges 应按创建顺序排列。

要用我们提供的测试测试你的 BPE training function，首先需要实现 test adapter `[adapters.run_train_bpe]`。然后运行：

```bash
uv run pytest tests/test_train_bpe.py
```

你的实现应能通过所有测试。可选地（这可能需要大量时间投入），你可以用系统语言实现训练方法的关键部分，例如 C++（考虑 `cppyy` 或 `nanobind`）或 Rust（使用 PyO3）。如果这样做，请注意哪些操作需要复制、哪些可以直接从 Python memory 读取，并留下 build instructions，或者确保它只用 `pyproject.toml` 就能构建。另外注意，GPT-2 regex 在多数 regex engines 中支持不好，并且即使支持通常也太慢。我们验证过 Oniguruma 速度合理并支持 negative lookahead，但 Python 的 `regex` package 甚至可能更快。

**Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)**

(a) 在 TinyStories 数据集上训练 byte-level BPE tokenizer，最大 vocabulary size 为 10,000。确保把 TinyStories 的 `<|endoftext|>` special token 加入 vocabulary。将得到的 vocabulary 和 merges 序列化到磁盘，方便后续检查。训练花费了多少时间和内存？Vocabulary 中最长的 token 是什么？它合理吗？  
Resource requirements：≤ 30 minutes（无 GPU），≤ 30 GB RAM。  
Hint：使用 multiprocessing 做 pre-tokenization，并利用以下两个事实，你应该能把 BPE training 控制在 2 分钟以内：

1. `<|endoftext|>` token 在数据文件中分隔 documents。
2. `<|endoftext|>` token 在应用 BPE merges 前作为 special case 处理。

Deliverable：一到两句话回答。

(b) Profile 你的代码。Tokenizer training process 中哪一部分最耗时？  
Deliverable：一到两句话回答。

接下来，我们会尝试在 OpenWebText 数据集上训练 byte-level BPE tokenizer。和前面一样，建议先看看数据集以更好理解其内容。

**Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)**

(a) 在 OpenWebText 数据集上训练 byte-level BPE tokenizer，最大 vocabulary size 为 32,000。将得到的 vocabulary 和 merges 序列化到磁盘，方便后续检查。Vocabulary 中最长 token 是什么？它合理吗？  
Resource requirements：≤ 12 hours（无 GPU），≤ 100 GB RAM。  
Deliverable：一到两句话回答。

(b) 比较并对比你在 TinyStories 和 OpenWebText 上训练得到的 tokenizer。  
Deliverable：一到两句话回答。

### 2.6 BPE Tokenizer：Encoding 和 Decoding

上一部分中，我们实现了一个函数，用输入文本训练 BPE tokenizer，得到 tokenizer vocabulary 和 BPE merges 列表。现在，我们将实现一个 BPE tokenizer，它加载给定的 vocabulary 和 merges 列表，并使用它们在文本与 token IDs 之间进行 encode/decode。

#### 2.6.1 Encoding text

用 BPE 编码文本的过程与训练 BPE vocabulary 的过程相似。主要有几个步骤。

Step 1: Pre-tokenize。首先像 BPE training 一样，对序列进行 pre-tokenize，并把每个 pre-token 表示为 UTF-8 bytes 序列。我们会在每个 pre-token 内部把这些 bytes 合并为 vocabulary elements，独立处理每个 pre-token（不跨 pre-token boundaries merge）。

Step 2: Apply the merges。然后取 BPE training 期间创建的 vocabulary element merges 序列，并按创建顺序应用到 pre-tokens 上。

**Example (bpe_encoding): BPE encoding example**

例如，假设输入字符串是 `'the cat ate'`，vocabulary 是 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`，学习到的 merges 是 `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`。首先，pre-tokenizer 会把字符串分割为 `['the', ' cat', ' ate']`。然后我们查看每个 pre-token 并应用 BPE merges。

第一个 pre-token `'the'` 初始表示为 `[b't', b'h', b'e']`。查看 merges 列表，找到第一个可应用 merge 为 `(b't', b'h')`，用它把 pre-token 转换为 `[b'th', b'e']`。然后回到 merges 列表，找到下一个可应用 merge 为 `(b'th', b'e')`，把 pre-token 转换为 `[b'the']`。最后再次查看 merges 列表，发现没有更多可应用 merge（因为整个 pre-token 已合并为单个 token），因此完成。对应整数序列为 `[9]`。

对剩余 pre-tokens 重复此过程：pre-token `' cat'` 在应用 BPE merges 后表示为 `[b' c', b'a', b't']`，对应整数序列 `[7, 1, 5]`。最后一个 pre-token `' ate'` 为 `[b' at', b'e']`，对应整数序列 `[10, 3]`。因此，输入字符串最终编码结果为 `[9, 7, 1, 5, 10, 3]`。

**Special tokens**

你的 tokenizer 在编码文本时应能正确处理用户定义的 special tokens（构造 tokenizer 时提供）。

**Memory considerations**

假设我们想 tokenize 一个无法放入内存的大型文本文件。为了高效 tokenize 这个大文件（或任何其他数据流），需要把它拆成可管理的 chunks，并依次处理每个 chunk，使 memory complexity 为常数，而不是随文本大小线性增长。这样做时，必须确保 token 不会跨越 chunk boundaries，否则得到的 tokenization 会不同于把整个序列一次性载入内存并 tokenize 的朴素方法。

#### 2.6.2 Decoding text

要把整数 token IDs 序列 decode 回原始文本，只需在 vocabulary 中查找每个 ID 对应的条目（byte sequence），将它们拼接在一起，然后把 bytes decode 成 Unicode string。注意，输入 IDs 不保证映射到有效 Unicode strings（因为用户可以输入任意整数 ID 序列）。如果输入 token IDs 没有产生有效 Unicode string，应使用官方 Unicode replacement character U+FFFD 替换 malformed bytes。`bytes.decode` 的 `errors` 参数控制如何处理 Unicode decoding errors，使用 `errors='replace'` 会自动用 replacement marker 替换 malformed data。

**Problem (tokenizer): Implementing the tokenizer (15 points)**

Deliverable：实现一个 `Tokenizer` class，给定 vocabulary 和 merges 列表，把文本编码为 integer IDs，并把 integer IDs 解码为文本。你的 tokenizer 还应支持用户提供的 special tokens（如果它们尚不在 vocabulary 中，则追加到 vocabulary）。建议接口如下：

```python
def __init__(self, vocab, merges, special_tokens=None)
```

从给定 vocabulary、merges 列表和（可选）special tokens 列表构造 tokenizer。参数：

- `vocab: dict[int, bytes]`
- `merges: list[tuple[bytes, bytes]]`
- `special_tokens: list[str] | None = None`

```python
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
```

Class method，从序列化的 vocabulary 和 merges 列表（格式与 BPE training code 输出相同）以及（可选）special tokens 列表构造并返回 `Tokenizer`。额外参数：

- `vocab_filepath: str`
- `merges_filepath: str`
- `special_tokens: list[str] | None = None`

```python
def encode(self, text: str) -> list[int]
```

把输入文本编码为 token IDs 序列。

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]
```

给定字符串 iterable（例如 Python file handle），返回一个 generator，惰性地产生 token IDs。这是对无法直接载入内存的大文件进行 memory-efficient tokenization 所必需的。

```python
def decode(self, ids: list[int]) -> str
```

把 token IDs 序列解码为文本。

要用我们提供的测试测试你的 `Tokenizer`，首先需要实现 test adapter `[adapters.get_tokenizer]`。然后运行：

```bash
uv run pytest tests/test_tokenizer.py
```

你的实现应能通过所有测试。

### 2.7 实验

**Problem (tokenizer_experiments): Experiments with tokenizers (4 points)**

(a) 从 TinyStories 和 OpenWebText 各采样 10 篇 documents。使用你之前训练的 TinyStories 和 OpenWebText tokenizers（vocabulary sizes 分别为 10K 和 32K），把这些 sampled documents 编码为 integer IDs。每个 tokenizer 的 compression ratio（bytes/token）是多少？  
Deliverable：一到两句话回答。

(b) 如果用 TinyStories tokenizer tokenize 你的 OpenWebText sample，会发生什么？比较 compression ratio 和/或定性描述结果。  
Deliverable：一到两句话回答。

(c) 估计你的 tokenizer throughput（例如 bytes/second）。Tokenize Pile dataset（825GB 文本）需要多久？  
Deliverable：一到两句话回答。

(d) 使用你的 TinyStories 和 OpenWebText tokenizers，把各自 training 和 development datasets 编码为 integer token IDs 序列。之后我们会用它训练 language model。建议把 token IDs 序列化为 datatype 为 `uint16` 的 NumPy array。为什么 `uint16` 是合适选择？  
Deliverable：一到两句话回答。

## 3 Transformer Language Model Architecture

Language model 的输入是一个 batched integer token IDs 序列（即 shape 为 `(batch_size, sequence_length)` 的 `torch.Tensor`），输出是 vocabulary 上的 batched normalized probability distribution（即 shape 为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch Tensor），其中每个输入 token 对应的预测分布是下一个词的分布。训练 language model 时，我们用这些 next-word predictions 计算真实下一个词与预测下一个词之间的 cross-entropy loss。推理时从 language model 生成文本，我们取最终时间步（即序列最后一个 item）的 predicted next-word distribution 来生成序列中的下一个 token（例如取概率最高的 token、从分布中采样等），将生成的 token 加到输入序列中，然后重复。

在本作业这一部分，你将从零开始构建这个 Transformer language model。我们先高层描述模型，然后逐步细化各个组件。

### 3.1 Transformer LM

给定 token IDs 序列，Transformer language model 使用 input embedding 把 token IDs 转换为 dense vectors，将 embedded tokens 通过 `num_layers` 个 Transformer blocks，然后应用一个 learned linear projection（“output embedding” 或 “LM head”）来产生 predicted next-token logits。示意图见原 PDF Figure 1。

**Token Embeddings**

在第一步，Transformer 把（batched）token IDs 序列嵌入为包含 token identity 信息的 vector 序列（Figure 1 中红色块）。

更具体地说，给定 token IDs 序列，Transformer language model 使用 token embedding layer 产生 vector 序列。每个 embedding layer 输入 shape 为 `(batch_size, sequence_length)` 的整数 tensor，并输出 shape 为 `(batch_size, sequence_length, d_model)` 的 vector 序列。

**Pre-norm Transformer Block**

Embedding 之后，activations 会被若干结构相同的 neural net layers 处理。标准 decoder-only Transformer language model 包含 `num_layers` 个相同层（通常称为 Transformer “blocks”）。每个 Transformer block 输入 shape 为 `(batch_size, sequence_length, d_model)`，输出 shape 也为 `(batch_size, sequence_length, d_model)`。每个 block 通过 self-attention 聚合序列中的信息，并通过 feed-forward layers 对其进行非线性变换。

经过 `num_layers` 个 Transformer blocks 后，我们会取最终 activations，并把它们转换为 vocabulary 上的分布。

我们将实现 “pre-norm” Transformer block（第 3.4 节详述）。它还要求在最后一个 Transformer block 之后使用 layer normalization（下文详述），确保输出被适当缩放。经过此 normalization 后，我们会使用标准 learned linear transformation 把 Transformer blocks 的输出转换为 predicted next-token logits（例如见 A. Radford et al. [7] equation 2）。

### 3.2 备注：Batching、Einsum 和高效计算

在整个 Transformer 中，我们会对许多 batch-like inputs 应用相同计算。例如：

- Batch 中的元素：对每个 batch element 应用相同 Transformer forward operation。
- Sequence length：RMSNorm 和 feed-forward 等 “position-wise” operations 对序列中每个 position 以相同方式操作。
- Attention heads：在 “multi-headed” attention operation 中，attention operation 会跨 attention heads batch。

拥有一种符合人体工学的方式来执行这些操作很有用：它既能充分利用 GPU，又易读易懂。许多 PyTorch operations 可以在 tensor 开头接受多余的 “batch-like” dimensions，并高效地在这些维度上重复/广播操作。

例如，假设我们在做一个 position-wise、batched operation。数据 tensor `D` 的 shape 为 `(batch_size, sequence_length, d_model)`，希望与 shape 为 `(d_model, d_model)` 的矩阵 `A` 做 batched vector-matrix multiply。在这种情况下，`D @ A` 会进行 batched matrix multiply，这是 PyTorch 中的高效 primitive，其中 `(batch_size, sequence_length)` 维度被 batched over。

因此，假设你的函数可能收到额外 batch-like dimensions，并把这些 dimensions 放在 PyTorch shape 的开头，会很有帮助。为了组织 tensors 以这种方式 batch，它们可能需要通过许多 `view`、`reshape` 和 `transpose` 步骤来变形。这会有些麻烦，而且代码在做什么、tensor shapes 是什么常常变得难读。

更符合人体工学的选择是在 `torch.einsum` 中使用 einsum notation，或者使用 framework-agnostic libraries，例如 `einops` 或 `einx`。两个关键 ops 是 `einsum` 和 `rearrange`：`einsum` 可以对任意维度输入 tensor 做 tensor contractions，`rearrange` 可以重排、连接和拆分任意维度。事实证明，机器学习中几乎所有操作都是维度调整和 tensor contraction 的某种组合，偶尔加上（通常是 pointwise 的）nonlinear function。这意味着使用 einsum notation 可以让大量代码更可读、更灵活。

我们强烈建议在本课程中学习并使用 einsum notation。以前没接触过 einsum notation 的学生应使用 `einops`（见文档）；已经熟悉 `einops` 的学生应学习更通用的 `einx`。这两个包已经安装在我们提供的环境中。

下面给出一些 einsum notation 用法例子。这些是 `einops` 文档的补充，你应先阅读该文档。

**Example (einstein_example1): Batched matrix multiplication with `einops.einsum`**

```python
import torch
from einops import rearrange, einsum

## Basic implementation
Y = D @ A.T
# Hard to tell the input and output shapes and what they mean.
# What shapes can D and A have, and do any of these have unexpected behavior?

## Einsum is self-documenting and robust
#                          D                A     ->          Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

## Or, a batched version where D can have any leading dimensions but A is constrained.
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

**Example (einstein_example2): Broadcasted operations with `einops.rearrange`**

我们有一批 images，希望对每张 image 根据某个 scaling factor 生成 10 个变暗版本：

```python
images = torch.randn(64, 128, 128, 3) # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

## Reshape and multiply
dim_value = rearrange(dim_by,    "dim_value              -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

## Or in one go:
dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

**Example (einstein_example3): Pixel mixing with `einops.rearrange`**

假设有一批 images，表示为 shape `(batch, height, width, channel)` 的 tensor。我们希望对 image 的所有 pixels 执行 linear transformation，但这个 transformation 应对每个 channel 独立发生。我们的 linear transformation 表示为 shape `(height * width, height * width)` 的矩阵 `B`。

```python
channels_last = torch.randn(64, 32, 32, 3)       # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

## Rearrange an image tensor for mixing across all pixels
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)

## Instead, using einops:
height = width = 32
channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height, width=width
)

## Or, if you’re feeling crazy: all in one go using einx.dot
height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width
)
```

这里第一个实现可以通过在前后加注释来说明 input/output shapes，但这很笨重，也容易出 bug。使用 einsum notation 时，documentation 就是 implementation。

Einsum notation 可以处理任意 input batching dimensions，同时还有自文档化的关键优点。在使用 einsum notation 的代码中，输入和输出 tensors 的相关 shapes 更清楚。对剩余 tensors，可以考虑使用 Tensor type hints，例如 `jaxtyping` library（不限于 JAX）。

我们会在 assignment 2 中更多讨论使用 einsum notation 的 performance implications；目前只需知道它们几乎总是比替代写法更好。

#### 3.2.1 Mathematical Notation and Memory Ordering

许多机器学习论文使用 row vectors 作为 notation，这与 NumPy 和 PyTorch 默认使用的 row-major memory ordering 很契合。使用 row vectors 时，linear transformation 写作：

```text
y = x W^T
```

其中 row-major `W ∈ R^{d_out × d_in}`，row-vector `x ∈ R^{1 × d_in}`。注意这允许我们通过增加 `x` 的最外层维度来 batch inputs，即把 vector input `x` 替换为 matrix input `X ∈ R^{batch × d_in}`。

在线性代数中，更常见的是 column vectors，其中 linear transformations 写作：

```text
y = W x
```

给定 row-major `W ∈ R^{d_out × d_in}` 和 column-vector `x ∈ R^{d_in}`。在这种设定下 batch input 时，batch dimension 必须放在 `x` 的最后，因此 `x` 需要替换为 matrix `X~ ∈ R^{d_in × batch}`。

本作业中数学 notation 大多使用 column vectors，因为数学通常遵循这种 notation。你应记住，如果想使用普通 matrix multiplication notation，因为 PyTorch 使用 row-major memory ordering，你需要像 Equation 1 中 row vector convention 那样以 transpose 方式应用矩阵。如果使用 einsum 做 linear algebra operations，只要正确标记 axes，这不应成为问题。顺带一提，Matlab、Julia 和 Fortran 等语言/linear algebra packages 使用 column-major memory ordering，意味着 batching dimensions 放在最后；但 Python 及相关 packages 采用了 C 标准的 row-major ordering。

### 3.3 Basic Building Blocks：Linear 和 Embedding Modules

#### 3.3.1 Parameter Initialization

有效训练 neural networks 往往需要仔细初始化 model parameters；糟糕初始化可能导致 vanishing 或 exploding gradients 等不良行为。Pre-norm transformers 对初始化异常鲁棒，但初始化仍会显著影响 training speed 和 convergence。由于本作业已经很长，细节留到 assignment 3；这里先给出一些近似 initialization，大多数情况下应能很好工作。

目前使用：

- Linear weights：`N(μ = 0, σ² = 2 / (d_in + d_out))`，截断到 `[-3σ, 3σ]`。
- Embedding：`N(μ = 0, σ² = 1)`，截断到 `[-3, 3]`。
- RMSNorm：`1`。

应使用 `torch.nn.init.trunc_normal_` 初始化 truncated normal weights。

#### 3.3.2 Linear Module

Linear layers 是 Transformers 和一般 neural nets 的基础构件。首先，你将实现自己的 `Linear` class，它继承 `torch.nn.Module` 并执行 linear transformation：

```text
y = W x
```

注意，遵循多数现代 LLMs，我们不包含 bias term。

**Problem (linear): Implementing the linear module (1 point)**

Deliverable：实现一个继承 `torch.nn.Module` 的 `Linear` class，用于执行 linear transformation。你的实现应遵循 PyTorch 内置 `nn.Linear` module 的接口，但不包含 bias argument 或 parameter。建议接口：

```python
def __init__(self, in_features, out_features, device=None, dtype=None)
```

构造 linear transformation module。参数：

- `in_features: int`：输入的 final dimension
- `out_features: int`：输出的 final dimension
- `device: torch.device | None = None`：parameters 存储设备
- `dtype: torch.dtype | None = None`：parameters 数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

对输入应用 linear transformation。

确保：

- subclass `nn.Module`
- 调用 superclass constructor
- 构造并存储你的 parameter 为 `W`（不是 `W^T`），并放入 `nn.Parameter`
- 当然，不要使用 `nn.Linear` 或 `nn.functional.linear`

初始化时使用上面的设置，并用 `torch.nn.init.trunc_normal_` 初始化 weights。

要测试你的 `Linear` module，实现 test adapter `[adapters.run_linear]`。Adapter 应把给定 weights 加载进你的 `Linear` module。可以为此使用 `Module.load_state_dict`。然后运行：

```bash
uv run pytest -k test_linear
```

#### 3.3.3 Embedding Module

如上所述，Transformer 的第一层是 embedding layer，它把 integer token IDs 映射到维度为 `d_model` 的 vector space。我们会实现一个继承 `torch.nn.Module` 的自定义 `Embedding` class（所以不应使用 `nn.Embedding`）。`forward` method 应使用 shape 为 `(batch_size, sequence_length)` 的 `torch.LongTensor` token IDs，对 shape 为 `(vocab_size, d_model)` 的 embedding matrix 做 indexing，选出每个 token ID 对应的 embedding vector。

**Problem (embedding): Implement the embedding module (1 point)**

Deliverable：实现继承 `torch.nn.Module` 的 `Embedding` class，执行 embedding lookup。你的实现应遵循 PyTorch 内置 `nn.Embedding` module 的接口。建议接口：

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
```

构造 embedding module。参数：

- `num_embeddings: int`：Vocabulary size
- `embedding_dim: int`：Embedding vectors 的维度，即 `d_model`
- `device: torch.device | None = None`：parameters 存储设备
- `dtype: torch.dtype | None = None`：parameters 数据类型

```python
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```

查找给定 token IDs 的 embedding vectors。

确保：

- subclass `nn.Module`
- 调用 superclass constructor
- 把 embedding matrix 初始化为 `nn.Parameter`
- 存储 embedding matrix 时让 `d_model` 是 final dimension
- 当然，不要使用 `nn.Embedding` 或 `nn.functional.embedding`

同样，初始化使用上面的设置，并用 `torch.nn.init.trunc_normal_` 初始化 weights。

要测试实现，完成 test adapter `[adapters.run_embedding]`。然后运行：

```bash
uv run pytest -k test_embedding
```

### 3.4 Pre-Norm Transformer Block

每个 Transformer block 有两个 sub-layers：multi-head self-attention mechanism 和 position-wise feed-forward network（A. Vaswani et al. [2017], section 3.1）。

在原始 Transformer 论文中，模型在两个 sub-layers 外分别使用 residual connection，然后进行 layer normalization。这种架构通常称为 “post-norm” Transformer，因为 layer normalization 应用于 sub-layer output。然而，许多工作发现，把 layer normalization 从每个 sub-layer 的 output 移到每个 sub-layer 的 input（并在最后一个 Transformer block 后额外进行 layer normalization）可以提高 Transformer training stability [T. Q. Nguyen et al., 2019; R. Xiong et al., 2020]。这个 “pre-norm” Transformer block 的视觉表示见 Figure 2。每个 Transformer block sub-layer 的输出随后通过 residual connection 加到 sub-layer input 上（A. Vaswani et al. [8], section 5.4）。对 pre-norm 的一种直觉是，从 input embeddings 到 Transformer 最终输出之间存在一条干净的、不经过任何 normalization 的 “residual stream”，据称这能改善 gradient flow。Pre-norm Transformer 现在已成为当今 language models 的标准（例如 GPT-3、LLaMA、PaLM 等），因此我们会实现这个变体。我们将逐个讲解 pre-norm Transformer block 的组件并依次实现。

#### 3.4.1 Root Mean Square Layer Normalization

A. Vaswani et al. [8] 的原始 Transformer 实现使用 layer normalization [J. L. Ba et al., 2016] 来 normalize activations。遵循 H. Touvron et al. [12]，我们会使用 root mean square layer normalization（RMSNorm；B. Zhang et al. [13], equation 4）进行 layer normalization。给定 activations vector `a ∈ R^{d_model}`，RMSNorm 会按如下方式重新缩放每个 activation `a_i`：

```text
RMSNorm(a_i) = a_i / RMS(a) * g_i
RMS(a) = sqrt((1 / d_model) * sum_i a_i^2 + ε)
```

其中 `g_i` 是可学习的 “gain” parameter（总共有 `d_model` 个这样的参数），`ε` 是一个 hyperparameter，通常固定为 `1e-5`。

在对输入平方之前，应把输入 upcast 到 `torch.float32` 以防 overflow。整体上，`forward` method 应类似：

```python
in_dtype = x.dtype
x = x.to(torch.float32)

# Your code here performing RMSNorm
...
result = ...

# Return the result in the original dtype
return result.to(in_dtype)
```

**Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)**

Deliverable：把 RMSNorm 实现为 `torch.nn.Module`。建议接口：

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
```

构造 RMSNorm module。参数：

- `d_model: int`：模型 hidden dimension
- `eps: float = 1e-5`：数值稳定用 epsilon
- `device: torch.device | None = None`：parameters 存储设备
- `dtype: torch.dtype | None = None`：parameters 数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

处理 shape 为 `(batch_size, sequence_length, d_model)` 的输入 tensor，并返回相同 shape 的 tensor。

注意：如上所述，执行 normalization 前记得把输入 upcast 到 `torch.float32`（之后再 downcast 回原 dtype）。

要测试实现，完成 test adapter `[adapters.run_rmsnorm]`。然后运行：

```bash
uv run pytest -k test_rmsnorm
```

#### 3.4.2 Position-Wise Feed-Forward Network

原始 Transformer 论文（A. Vaswani et al. [8] section 3.3）中的 feed-forward network 由两次 linear transformations 构成，中间是 ReLU activation（`ReLU(x) = max(0, x)`）。在原始架构中，内部 feed-forward layer 的维度通常是输入维度的 4 倍。

然而，与这个原始设计相比，现代 language models 通常有两个主要变化：使用另一种 activation function，并采用 gating mechanism。具体来说，我们会实现 Llama 3 [A. Grattafiori et al., 2024] 和 Qwen 2.5 [A. Yang et al., 2024] 等 LLMs 采用的 “SwiGLU” activation function。它把 SiLU（常称为 Swish）activation 与称为 Gated Linear Unit (GLU) 的 gating mechanism 结合起来。遵循 PaLM [A. Chowdhery et al., 2022] 和 LLaMA [H. Touvron et al., 2023] 以来多数现代 LLMs，我们也会省略 linear layers 中有时使用的 bias terms。

SiLU 或 Swish activation function [D. Hendrycks et al., 2016; S. Elfwing et al., 2017] 定义为：

```text
SiLU(x) = x * σ(x) = x / (1 + e^{-x})
```

如 Figure 3 所示，SiLU activation function 与 ReLU activation function 相似，但在零点处是 smooth 的。

Gated Linear Units (GLUs) 最早由 Y. N. Dauphin et al. [19] 定义为：一个 linear transformation 经过 sigmoid function 后与另一个 linear transformation 做 element-wise product：

```text
GLU(x, W1, W2) = σ(W1 x) ⊙ W2 x
```

其中 `⊙` 表示 element-wise multiplication。Gated Linear Units 被认为可以“通过为 gradients 提供 linear path 同时保留 nonlinear capabilities，减少 deep architectures 中的 vanishing gradient problem”。

把 SiLU/Swish 和 GLU 合在一起，就得到我们将在 feed-forward networks 中使用的 SwiGLU：

```text
FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1 x) ⊙ W3 x)
```

其中 `x ∈ R^{d_model}`，`W1, W3 ∈ R^{d_ff × d_model}`，`W2 ∈ R^{d_model × d_ff}`，标准上 `d_ff = 8/3 d_model`。在具体实现中，为硬件效率，把它 round 到附近的 64 的倍数是可以的。

N. Shazeer [20] 首先提出把 SiLU/Swish activation 与 GLUs 结合，并通过实验表明 SwiGLU 在 language modeling tasks 上优于 ReLU 和 SiLU（无 gating）等 baselines。作业后面你会比较 SwiGLU 和 SiLU。虽然我们提到了一些关于这些组件的启发式论证（论文中也提供了更多支持证据），但保持经验主义视角是好的；Shazeer 论文中有一句如今很有名的话：

> “We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.”

**Problem (positionwise_feedforward): Implement the position-wise feed-forward network (2 points)**

Deliverable：实现 SwiGLU feed-forward network，由 SiLU activation function 和 GLU 组成。

注意：在这个特定情况下，为数值稳定，你可以在实现中使用 `torch.sigmoid`。

你应在实现中把 `d_ff` 设为约 `8/3 × d_model`，同时确保内部 feed-forward layer 的维度是 64 的倍数，以便更好利用硬件。要用我们提供的测试测试实现，需要完成 test adapter `[adapters.run_swiglu]`。然后运行：

```bash
uv run pytest -k test_swiglu
```

#### 3.4.3 Relative Positional Embeddings

为了向模型注入 positional information，我们将实现 Rotary Position Embeddings [J. Su et al., 2021]，通常称为 RoPE。对于 token position `i` 处的 query token `q^(i) = W_q x^(i) ∈ R^d`，我们会应用 pairwise rotation matrix `R_i`，得到 `q'^(i) = R_i q^(i) = R_i W_q x^(i)`。这里，`R_i` 会把 embedding elements 的成对元素 `q_{2k-1:2k}^{(i)}` 当作 2D vectors，按角度 `θ_{i,k} = i / Θ^{(2k-2)/d}` 旋转，其中 `k ∈ {1, ..., d/2}`，`Θ` 是某个常数。因此，可把 `R_i` 视为大小 `d × d` 的 block-diagonal matrix，其 blocks 为 `R_k^i`，其中：

```text
R_k^i = [[cos(θ_{i,k}), -sin(θ_{i,k})],
         [sin(θ_{i,k}),  cos(θ_{i,k})]]
```

由此得到完整 rotation matrix `R^i`，它是这些 `2 × 2` rotation blocks 组成的 block diagonal matrix，其他位置为 `2 × 2` zero matrices。

虽然可以构造完整的 `d × d` 矩阵，一个好的解法应利用该矩阵的性质更高效地实现 transformation。由于我们只关心给定序列中 tokens 的相对旋转，可以在 layers 和不同 batches 之间复用为 `cos(θ_{i,k})` 和 `sin(θ_{i,k})` 计算的值。如果想优化，可以使用被所有 layers 引用的单个 RoPE module，并在 init 中用 `self.register_buffer(persistent=False)` 创建一个预计算的 sin/cos 2D buffer，而不是 `nn.Parameter`（因为我们不想学习这些固定的 cosine 和 sine 值）。随后，对 `k^(j)` 执行和对 `q^(i)` 完全相同的 rotation process，按对应的 `R_j` 旋转。注意这一层没有 learnable parameters。

**Problem (rope): Implement RoPE (2 points)**

Deliverable：实现一个 `RotaryPositionalEmbedding` class，把 RoPE 应用于输入 tensor。

建议接口：

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```

构造 RoPE module，并在需要时创建 buffers。

- `theta: float`：RoPE 的 `Θ` 值
- `d_k: int`：query 和 key vectors 的维度
- `max_seq_len: int`：将输入的最大 sequence length
- `device: torch.device | None = None`：buffer 存储设备

```python
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

处理 shape 为 `(..., seq_len, d_k)` 的输入 tensor，并返回相同 shape 的 tensor。注意，应允许 `x` 有任意数量的 batch dimensions。你应假设 token positions 是 shape 为 `(..., seq_len)` 的 tensor，指定 `x` 沿 sequence dimension 的 token positions。

你应使用 token positions 沿 sequence dimension slice 你的（可能预计算的）cos 和 sin tensors。

要测试实现，完成 `[adapters.run_rope]` 并确保通过：

```bash
uv run pytest -k test_rope
```

#### 3.4.4 Scaled Dot-Product Attention

现在我们将实现 A. Vaswani et al. [8]（section 3.2.1）所描述的 scaled dot-product attention。作为预备，Attention operation 的定义会用到 softmax。Softmax 是一种把未归一化 score vector 转换为 normalized distribution 的操作：

```text
softmax(v)_i = exp(v_i) / sum_j exp(v_j)
```

注意对于很大的值，`exp(v_i)` 可能变成 `inf`（然后 `inf / inf = NaN`）。我们可以利用 softmax 对给所有 inputs 加上任意常数 `c` 不变这一性质来避免该问题。为了数值稳定，通常从 `v` 的所有元素中减去 `v` 的最大 entry，使新的最大 entry 为 0。现在你将使用这个技巧实现 softmax。

**Problem (softmax): Implement softmax (1 point)**

Deliverable：写一个函数，对 tensor 应用 softmax operation。函数应接受两个参数：一个 tensor 和一个 dimension `i`，并把 softmax 应用于输入 tensor 的第 `i` 个 dimension。输出 tensor 应具有与输入 tensor 相同的 shape，但其第 `i` 个 dimension 会变成 normalized probability distribution。使用从第 `i` 个 dimension 中所有元素减去该维度最大值的技巧，避免 numerical stability issues。

要测试实现，完成 `[adapters.run_softmax]` 并确保通过：

```bash
uv run pytest -k test_softmax_matches_pytorch
```

现在可以如下数学定义 Attention operation：

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

其中 `Q ∈ R^{n × d_k}`，`K ∈ R^{m × d_k}`，`V ∈ R^{m × d_v}`。这里 `Q`、`K` 和 `V` 都是该 operation 的输入；注意它们不是 learnable parameters。

Masking：有时需要 mask attention operation 的输出。Mask 应有 shape `M ∈ {True, False}^{n × m}`，这个 boolean matrix 的每一行 `i` 表示 query `i` 应 attend to 哪些 keys。按惯例（也略令人困惑），位置 `(i, j)` 为 `True` 表示 query `i` attend to key `j`，为 `False` 表示 query 不 attend to 该 key。换句话说，`True` 的 `(i, j)` pairs 处“information flows”。例如，考虑 entries 为 `[[True, True, False]]` 的 `1 × 3` mask matrix。这个单个 query vector 只 attend to 前两个 keys。

计算上，使用 masking 会比在 subsequences 上计算 attention 高效得多。我们可以在 pre-softmax values（`QK^T / sqrt(d_k)`）中，对 mask matrix 中为 `False` 的任何 entry 加上 `-∞`。

**Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5 points)**

Deliverable：实现 scaled dot-product attention function。实现应处理 shape 为 `(batch_size, ..., seq_len, d_k)` 的 keys 和 queries，以及 shape 为 `(batch_size, ..., seq_len, d_v)` 的 values，其中 `...` 表示任意数量的其他 batch-like dimensions（如果提供）。实现应返回 shape 为 `(batch_size, ..., seq_len, d_v)` 的输出。关于 batch-like dimensions 的讨论见第 3.2 节。

你的实现还应支持可选的用户提供 boolean mask，shape 为 `(seq_len, seq_len)`。Mask value 为 `True` 的 positions 的 attention probabilities 应总和为 1，mask value 为 `False` 的 positions 的 attention probabilities 应为 0。

要用我们提供的测试测试实现，需要实现 test adapter `[adapters.run_scaled_dot_product_attention]`。`uv run pytest -k test_scaled_dot_product_attention` 会在 third-order input tensors 上测试实现；`uv run pytest -k test_4d_scaled_dot_product_attention` 会在 fourth-order input tensors 上测试实现。

#### 3.4.5 Causal Multi-Head Self-Attention

我们将实现 A. Vaswani et al. [8] section 3.2.2 中描述的 multi-head self-attention。回忆一下，应用 multi-head attention 的数学 operation 定义为：

```text
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)
head_i = Attention(Q_i, K_i, V_i)
```

其中 `Q_i`、`K_i`、`V_i` 分别是 `Q`、`K`、`V` 的 embedding dimension 上大小为 `d_k` 或 `d_v` 的第 `i` 个 slice。Attention 是第 3.4.4 节中定义的 scaled dot-product attention operation。

由此形成 multi-head self-attention operation：

```text
MultiHeadSelfAttention(x) = W_O MultiHead(W_Q x, W_K x, W_V x)
```

这里 learnable parameters 为 `W_Q ∈ R^{h d_k × d_model}`，`W_K ∈ R^{h d_k × d_model}`，`W_V ∈ R^{h d_v × d_model}`，以及 `W_O ∈ R^{d_model × h d_v}`。由于 `Q`、`K`、`V` 在 multi-head attention operation 中会被切片，可以把 `W_Q`、`W_K`、`W_V` 理解为沿 output dimension 为每个 head 分开。当你完成实现后，应通过总共三次 matrix multiplies 计算 key、value 和 query projections。作为 stretch goal，可以尝试把 key、query 和 value projections 合并到单个 weight matrix 中，这样只需要一次 matrix multiply。

**Causal masking**

你的实现应防止模型 attend to 序列中的 future tokens。换句话说，如果模型给定 token sequence `t_1, ..., t_n`，而我们要为 prefix `t_1, ..., t_i`（其中 `i < n`）计算 next-word predictions，模型不应能够访问（attend to）位置 `t_{i+1}, ..., t_n` 的 token representations，因为在 inference 生成文本时模型无法访问这些 tokens（这些 future tokens 会泄漏真实下一个词的 identity，使 language modeling pre-training objective 变得 trivial）。对输入 token sequence `t_1, ..., t_n`，一种朴素做法是为 `n` 个 unique prefixes 分别运行 multi-head self-attention `n` 次，以阻止访问 future tokens。相反，我们会使用 causal attention masking，它允许 token `i` attend to 序列中所有 positions `j ≤ i`。可以使用 `torch.triu` 或 broadcasted index comparison 构造此 mask，并利用第 3.4.4 节中 scaled dot-product attention 实现已经支持 attention masking 这一事实。

**Applying RoPE**

RoPE 应应用于 query 和 key vectors，但不应用于 value vectors。此外，head dimension 应作为 batch dimension 处理，因为在 multi-head attention 中，attention 对每个 head 独立应用。这意味着对每个 head，应该向 query 和 key vectors 应用完全相同的 RoPE rotation。

**Problem (multihead_self_attention): Implement causal multi-head self-attention (5 points)**

Deliverable：把 causal multi-head self-attention 实现为 `torch.nn.Module`。实现至少应接受以下参数：

- `d_model: int`：Transformer block inputs 的 dimensionality。
- `num_heads: int`：multi-head self-attention 中使用的 heads 数量。

遵循 A. Vaswani et al. [8]，设置 `d_k = d_v = d_model / h`。要用提供的测试测试实现，完成 test adapter `[adapters.run_multihead_self_attention]`。然后运行：

```bash
uv run pytest -k test_multihead_self_attention
```

### 3.5 The Full Transformer LM

先组装 Transformer block（参考 Figure 2 会有帮助）。一个 Transformer block 包含两个 “sub-layers”：一个用于 multihead self attention，另一个用于 SwiGLU feed-forward network。在每个 sub-layer 中，先执行 RMSNorm，再执行主 operation（MHA/FF），最后加上 residual connection。

具体来说，Transformer block 的前半部分（第一个 “sub-layer”）应从输入 `x` 产生输出 `y`，实现如下更新：

```text
y = x + MultiHeadSelfAttention(RMSNorm(x))
```

**Problem (transformer_block): Implement the Transformer block (3 points)**

实现第 3.4 节描述并在 Figure 2 中展示的 pre-norm Transformer block。你的 Transformer block 至少应接受以下参数：

- `d_model: int`：Transformer block inputs 的 dimensionality。
- `num_heads: int`：multi-head self-attention 中使用的 heads 数量。
- `d_ff: int`：position-wise feed-forward inner layer 的 dimensionality。

要测试实现，完成 adapter `[adapters.run_transformer_block]`。然后运行：

```bash
uv run pytest -k test_transformer_block
```

Deliverable：通过提供测试的 Transformer block code。

现在按照 Figure 1 的高层图把 blocks 组合起来。遵循第 3.1.0.1 节对 embedding 的描述，把它输入 `num_layers` 个 Transformer blocks，然后传入 final layer norm 和 LM head，得到 vocabulary 上的 unnormalized distribution（logits）。

**Problem (transformer_lm): Implementing the Transformer LM (3 points)**

是时候把所有东西组合起来了。实现第 3.1 节描述并在 Figure 1 中展示的 Transformer language model。实现至少应接受前面 Transformer block 的所有 construction parameters，以及这些额外参数：

- `vocab_size: int`：Vocabulary size，用来确定 token embedding matrix 的 dimensionality。
- `context_length: int`：Maximum context length，用来确定 RoPE sin/cos buffer 的 dimensionality。
- `num_layers: int`：使用的 Transformer blocks 数量。

要用提供的测试测试实现，首先需要实现 test adapter `[adapters.run_transformer_lm]`。然后运行：

```bash
uv run pytest -k test_transformer_lm
```

Deliverable：通过上述测试的 Transformer LM module。

**Resource accounting**

理解 Transformer 各部分如何消耗 compute 和 memory 很有用。我们将完成一些基本的 “FLOPs accounting”。Transformer 中绝大多数 FLOPs 来自 matrix multiplies，因此核心方法很简单：

1. 写出 Transformer forward pass 中所有 matrix multiplies。
2. 把每个 matrix multiply 转换为所需 FLOPs。

第二步会用到以下事实：

Rule：给定 `A ∈ R^{m × n}` 和 `B ∈ R^{n × p}`，matrix-matrix product `AB` 需要 `2mnp` FLOPs。

原因是 `(AB)[i, j] = A[i, :] ⋅ B[:, j]`，这个 dot product 需要 `n` 次加法和 `n` 次乘法（`2n` FLOPs）。而 matrix-matrix product `AB` 有 `m × p` 个 entries，因此总 FLOPs 为 `(2n)(mp) = 2mnp`。

在做下一题前，逐个检查你的 Transformer block 和 Transformer LM 组件，列出所有 matrix multiplies 及其 FLOPs costs，会很有帮助。

**Problem (transformer_accounting): Transformer LM resource accounting (5 points)**

(a) 考虑一个使用本作业架构的 GPT-2 XL 大小模型，配置如下：

```text
vocab_size: 50,257
context_length: 1,024
num_layers: 48
d_model: 1,600
num_heads: 25
d_ff: 4,288 (the nearest multiple of 64 to 8/3 × 1,600)
```

假设用该配置构造模型。模型会有多少 trainable parameters？假设每个 parameter 使用 single-precision floating point 表示，仅加载该模型需要多少内存？  
Deliverable：一到两句话回答。

(b) 识别完成 GPT-2 XL-shaped model forward pass 所需的 matrix multiplies。这些 matrix multiplies 总共需要多少 FLOPs？假设输入序列有 `context_length` 个 tokens。  
Deliverable：列出 matrix multiplies（附描述），并给出所需 FLOPs 总数。

(c) 根据上面的分析，模型哪些部分需要最多 FLOPs？  
Deliverable：一到两句话回答。

(d) 对 GPT-2 small（12 layers, 768 `d_model`, 12 heads）、GPT-2 medium（24 layers, 1024 `d_model`, 16 heads）和 GPT-2 large（36 layers, 1280 `d_model`, 20 heads）重复分析。随着模型规模增大，Transformer LM 的哪些部分在总 FLOPs 中所占比例变多或变少？  
Deliverable：对每个模型，提供 model components 及其对应 FLOPs（占 forward pass 总 FLOPs 的比例）的 breakdown。此外，用一到两句话描述改变模型大小如何改变各组件 FLOPs 比例。

(e) 取 GPT-2 XL，并把 context length 增加到 16,384。一次 forward pass 的总 FLOPs 如何变化？各 model components 的 FLOPs 相对贡献如何变化？  
Deliverable：一到两句话回答。

## 4 Training a Transformer LM

现在我们已经有了预处理数据（通过 tokenizer）和模型（Transformer）的步骤。剩下的是构建所有支持训练的代码。这包括：

- Loss：需要定义 loss function（cross-entropy）。
- Optimizer：需要定义 optimizer 来最小化该 loss（AdamW）。
- Training loop：需要所有支持基础设施，包括加载数据、保存 checkpoints 和管理训练。

### 4.1 Cross-entropy loss

回忆一下，Transformer language model 为长度为 `m + 1` 的每个序列 `x` 以及 `i = 1, ..., m` 定义分布 `p_θ(x_{i+1} | x_{1:i})`。给定由长度为 `m + 1` 的序列构成的训练集 `D`，我们定义标准 cross-entropy（negative log-likelihood）loss function：

```text
ℓ(θ; D) = (1 / (|D| m)) * sum_{x∈D} sum_{i=1}^m -log p_θ(x_{i+1} | x_{1:i})
```

注意，Transformer 中一次 forward pass 会同时给出所有 `i = 1, ..., m` 的 `p_θ(x_{i+1} | x_{1:i})`。

具体来说，Transformer 为每个位置 `i` 计算 logits `o_i ∈ R^{vocab_size}`，从而得到：

```text
p(x_{i+1} | x_{1:i}) = softmax(o_i)[x_{i+1}]
                      = exp(o_i[x_{i+1}]) / sum_a exp(o_i[a])
```

Cross-entropy loss 通常相对于 logits vector `o_i ∈ R^{vocab_size}` 和 target `x_{i+1}` 定义。注意，`o_i[k]` 指 vector `o_i` 在 index `k` 处的值；这对应于 `x_{i+1}` 上的 Dirac delta distribution 和 predicted `softmax(o_i)` distribution 之间的 cross-entropy。

和 softmax 一样，实现 cross-entropy loss 也需要注意 numerical issues。

**Problem (cross_entropy): Implement cross-entropy (1 point)**

Deliverable：写一个函数计算 cross-entropy loss，它接受 predicted logits (`o_i`) 和 targets (`x_{i+1}`)，并计算 cross-entropy `ℓ_i = -log softmax(o_i)[x_{i+1}]`。函数应处理：

- 为数值稳定，减去最大元素。
- 尽可能抵消 `log` 和 `exp`。
- 处理任意额外 batch dimensions，并返回 batch 上的平均值。和第 3.2 节一样，我们假设 batch-like dimensions 总是在前面，位于 vocabulary size dimension 之前。

实现 `[adapters.run_cross_entropy]`，然后运行：

```bash
uv run pytest -k test_cross_entropy
```

**Perplexity**

Cross-entropy 足以用于训练，但评估模型时，我们还希望报告 perplexity。对于长度为 `m` 的序列，若遭受 cross-entropy losses `ℓ_1, ..., ℓ_m`：

```text
perplexity = exp((1 / m) * sum_i ℓ_i)
```

### 4.2 The SGD Optimizer

现在有了 loss function，我们开始探索 optimizers。最简单的 gradient-based optimizer 是 Stochastic Gradient Descent (SGD)。从随机初始化参数 `θ_0` 开始。然后对每一步 `t = 0, ..., T - 1`，执行更新：

```text
θ_{t+1} ← θ_t - α_t ∇L(θ_t; B_t)
```

其中 `B_t` 是从 dataset `D` 采样的随机 batch，learning rate `α_t` 和 batch size `|B_t|` 是 hyperparameters。

#### 4.2.1 Implementing SGD in PyTorch

要实现 optimizers，我们会 subclass PyTorch 的 `torch.optim.Optimizer` class。一个 `Optimizer` subclass 必须实现两个 methods：

```python
def __init__(self, params, ...)
```

应初始化 optimizer。这里 `params` 是要优化的 parameters 集合（或 parameter groups，如果用户想对模型不同部分使用不同 hyperparameters，例如 learning rates）。确保把 `params` 传给 base class 的 `__init__` method，它会存储这些 parameters 供 `step` 使用。你可以根据 optimizer 接受额外 arguments（例如 learning rate 很常见），并以 dictionary 形式把它们传给 base class constructor，keys 是你为这些 parameters 选择的 names（strings）。

```python
def step(self)
```

应对 parameters 做一次更新。Training loop 中，它会在 backward pass 后调用，因此你可以访问上一 batch 的 gradients。该 method 应遍历每个 parameter tensor `p` 并原地修改它们，即根据 gradient `p.grad`（如果存在）设置 `p.data`，其中 `p.data` 是与该 parameter 关联的 tensor，`p.grad` 表示 loss 对该 parameter 的 gradient。

PyTorch optimizer API 有一些细节，因此用例子解释更容易。为了让例子更丰富，我们会实现 SGD 的一个小变体：learning rate 随训练衰减，从 initial learning rate `α` 开始，随着时间采取越来越小的步长：

```text
θ_{t+1} = θ_t - α / sqrt(t + 1) * ∇L(θ_t; B_t)
```

这个 SGD 版本作为 PyTorch Optimizer 的实现如下：

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss
```

在 `__init__` 中，我们把 parameters 和默认 hyperparameters 传给 base class constructor（parameters 可能分组，每组有不同 hyperparameters）。如果 parameters 只是一个 `torch.nn.Parameter` objects 集合，base constructor 会创建一个单独 group，并给它分配默认 hyperparameters。然后在 `step` 中，先遍历每个 parameter group，再遍历该 group 中的每个 parameter，并应用上式。这里，我们把 iteration number 作为与每个 parameter 关联的 state：先读取该值，在 gradient update 中使用它，然后更新它。

API 规定用户可能传入 callable `closure`，用于在 optimizer step 前重新计算 loss。我们用的 optimizers 不需要它，但为了符合 API 添加它。

下面这个最小 training loop 例子展示了它如何工作：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)

for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.
```

这就是 typical training loop 结构：每次 iteration 中，计算 loss 并运行 optimizer step。训练 language models 时，learnable parameters 来自模型（在 PyTorch 中，`m.parameters()` 给出该集合）。Loss 会在 sampled batch of data 上计算，但 training loop 的基本结构相同。

**Problem (learning_rate_tuning): Tuning the learning rate (1 point)**

正如我们会看到的，learning rate 是最影响训练的 hyperparameters 之一。让我们在 toy example 中实践这一点。把上面的 SGD example 用另外三个 learning rate 值运行：`1e1`、`1e2` 和 `1e3`，只跑 10 个 training iterations。每个 learning rate 下 loss 会发生什么？它 decay 更快、更慢，还是 diverge（即训练过程中增加）？  
Deliverable：用一到两句话描述你观察到的行为。

### 4.3 AdamW

现代 language models 通常使用比 SGD 更复杂的 optimizers 训练。最近使用的大多数 optimizers 都是 Adam optimizer [D. P. Kingma et al., 2015] 的衍生。我们会使用 AdamW [I. Loshchilov et al., 2019]，它在近期工作中被广泛使用。AdamW 提出了对 Adam 的修改，通过添加 weight decay（每次 iteration 把 parameters 拉向 0）来改善 regularization，并且这种 weight decay 与 gradient update 解耦。我们将按照 I. Loshchilov et al. [23] algorithm 2 描述实现 AdamW。

AdamW 是 stateful 的：对每个 parameter，它维护 first 和 second moments 的 running estimate。因此，AdamW 用额外内存换取更好的 stability 和 convergence。除了 learning rate `α`，AdamW 还有一对 hyperparameters `(β_1, β_2)` 控制 moment estimates 的更新，以及 weight decay rate `λ`。典型应用把 `(β_1, β_2)` 设为 `(0.9, 0.999)`，但 LLaMA [H. Touvron et al., 2023] 和 GPT-3 [T. B. Brown et al., 2020] 等 large language models 常用 `(0.9, 0.95)`。算法可写作如下，其中 `ε` 是很小的值（例如 `10^{-8}`），用于在 `v` 极小时提高 numerical stability：

```text
Algorithm: AdamW Optimizer
1 init(θ)                 Initialize learnable parameters
2 m ← 0                   Initial value of first moment vector; same shape as θ
3 v ← 0                   Initial value of second moment vector; same shape as θ
4 for t = 1, ..., T do
5   Sample batch of data B_t
6   g ← ∇_θ ℓ(θ; B_t)      Compute gradient of loss
7   α_t ← α * sqrt(1 - β_2^t) / (1 - β_1^t)
8   θ ← θ - α λ θ          Apply weight decay
9   m ← β_1 m + (1 - β_1) g
10  v ← β_2 v + (1 - β_2) g^2
11  θ ← θ - α_t * m / (sqrt(v) + ε)
12 end for
```

注意 `t` 从 1 开始。现在你将实现这个 optimizer。

**Problem (adamw): Implement AdamW (2 points)**

Deliverable：把 AdamW optimizer 实现为 `torch.optim.Optimizer` 的 subclass。你的 class 应在 `__init__` 中接受 learning rate `α`，以及 `β`、`ε` 和 `λ` hyperparameters。为了帮助维护 state，base `Optimizer` class 提供一个 dictionary `self.state`，它把 `nn.Parameter` objects 映射到一个 dictionary，用来存储该 parameter 需要的任何信息（对 AdamW 来说，就是 moment estimates）。实现 `[adapters.get_adamw_cls]` 并确保通过：

```bash
uv run pytest -k test_adamw
```

**Problem (adamw_accounting): Resource accounting for training with AdamW (2 points)**

让我们计算运行 AdamW 需要多少 memory 和 compute。假设所有 tensors 都使用 `float32`。

(a) 运行 AdamW 需要多少 peak memory？请按照 parameters、activations、gradients 和 optimizer state 的 memory usage 分解答案。用 `batch_size` 和模型 hyperparameters（`vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads`）表达。假设 `d_ff = 8/3 × d_model`。

为简单起见，计算 activations memory usage 时只考虑以下 components：

- Transformer block
  - RMSNorm(s)
  - Multi-head self-attention sublayer：`QKV` projections、`QK^T` matrix multiply、softmax、weighted sum of values、output projection
  - Position-wise feed-forward (SwiGLU)：`W1`、`W2`、gate branch 上的 SiLU、element-wise product、`W3`
- final RMSNorm
- output embedding
- logits 上的 cross-entropy

Deliverable：给出 parameters、activations、gradients 和 optimizer state 的 algebraic expression，以及 total。

(b) 对 GPT-2 XL-shaped model 代入你的答案，得到只依赖 `batch_size` 的 expression。仍能放入 80GB memory 的最大 batch size 是多少？  
Deliverable：形如 `a ⋅ batch_size + b` 的 expression（`a`、`b` 为数值），以及表示最大 batch size 的数字。

(c) 运行 AdamW 的一步需要多少 FLOPs？  
Deliverable：一个 algebraic expression，并简要说明理由。

(d) Model FLOPs utilization (MFU) 定义为 observed throughput（tokens per second）相对于 hardware theoretical peak FLOP throughput 的比值 [A. Chowdhery et al., 2022]。NVIDIA H100 GPU 对 “float32”（实际是 TensorFloat-32，也就是 “bfloat19”）operations 的 theoretical peak 是 495 teraFLOP/s。假设你能达到 50% MFU，在单张 H100 上以 batch size 1024 训练 GPT-2 XL 400K steps 需要多久？遵循 J. Kaplan et al. [25] 和 J. Hoffmann et al. [26]，假设 backward pass 的 FLOPs 是 forward pass 的两倍。  
Deliverable：训练需要的小时数，并简要说明理由。

### 4.4 Learning rate scheduling

导致 loss 最快下降的 learning rate 值通常会随训练过程变化。训练 Transformers 时，典型做法是使用 learning rate schedule：一开始使用较大 learning rate，让早期更新更快，然后随着模型训练逐渐 decay 到较小值。本作业中，我们会实现训练 LLaMA [H. Touvron et al., 2023] 时使用的 cosine annealing schedule。有时也会使用 learning rate 再次升高（restarts）的 schedule，帮助越过 local minima。

Scheduler 只是一个函数，它接受当前 step `t` 和其他相关 parameters（例如 initial 和 final learning rates），并返回 step `t` 的 gradient update 应使用的 learning rate。最简单的 schedule 是 constant function，对任意 `t` 返回相同 learning rate。

Cosine annealing learning rate schedule 接受：(i) 当前 iteration `t`，(ii) maximum learning rate `α_max`，(iii) minimum（final）learning rate `α_min`，(iv) warm-up iterations 数 `T_w`，以及 (v) cosine annealing 的 final iteration `T_c`。Iteration `t` 的 learning rate 定义为：

- Warm-up：如果 `t < T_w`，则 `α_t = (t / T_w) α_max`。
- Cosine annealing：如果 `T_w ≤ t ≤ T_c`，则 `α_t = α_min + 1/2 (1 + cos(((t - T_w) / (T_c - T_w)) π)) (α_max - α_min)`。
- Post-annealing：如果 `t > T_c`，则 `α_t = α_min`。

**Problem (learning_rate_schedule): Implement cosine learning rate schedule with warmup (1 point)**

写一个函数，接受 `t`、`α_max`、`α_min`、`T_w` 和 `T_c`，并根据上面定义的 scheduler 返回 learning rate `α_t`。然后实现 `[adapters.get_lr_cosine_schedule]` 并确保通过：

```bash
uv run pytest -k test_get_lr_cosine_schedule
```

### 4.5 Gradient clipping

训练期间，我们有时会遇到产生大 gradients 的 training examples，这会使训练不稳定。为了缓解这一点，实践中常用一种技术：gradient clipping。其思想是在每个 backward pass 后、optimizer step 前，对 gradient norm 施加上限。

给定所有 parameters 的 gradient `g`，计算其 `ℓ2` norm `||g||_2`。如果该 norm 小于最大值 `M`，则保持 `g` 不变；否则，把 `g` 按 `M / (||g||_2 + ε)` 的因子缩小（其中加入小的 `ε`，如 `10^{-6}`，用于 numerical stability）。注意最终 norm 会略小于 `M`。

**Problem (gradient_clipping): Implement gradient clipping (1 point)**

写一个函数实现 gradient clipping。函数应接受 parameters 列表和 maximum `ℓ2` norm。它应原地修改每个 parameter gradient。使用 `ε = 10^{-6}`（PyTorch default）。然后实现 adapter `[adapters.run_gradient_clipping]` 并确保通过：

```bash
uv run pytest -k test_gradient_clipping
```

## 5 Training loop

现在终于可以把目前构建的主要组件组合起来：tokenized data、model 和 optimizer。

### 5.1 Data Loader

Tokenized data（例如在 `tokenizer_experiments` 中准备的数据）是单个 token 序列 `x = (x_1, ..., x_n)`。即使 source data 可能由独立 documents 构成（例如不同网页或 source code files），常见做法是把它们全部 concatenate 成单个 token 序列，并在它们之间加入 delimiter（例如 `<|endoftext|>` token）。

Data loader 把它变成 batches 流，其中每个 batch 包含 `B` 个长度为 `m` 的 sequences，并配对对应的 next tokens，长度也为 `m`。例如，当 `B = 1`、`m = 3` 时，`([x_2, x_3, x_4], [x_3, x_4, x_5])` 是一个可能的 batch。

以这种方式加载数据从多方面简化训练。首先，任意 `1 ≤ i ≤ n - m` 都给出一个有效 training sequence，因此采样 training sequences 很简单。由于所有 training sequences 长度相同，无需 padding input sequences，这能提高 hardware utilization（也通过增加 batch size `B` 提高）。最后，我们也不需要加载完整 dataset 来采样 training data，使处理原本不适合内存的大数据集变得容易。

**Problem (data_loading): Implement data loading (2 points)**

Deliverable：写一个函数，接受 numpy array `x`（包含 token IDs 的 integer array）、`batch_size`、`context_length` 和 PyTorch device string（例如 `'cpu'` 或 `'cuda:0'`），并返回一对 tensors：sampled input sequences 和对应 next-token targets。两个 tensors 都应有 shape `(batch_size, context_length)`，包含 token IDs，并放在请求的 device 上。要用提供的测试测试实现，首先需要实现 test adapter `[adapters.run_get_batch]`。然后运行：

```bash
uv run pytest -k test_get_batch
```

**Low-Resource Tip: Data loading on CPU or Apple Silicon**

如果计划在 CPU 或 Apple Silicon 上训练 LM，需要把数据移动到正确 device（同样，之后模型也应使用相同 device）。

如果在 CPU 上，可以使用 `'cpu'` device string；在 Apple Silicon（M* chips）上，可以使用 `'mps'` device string。更多 MPS 资料：

- `https://docs.pytorch.org/docs/stable/mps.html`
- `https://docs.pytorch.org/docs/stable/notes/mps.html`
- `https://developer.apple.com/documentation/metalperformanceshaders`

如果 dataset 太大，无法加载进内存怎么办？可以使用名为 `mmap` 的 Unix system call，它把磁盘上的文件映射到 virtual memory，并在访问对应 memory location 时惰性加载文件内容。因此，你可以“假装”整个 dataset 都在内存中。Numpy 通过 `np.memmap` 实现这一点（如果最初用 `np.save` 保存 array，也可以在 `np.load` 中使用 `mmap_mode='r'` flag），它会返回一个 numpy array-like object，在你访问 entries 时按需加载。训练期间从 dataset（即 numpy array）采样时，请确保以 memory-mapped mode 加载 dataset（通过 `np.memmap` 或 `np.load` 的 `mmap_mode='r'` flag，取决于你如何保存 array）。还要确保指定与所加载 array 匹配的 dtype。

显式验证 memory-mapped data 看起来正确（例如，不包含超过预期 vocabulary size 的值）可能会有帮助。

### 5.2 Checkpointing

除了加载数据，我们训练时还需要保存模型。运行 jobs 时，我们经常希望能够恢复中途停止的 training run（例如由于 job timeout、machine failure 等）。即使一切顺利，我们也可能希望之后访问 intermediate models（例如事后研究 training dynamics、从不同阶段的模型采样等）。

Checkpoint 应包含恢复训练所需的所有 state。至少当然要能恢复 model weights。如果使用 stateful optimizer（例如 AdamW），还需要保存 optimizer state（例如 AdamW 的 moment estimates）。最后，为了恢复 learning rate schedule，需要知道停止时的 iteration number。PyTorch 让保存这些内容很容易：每个 `nn.Module` 都有 `state_dict()` method，返回包含所有 learnable weights 的 dictionary；之后可以用对应的 `load_state_dict()` method 恢复这些 weights。任何 `torch.optim.Optimizer` 也是如此。最后，`torch.save(obj, dest)` 可以把一个 object（例如某些 values 是 tensors、也包含普通 Python objects 如 integers 的 dictionary）dump 到 file（path）或 file-like object；然后可以用 `torch.load(src)` 重新加载到内存。

**Problem (checkpointing): Implement model checkpointing (1 point)**

实现以下两个函数来保存和加载 checkpoints：

```python
def save_checkpoint(model, optimizer, iteration, out)
```

应把 model、optimizer 和 iteration 的所有 state dump 到 file-like object `out`。可以使用 model 和 optimizer 的 `state_dict` method 获取相关 states，并使用 `torch.save(obj, out)` 把 `obj` dump 到 `out`（PyTorch 这里支持 path 或 file-like object）。典型选择是让 `obj` 成为 dictionary，但只要之后能加载 checkpoint，你可以使用任何格式。

参数：

- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`
- `iteration: int`
- `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`

```python
def load_checkpoint(src, model, optimizer)
```

应从 `src`（path 或 file-like object）加载 checkpoint，然后从该 checkpoint 恢复 model 和 optimizer states。函数应返回保存到 checkpoint 的 iteration number。可以使用 `torch.load(src)` 恢复你在 `save_checkpoint` 实现中保存的内容，并用 model 和 optimizer 的 `load_state_dict` method 把它们恢复到之前状态。

参数：

- `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`

实现 `[adapters.run_save_checkpoint]` 和 `[adapters.run_load_checkpoint]` adapters，并确保通过：

```bash
uv run pytest -k test_checkpointing
```

### 5.3 Training loop

现在终于到了把你实现的所有组件放进主训练脚本的时候。让用不同 hyperparameters 启动 training runs 变得容易（例如把它们作为 command-line arguments）会很有回报，因为之后你会多次运行这些实验，研究不同选择如何影响训练。

**Problem (training_together): Put it together (4 points)**

Deliverable：写一个脚本，运行 training loop，在用户提供的输入上训练模型。具体来说，建议你的 training script 至少允许：

- 配置并控制各种 model 和 optimizer hyperparameters。
- 使用 `np.memmap` 对大型 training 和 validation datasets 进行 memory-efficient loading。
- 把 checkpoints 序列化到用户提供的路径。
- 周期性记录 training 和 validation performance（例如记录到 console 和/或 Weights and Biases 等外部服务）。

## 6 Generating text

现在我们可以训练模型了，还剩最后一块：让模型生成文本。回忆一下，language model 接受一个（可能 batched 的）长度为 `sequence_length` 的 integer sequence，并产生大小为 `(sequence_length, vocab_size)` 的 matrix，其中序列的每个元素都是预测该位置后下一个 token 的 probability distribution。现在我们会写几个函数，把它变成新序列的 sampling scheme。

**Softmax**

按照标准约定，language model output 是 final linear layer 的输出（“logits”），因此需要通过前面 Equation 10 中看到的 softmax operation，把它转换为 normalized probability。

**Decoding**

为了从模型生成文本（decode），我们会向模型提供 prefix tokens 序列（“prompt”），并要求它产生 vocabulary 上的 probability distribution，用于预测序列中的下一个 token。然后，我们从这个 vocabulary items 分布中采样，决定下一个 output token。

具体来说，decoding process 的一步应接受 sequence `x_{1...t}`，并通过下面方程返回 token `x_{t+1}`：

```text
P(x_{t+1} = i | x_{1...t}) = exp(v_i) / sum_j exp(v_j)
v = TransformerLM(x_{1...t})_t ∈ R^{vocab_size}
```

其中 `TransformerLM` 是我们的模型，它输入长度为 `sequence_length` 的序列，并产生大小为 `(sequence_length, vocab_size)` 的 matrix；由于我们要找第 `t` 个 position 的 next-token prediction，所以取这个 matrix 的最后一个元素。

通过重复从这些 one-step conditionals 中采样（把上一步生成的 output token 追加到下一 decoding timestep 的输入中），直到生成 end-of-sequence token `<|endoftext|>`（或达到用户指定的 maximum number of tokens to generate），就得到一个 basic decoder。

**Decoder tricks**

我们会用小模型实验，而小模型有时生成质量很低的文本。两个简单 decoder tricks 可以帮助改善这些问题。首先，在 temperature scaling 中，我们使用 temperature parameter `τ` 修改 softmax，新的 softmax 为：

```text
softmax(v, τ)_i = exp(v_i / τ) / sum_j exp(v_j / τ)
```

注意当 `τ → 0` 时，`v` 的最大元素会占主导，softmax 输出变为集中在该最大元素上的 one-hot vector。

第二个技巧是 nucleus 或 top-p sampling，我们通过截断 low-probability tokens 来修改 sampling distribution。令 `q` 为从大小为 `vocab_size` 的（temperature-scaled）softmax 得到的 probability distribution。Nucleus sampling with hyperparameter `p` 根据下面方程产生 next token：

```text
P(x_{t+1} = i | q) =
    q_i / sum_{j∈V(p)} q_j,  if i ∈ V(p)
    0,                       otherwise
```

其中 `V(p)` 是最小的 indices 集合，使得 `sum_{j∈V(p)} q_j ≥ p`。可以先按大小排序 probability distribution `q`，再选择最大的 vocabulary elements，直到达到目标 `p`，从而轻松计算该集合。

**Problem (decoding): Decoding (3 points)**

Deliverable：实现一个函数，从 language model decode。建议支持以下 features：

- 为用户提供的 prompt 生成 completions（即输入某个 `x_{1...t}` 并采样 completion，直到遇到 `<|endoftext|>` token）。
- 允许用户控制 maximum number of generated tokens。
- 给定 desired temperature value，在采样前对 predicted next-token distributions 应用 softmax temperature scaling。
- Top-`p` sampling（[A. Holtzman et al., 2020]，也称为 nucleus sampling），给定用户指定 threshold value。

## 7 Experiments

现在是时候把一切组合起来，在 pretraining dataset 上训练（小型）language models 了。

### 7.1 如何运行实验和 Deliverables

理解 Transformer 架构组件背后理由的最好方式，是亲自修改它并运行它。没有什么能替代 hands-on experience。

为此，能够快速、一致地实验，并记录你做了什么，非常重要。为了快速实验，我们会在 small-scale model（约 17M total parameters）和简单数据集（TinyStories）上运行许多实验。为了保持一致，你将系统性地 ablate components 和改变 hyperparameters；为了记录，我们会要求你提交实验日志以及每个实验对应的 learning curves。

为了能够提交 loss curves，请确保周期性评估 validation losses，并同时记录 step 数和 wall-clock times。你可能会发现 Weights and Biases 等 logging infrastructure 很有帮助。

**Problem (experiment_log): Experiment logging (3 points)**

为 training 和 evaluation code 创建 experiment tracking infrastructure，使你能够相对于 gradient steps 和 wall-clock time 跟踪 experiments 和 loss curves。

Deliverable：用于实验的 logging infrastructure code，以及本节后续作业问题所需的 experiment log（记录你尝试过的所有事情的文档）。

### 7.2 TinyStories

我们将从一个非常简单的数据集 TinyStories [R. Eldan et al. [1]] 开始，模型会训练得很快，我们也能看到一些有趣行为。获取该数据集的说明在第 1 节。下面是该数据集的一个例子。

**Example (tinystories_example): One example from TinyStories**

```text
Once upon a time there was a little boy named Ben. Ben loved to explore the world around him.
He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben
was walking through the store when he came across a very special vase. When Ben saw it he was
amazed! He said, “Wow, that is a really amazing vase! Can I buy it?” The shopkeeper smiled and
said, “Of course you can. You can take it home and show all your friends how amazing it is!” So
Ben took the vase home and he was so proud of it! He called his friends over and showed them
the amazing vase. All his friends thought the vase was beautiful and couldn’t believe how lucky
Ben was. And that’s how Ben found an amazing vase in the store!
```

#### 7.2.1 Hyperparameter tuning

我们会告诉你一些非常基础的 hyperparameters 作为起点，并要求你为其他 hyperparameters 找到表现良好的设置。

`Vocab size 10000`。典型 vocabulary size 在几万到几十万之间。你应改变它，看看 vocabulary 和 model behavior 如何变化。

`Context length 256`。像 TinyStories 这样的简单数据集可能不需要长 sequence lengths，但后面的 OpenWebText 数据可能需要。尝试改变它，观察它对 per-iteration runtime 和 final perplexity 的影响。

`d_model 512`。这略小于许多 small Transformer papers 使用的 768 dimensions，但会让运行更快。

`d_ff 1344`。这大约是 `8/3 d_model`，同时是 64 的倍数，有利于 GPU performance。

RoPE theta parameter `Θ 10000`。

Number of layers and heads：4 layers，16 heads。合起来约有 17M non-embedding parameters，是相当小的 Transformer。

Total tokens processed：327,680,000（你的 `batch size × total step count × context length` 应大致等于该值）。

你应通过 trial and error 为以下其他 hyperparameters 找到好默认值：learning rate、learning rate warmup、其他 AdamW hyperparameters（`β_1`、`β_2`、`ε`）以及 weight decay。可以在 D. P. Kingma et al. [22] 中找到这些 hyperparameters 的一些典型选择。

#### 7.2.2 Putting it together

现在可以把所有东西组合起来：获取训练好的 BPE tokenizer，tokenize training dataset，并在你写的 training loop 中运行。重要提示：如果你的实现正确且高效，上述 hyperparameters 在 1 张 B200 GPU 上应大约运行 20–30 分钟。如果你的 runtime 长很多，请检查并确保 dataloading、checkpointing 或 validation loss code 没有成为瓶颈，并且你的实现 properly batched。

#### 7.2.3 Tips and tricks for debugging model architectures

我们强烈建议你熟悉 IDE 内置 debugger（例如 VSCode/Zed），它会比用 print statements 调试更省时间。如果使用 text editor，可以用类似 `ipdb` 的工具。调试 model architectures 时，还有一些好习惯：

- 开发任何 neural net architecture 时，一个常见的第一步是 overfit 到单个 minibatch。如果实现正确，应能很快把 training loss 降到接近 zero。
- 在各个 model components 中设置 debug breakpoints，检查 intermediate tensors 的 shapes，确保它们符合预期。
- 监控 activations、model weights 和 gradients 的 norms，确保它们没有 exploding 或 vanishing。

**Problem (learning_rate): Tune the learning rate (2 B200 hrs) (3 points)**

Learning rate 是最重要的 hyperparameters 之一。基于你训练的 base model，回答以下问题：

(a) 对 learning rates 做 hyperparameter sweep，并报告 final losses（如果 optimizer diverges，则说明 divergence）。  
Deliverable：多个 learning rates 对应的 learning curves。解释你的 hyperparameter search strategy。  
Deliverable：一个在 TinyStories 上 validation loss（per-token）不超过 1.45 的模型。

**Low-Resource Tip: Train for a few steps on CPU or Apple Silicon**

如果你在 CPU 或 MPS 上运行，应把 total tokens processed count 降到 40,000,000，这足以产生相当流畅的文本。也可以把 target validation loss 从 1.45 提高到 2.00。

在配备 36 GB RAM 的 M4 Max chip 上运行我们的 solution code，并调好 learning rate，我们使用 `batch size × total step count × context length = 32 × 5000 × 256 = 40,960,000` tokens，在 CPU 上需要 1 小时 22 分钟，在 MPS 上需要 36 分钟。在 step 5000 时，validation loss 达到 1.80。

额外建议：

- 使用 `N` training steps 时，建议调整 cosine learning rate decay schedule，使其 decay 在正好 step `N` 时结束（即达到 minimum learning rate）。
- 使用 MPS 时，不要使用 TF32 kernels，即不要像在 cuda devices 上可能做的那样设置：

```python
torch.set_float32_matmul_precision('high')
```

我们在 MPS（torch version 2.9.0）上尝试启用 TF32 kernels，发现 backend 有时会使用 silently broken kernels，导致 unstable training。

- 可以通过 JIT-compiling 模型加速训练：
  - 在 CPU 上，使用 `model = torch.compile(model)`。
  - 在 MPS 上，可以用 `model = torch.compile(model, backend="aot_eager")` 在一定程度上优化 backward pass。
  - 截至 torch version 2.9.0，MPS 不支持使用 Inductor 编译。

(b) 经验上常说最好的 learning rate 处在 “edge of stability”。研究 learning rates diverge 的点与你的最佳 learning rate 之间的关系。  
Deliverable：逐渐增大 learning rate 的 learning curves，其中至少包含一个 divergent run，并分析这与 convergence rates 的关系。

现在改变 batch size，看看训练会发生什么。Batch sizes 很重要：它们让我们通过更大的 matrix multiplies 更高效地使用 GPUs。但我们是否总是希望 batch size 越大越好？让我们运行实验看看。

**Problem (batch_size_experiment): Batch size variations (1 B200 hr) (1 point)**

把 batch size 从 1 一直变化到 GPU memory limit。中间至少尝试几个 batch sizes，包括 64 和 128 等典型大小。

Deliverable：不同 batch sizes 的 runs 对应的 learning curves。必要时应重新优化 learning rates。  
Deliverable：用几句话讨论你关于 batch sizes 及其对训练影响的发现。

有了 decoder 后，现在可以生成文本了。我们会从模型生成文本并观察质量。作为参考，你的输出应至少和下面例子一样好。

**Example (ts_generate_example): Sample output from a TinyStories language model**

```text
Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big
black one. One day, Lily’s mom asked her to help cook dinner. Lily was so excited! She loved to
help her mom. Lily’s mom made a big pot of soup for dinner. Lily was so happy and said, “Thank
you, Mommy! I love you.” She helped her mom pour the soup into a big bowl. After dinner, Lily’s
mom made some yummy soup. Lily loved it! She said, “Thank you, Mommy! This soup is so
yummy!” Her mom smiled and said, “I’m glad you like it, Lily.” They finished cooking and
continued to cook together. The end.
```

**Low-Resource Tip: Generate text on CPU or Apple Silicon**

如果你使用的是处理 40M tokens 的 low-resource configuration，生成结果仍应像英语，但不会像上面那样流畅。例如，我们从训练在 40M tokens 上的 TinyStories language model 得到的 sample output 如下：

```text
Once upon a time, there was a little girl named Sue. Sue had a tooth that she loved very much. It
was his best head. One day, Sue went for a walk and met a ladybug! They became good friends
and played on the path together.
“Hey, Polly! Let’s go out!” said Tim. Sue looked at the sky and saw that it was difficult to find a
way to dance shining. She smiled and agreed to help the talking!“
As Sue watched the sky moved, what it was. She
```

下面是精确 problem statement 和要求：

**Problem (generate): Generate text (1 point)**

使用你的 decoder 和 trained checkpoint，报告模型生成的文本。你可能需要调整 decoder parameters（temperature、top-p 等）以得到流畅输出。

Deliverable：至少 256 tokens 的文本 dump（或直到第一个 `<|endoftext|>` token），并简要评论该输出的 fluency，以及至少两个影响输出好坏的因素。

### 7.3 Ablations and architecture modification

理解 Transformer 的最佳方式是实际修改它并观察行为。现在我们会做几个简单 ablations 和 modifications。

**Ablation 1: layer normalization**

常有人说 layer normalization 对 Transformer training stability 很重要。但也许我们想冒险一点。让我们从每个 Transformer block 中移除 RMSNorm，看看会发生什么。

**Problem (layer_norm_ablation): Remove RMSNorm and train (0.5 B200 hrs) (1 point)**

从 Transformer 中移除所有 RMSNorms 并训练。在之前的 optimal learning rate 下会发生什么？能否通过使用更低 learning rate 获得稳定性？

Deliverable：移除 RMSNorms 后训练的 learning curve，以及最佳 learning rate 对应的 learning curve。  
Deliverable：几句话评论 RMSNorm 的影响。

现在研究另一个乍看任意的 layer normalization 选择。Pre-norm Transformer blocks 定义为：

```text
z = x + MultiHeadSelfAttention(RMSNorm(x))
y = z + FFN(RMSNorm(z))
```

这是对原始 Transformer 架构的少数 “consensus” modifications 之一。原始 Transformer 使用 post-norm 方法：

```text
z = RMSNorm(x + MultiHeadSelfAttention(x))
y = RMSNorm(z + FFN(z))
```

让我们恢复 post-norm 方法，看看会发生什么。

**Problem (pre_norm_ablation): Implement post-norm and train (0.5 B200 hrs) (1 point)**

把你的 pre-norm Transformer 实现修改为 post-norm。用 post-norm model 训练，看看会发生什么。

Deliverable：post-norm Transformer 的 learning curve，并与 pre-norm 对比。

我们会看到 layer normalization 对 Transformer 行为有重大影响，而且 layer normalization 的位置也很重要。

**Ablation 2: position embeddings**

接下来研究 position embeddings 对模型性能的影响。具体来说，我们会比较 base model（带 RoPE）与完全不包含 position embeddings 的模型（NoPE）。事实证明，decoder-only transformers，也就是我们实现的带 causal mask 的 transformers，理论上即使没有显式提供 position embeddings，也可以推断 relative 或 absolute position information [Y.-H. H. Tsai et al., 2019; A. Kazemnejad et al., 2023]。现在我们 empirically 测试 NoPE 相比 RoPE 表现如何。

**Problem (no_pos_emb): Implement NoPE (0.5 B200 hrs) (1 point)**

修改你的带 RoPE 的 Transformer 实现，完全移除 position embedding information，看看会发生什么。

Deliverable：比较 RoPE 和 NoPE 表现的 learning curve。

**Ablation 3: SwiGLU vs. SiLU**

接下来，我们遵循 N. Shazeer [20]，通过比较使用 SwiGLU feed-forward networks 与使用 SiLU activations 但没有 gated linear unit (GLU) 的 feed-forward networks，测试 feed-forward network 中 gating 的重要性：

```text
FFN_SiLU(x) = W2 SiLU(W1 x)
```

回忆在 SwiGLU 实现中，我们把 inner feed-forward layer 的 dimensionality 设为约 `d_ff = 8/3 d_model`（同时确保 `d_ff mod 64 = 0`，以利用 GPU tensor cores）。在这个 ablation baseline 中，你的 `FFN_SiLU` 实现应改为设置 `d_ff = 4 × d_model`，以大致匹配默认 SwiGLU feed-forward network 的 parameter count（默认 SwiGLU 有三个而不是两个 weight matrices）。

**Problem (swiglu_ablation): SwiGLU vs. SiLU (0.5 B200 hrs) (1 point)**

Deliverable：在 parameter counts 大致匹配的情况下，比较 SwiGLU 和 SiLU feed-forward networks 性能的 learning curve。  
Deliverable：几句话讨论你的发现。

**Low-Resource Tip: Online students with limited GPU resources should test modifications on TinyStories**

在作业剩余部分，我们会转向更大规模、更嘈杂的 web dataset（OpenWebText），实验 architecture modifications，并（可选）向课程 leaderboard 提交。

在 OpenWebText 上训练 LM 到流畅需要很长时间，因此建议 GPU 资源有限的在线学生继续在 TinyStories 上测试 modifications（使用 validation loss 作为评估性能的 metric）。

### 7.4 Running on OpenWebText

现在我们转向一个由 web crawl 创建的更标准 pretraining dataset。OpenWebText [A. Gokaslan et al., 2019] 的一个小 sample 也以单个文本文件形式提供：如何访问该文件见第 1 节。

下面是 OpenWebText 中的一个例子。注意该文本更真实、复杂且多样。你可能想浏览 training dataset，以了解 web-scraped corpus 的 training data 长什么样。

**Example (owt_example): One example from OWT**

```text
Baseball Prospectus director of technology Harry Pavlidis took a risk when he hired Jonathan
Judge.
Pavlidis knew that, as Alan Schwarz wrote in The Numbers Game, “no corner of American
culture is more precisely counted, more passionately quantified, than performances of baseball
players.” With a few clicks here and there, you can find out that Noah Syndergaard’s fastball
revolves more than 2,100 times per minute on its way to the plate, that Nelson Cruz had the
game’s highest average exit velocity among qualified hitters in 2016 and myriad other tidbits that
seem ripped from a video game or science fiction novel. The rising ocean of data has empowered
an increasingly important actor in baseball’s culture: the analytical hobbyist.
That empowerment comes with added scrutiny – on the measurements, but also on the people
and publications behind them. With Baseball Prospectus, Pavlidis knew all about the backlash
that accompanies quantitative imperfection. He also knew the site’s catching metrics needed to be
reworked, and that it would take a learned mind – someone who could tackle complex statistical
modeling problems – to complete the job.
“He freaks us out.” Harry Pavlidis
Pavlidis had a hunch that Judge “got it” based on the latter’s writing and their interaction at a
site-sponsored ballpark event. […]
```

注意：这个实验可能需要你重新调试 hyperparameters，例如 learning rate 或 batch size。

**Problem (main_experiment): Experiment on OWT (2 B200 hrs) (2 points)**

用与 TinyStories 相同的 model architecture 和 total training iterations 在 OpenWebText 上训练 language model。这个模型表现如何？

Deliverable：你的 language model 在 OpenWebText 上的 learning curve。描述与 TinyStories 的 losses 差异；我们应如何解释这些 losses？  
Deliverable：OpenWebText LM 生成的文本，格式与 TinyStories outputs 相同。该文本 fluency 如何？为什么即使使用与 TinyStories 相同的模型和 compute budget，输出质量仍更差？

### 7.5 Your own modification + leaderboard

恭喜你走到这里。你快完成了！现在你将尝试改进 Transformer architecture，并看看你的 hyperparameters 和 architecture 与班上其他同学相比如何。

**Rules for the leaderboard**

除了以下规则外没有限制：

Runtime：你的 submission 在 B200 上最多运行 45 分钟。如果使用 SLURM 或 Modal，可能需要在 submission script 中强制限制这一点。

Data：你只能使用我们提供的 OpenWebText training dataset。

除此之外，你可以做任何你想做的事情。

如果想找一些实现思路，可以看这些资源：

- State-of-the-art open-source LLM families，例如 Llama 3 [A. Grattafiori et al., 2024] 或 Qwen 2.5 [A. Yang et al., 2024]。
- NanoGPT speedrun repository（`github.com/KellerJordan/modded-nanogpt`），community members 在其中发布许多用于 “speedrunning” small-scale language model pretraining 的有趣 modifications。例如，一个可追溯到原始 Transformer 论文的常见 modification 是 tie input 和 output embeddings 的 weights（见 A. Vaswani et al. [8] Section 3.4 和 A. Chowdhery et al. [16] Section 2）。如果尝试 weight tying，可能需要降低 embedding/LM head init 的 standard deviation。

在尝试完整 45-minute run 前，你需要先在 OpenWebText 小子集或 TinyStories 上测试这些 modifications。

需要提醒的是，我们注意到 leaderboard 中一些效果好的 modifications 可能无法 generalize 到更大规模 pretraining。我们会在课程的 scaling laws 单元进一步探索这个想法。

**Problem (leaderboard): Leaderboard (10 B200 hrs) (6 points)**

你将在上述 leaderboard rules 下训练模型，目标是在 0.75 B200-hours 内最小化 language model 的 validation loss。

Deliverable：记录到的 final validation loss、一条清楚显示 wall-clock-time x-axis 小于 45 分钟的相关 learning curve，以及你做了什么的描述。我们期望 leaderboard submission 至少超过 naive baseline，即 loss 5.0。提交到 leaderboard：

```text
github.com/stanford-cs336/assignment1-basics-leaderboard
```

## Bibliography

[1] R. Eldan and Y. Li, “TinyStories: How Small Can Language Models Be and Still Speak Coherent English?.” 2023.

[2] A. Gokaslan, V. Cohen, E. Pavlick, and S. Tellex, “OpenWebText corpus.” 2019.

[3] R. Sennrich, B. Haddow, and A. Birch, “Neural Machine Translation of Rare Words with Subword Units,” in Proc. of ACL, 2016.

[4] C. Wang, K. Cho, and J. Gu, “Neural Machine Translation with Byte-Level Subwords.” 2019.

[5] P. Gage, “A new algorithm for data compression,” C Users Journal, vol. 12, no. 2, pp. 23–38, Feb. 1994.

[6] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, “Language Models are Unsupervised Multitask Learners.” 2019.

[7] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, “Improving Language Understanding by Generative Pre-Training.” 2018.

[8] A. Vaswani et al., “Attention is All you Need,” in Proc. of NeurIPS, 2017.

[9] T. Q. Nguyen and J. Salazar, “Transformers without Tears: Improving the Normalization of Self-Attention,” in Proc. of IWSWLT, 2019.

[10] R. Xiong et al., “On Layer Normalization in the Transformer Architecture,” in Proc. of ICML, 2020.

[11] J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer Normalization.” 2016.

[12] H. Touvron et al., “LLaMA: Open and Efficient Foundation Language Models.” 2023.

[13] B. Zhang and R. Sennrich, “Root Mean Square Layer Normalization,” in Proc. of NeurIPS, 2019.

[14] A. Grattafiori et al., “The Llama 3 Herd of Models.” [Online]. Available: `https://arxiv.org/abs/2407.21783`

[15] A. Yang et al., “Qwen2.5 Technical Report,” arXiv preprint arXiv:2412.15115, 2024.

[16] A. Chowdhery et al., “PaLM: Scaling Language Modeling with Pathways.” 2022.

[17] D. Hendrycks and K. Gimpel, “Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units.” 2016.

[18] S. Elfwing, E. Uchibe, and K. Doya, “Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning.” [Online]. Available: `https://arxiv.org/abs/1702.03118`

[19] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, “Language Modeling with Gated Convolutional Networks.” [Online]. Available: `https://arxiv.org/abs/1612.08083`

[20] N. Shazeer, “GLU Variants Improve Transformer.” 2020.

[21] J. Su, Y. Lu, S. Pan, B. Wen, and Y. Liu, “RoFormer: Enhanced Transformer with Rotary Position Embedding.” 2021.

[22] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” in Proc. of ICLR, 2015.

[23] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” in Proc. of ICLR, 2019.

[24] T. B. Brown et al., “Language Models are Few-Shot Learners,” in Proc. of NeurIPS, 2020.

[25] J. Kaplan et al., “Scaling Laws for Neural Language Models.” 2020.

[26] J. Hoffmann et al., “Training Compute-Optimal Large Language Models.” 2022.

[27] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, “The Curious Case of Neural Text Degeneration,” in Proc. of ICLR, 2020.

[28] Y.-H. H. Tsai, S. Bai, M. Yamada, L.-P. Morency, and R. Salakhutdinov, “Transformer Dissection: An Unified Understanding for Transformer`s Attention via the Lens of Kernel,” in Proceedings of EMNLP-IJCNLP, Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 4344–4353. doi: 10.18653/v1/D19-1443.

[29] A. Kazemnejad, I. Padhi, K. Natesan, P. Das, and S. Reddy, “The Impact of Positional Encoding on Length Generalization in Transformers,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: `https://openreview.net/forum?id=Drrl2gcjzl`


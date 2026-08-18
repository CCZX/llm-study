# LLM 学习笔记

这是一个面向大模型学习的 Python 实践项目。当前内容聚焦 NumPy 数组基础、广播、向量化和矩阵运算，并结合大模型中常见的 `(batch, sequence, hidden)` 数据形状进行练习。

## 项目内容

```text
.
├── 第一阶段/
│   ├── 第08天-NumPy数组.md          # NumPy 数组讲义
│   ├── 第08天-NumPy数组-练习.py     # 带自动检查的练习
│   ├── 第09天-广播向量化与矩阵运算.md      # 广播与矩阵运算讲义
│   └── 第09天-广播向量化与矩阵运算-练习.py # 带自动检查的练习
└── Untitled.ipynb                   # NumPy 实验 Notebook
```

## 环境要求

- Python 3.11 或更高版本
- NumPy
- JupyterLab（运行 Notebook 时需要）

## 快速开始

在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy jupyterlab
```

Windows PowerShell 使用下面的命令激活虚拟环境：

```powershell
.venv\Scripts\Activate.ps1
```

## 学习方式

建议按以下顺序学习：

1. 阅读 [`第一阶段/第08天-NumPy数组.md`](第一阶段/第08天-NumPy数组.md)，完成[第 8 天练习](第一阶段/第08天-NumPy数组-练习.py)。
2. 阅读 [`第一阶段/第09天-广播向量化与矩阵运算.md`](第一阶段/第09天-广播向量化与矩阵运算.md)，完成[第 9 天练习](第一阶段/第09天-广播向量化与矩阵运算-练习.py)。
3. 打开 `Untitled.ipynb`，运行示例并尝试修改数组形状和运算方式。
4. 将练习文件中赋值右侧的 `...` 替换为 NumPy 表达式，根据自动检查结果修正答案。

启动 JupyterLab：

```bash
jupyter lab
```

运行练习：

```bash
python "第一阶段/第08天-NumPy数组-练习.py"
python "第一阶段/第09天-广播向量化与矩阵运算-练习.py"
```

脚本会为每道题输出 `[通过]`、`[未通过]` 或 `[待完成]`，未填写的题目不会中断后续检查。

## 大模型学习路线

整体主线：

```text
Python 与数学基础 → PyTorch → Transformer 原理 → 开源模型 → RAG → 微调与对齐 → 部署与工程化
```

预计用时为 4～6 个月。建议先掌握模型、检索和评测等底层能力，再根据实际需求选择 Agent 框架。

### 第一阶段：编程与数学基础（2～3 周）

学习内容：

- Python：函数、类、迭代器、装饰器和虚拟环境
- NumPy、Pandas、Jupyter
- Git 和 Linux 命令行
- 线性代数：向量、矩阵乘法、内积和特征值
- 微积分：导数、梯度和链式法则
- 概率：条件概率、期望和交叉熵
- 机器学习基础：数据集划分、过拟合和梯度下降

阶段项目：使用 NumPy 实现线性回归、Softmax 和简单神经网络。

当前仓库正处于这一阶段，现阶段目标是：

- 理解 `ndarray`、`shape`、`ndim`、`size` 和 `dtype`
- 掌握多维数组的创建、索引与切片
- 使用 `reshape` 调整数组形状
- 使用 `sum`、`mean` 和 `max` 完成聚合计算
- 根据 `axis` 推断计算结果的形状
- 理解大模型中常见的 `(batch, sequence, hidden)` 张量结构
- 掌握广播、向量化、点积和矩阵乘法
- 使用批量矩阵运算实现线性变换和余弦相似度

### 第二阶段：深度学习与 PyTorch（3～4 周）

学习内容：

- Tensor 与广播
- `Dataset`、`DataLoader` 和 `nn.Module`
- 前向传播、反向传播和自动微分
- 损失函数、优化器和学习率
- 正则化、Dropout 和归一化
- 模型训练、保存与加载

阶段项目：完成文本分类器和字符级语言模型，并在不依赖高级 Trainer 的情况下自己编写训练循环。可从 [PyTorch 官方基础教程](https://docs.pytorch.org/tutorials/beginner/basics/intro)开始。

### 第三阶段：Transformer 原理（3～4 周）

学习内容：

- Tokenization、BPE 和 Embedding
- Self-Attention 与 Query、Key、Value
- Multi-Head Attention 和位置编码
- Causal Mask、残差连接与 LayerNorm
- Encoder 与 Decoder
- 预训练目标和下一个 Token 预测
- Temperature、Top-k、Top-p 和 KV Cache

阶段项目：从零实现 Mini GPT，在小型文本数据集上完成训练、生成和采样。建议在理解讲解材料后阅读 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

### 第四阶段：使用开源模型（2～3 周）

学习 Hugging Face 生态中的 `transformers`、`datasets`、`tokenizers` 和 `accelerate`，掌握模型与 Tokenizer 加载、Chat Template、批量推理、Embedding 和模型量化基础。

阶段项目：运行一个开源指令模型，实现流式对话和结构化 JSON 输出，并比较不同 Prompt、采样参数与模型的结果。推荐学习 [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter1/1)。

### 第五阶段：LLM 应用开发与 RAG（3～4 周）

学习内容：

- API 调用、上下文管理、Prompt 和结构化输出
- Embedding、文档切分和语义检索
- 向量检索、关键词检索、混合检索和 Reranker
- 引用溯源和检索失败处理
- Tool Calling 与 Agent 的循环、状态和权限控制
- 缓存、重试和限流

阶段项目：实现一个带引用的个人知识库问答系统，支持 PDF/Markdown 导入、混合检索、回答引用和失败处理，并建立 20～50 条人工评测题，统计正确率、召回率与延迟。原理可参考 [RAG 原始论文](https://arxiv.org/abs/2005.11401)。

### 第六阶段：微调与对齐（3～4 周）

建议依次学习数据清洗与格式化、SFT、PEFT、LoRA、QLoRA、模型量化、偏好数据、DPO，以及 RLHF/RL 的基本概念。

阶段项目：准备小型领域数据集，对小参数模型进行 LoRA 微调，并比较微调前后的任务成功率、格式遵循率、幻觉率、推理延迟和通用能力变化。

### 第七阶段：部署与工程化（2～3 周）

学习内容：

- 模型显存估算以及 FP16、BF16、INT8、INT4
- Continuous Batching 和 KV Cache
- 吞吐量与首 Token 延迟
- OpenAI 兼容接口
- 日志、监控和 Tracing
- Prompt 注入与数据安全

阶段项目：将模型或 RAG 系统部署成 API，加入并发测试、日志、缓存、超时和错误处理。开源模型服务可从 [vLLM 官方文档](https://docs.vllm.ai/en/latest/)开始。

## 推荐项目顺序

1. PyTorch 文本分类器
2. 从零实现 Mini GPT
3. 开源模型聊天程序
4. 带引用的 RAG 知识库
5. LoRA 领域微调
6. 带自动评测的 Agent
7. 使用 vLLM 部署并进行压力测试

## 按职业方向调整重点

| 方向 | 重点投入 |
| --- | --- |
| LLM 应用工程师 | API、RAG、Agent、评测、后端与部署 |
| 大模型算法工程师 | Transformer、数据工程、SFT、LoRA、DPO、分布式训练 |
| 推理系统工程师 | CUDA、Triton、量化、KV Cache、并行策略、vLLM |
| 大模型研究 | 数学、论文复现、预训练、Scaling Law、对齐与推理 |
| 产品经理或创业者 | 模型边界、场景设计、评测、成本、安全与数据闭环 |

算法或研究方向可以在完成基础阶段后挑战 [Stanford CS336：Language Modeling from Scratch](https://cs336.stanford.edu/)。

建议将学习时间分配为：50% 编码和项目实践、30% 原理学习、20% 文档与论文阅读和复盘。是否真正掌握，应以能否解释模型失败原因、设计评测验证判断，以及独立完成可运行系统为标准。

## 参考资料

- [NumPy 初学者教程](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy 数组索引](https://numpy.org/doc/stable/user/basics.indexing.html)
- [NumPy 广播](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [NumPy 矩阵乘法](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)

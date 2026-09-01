# 第二阶段学习路线：深度学习与 PyTorch

## 阶段定位

第二阶段的任务是把第一阶段手写的 NumPy 模型迁移到 PyTorch，建立一套正确、可复用、可调试的深度学习训练流程，并为下一阶段从零实现 Transformer 做准备。

建议周期为 **4 周，每周学习 6～7 天，每天 2～3 小时**。如果已经熟悉其他深度学习框架，可以把前两周压缩为 1 周，但不应跳过自动微分、数据划分、训练/评估模式、检查点和小批量过拟合测试。

完成本阶段后，应当能够：

- 熟练使用 Tensor，并根据代码推断 shape、dtype 和 device
- 理解动态计算图、自动微分、叶子张量和梯度累积
- 使用 `Dataset`、`DataLoader` 和 `nn.Module` 组织项目
- 独立编写训练、验证、测试和推理循环
- 正确使用损失函数、优化器、学习率调度和梯度裁剪
- 理解初始化、权重衰减、Dropout、BatchNorm 和 LayerNorm
- 正确保存与恢复模型、优化器、轮次和实验配置
- 使用 Embedding 和池化实现文本分类器
- 使用字符级序列和 GRU 实现一个小型语言模型
- 用基线、曲线、指标和错误分析判断模型是否真正改进

## 学习原则

每天建议按下面的比例安排时间：

- 30～45 分钟：阅读教程和理解原理
- 60～90 分钟：独立编码、训练和调试
- 20～30 分钟：测试、记录实验和复盘错误

执行时遵守以下原则：

1. 每次张量运算前先写出预期 shape、dtype 和 device，再运行验证。
2. 先用 CPU 和极小数据验证正确性，再考虑 GPU/MPS 和完整数据。
3. 每个模型先尝试过拟合一个很小的 batch；做不到时不要直接增加训练轮数。
4. 训练集用于更新参数，验证集用于选择方案，测试集只用于最终报告。
5. 损失函数接收什么必须查清：例如 `CrossEntropyLoss` 接收未归一化 logits，不应提前做 Softmax。
6. 每轮明确切换 `model.train()` 和 `model.eval()`；验证和推理使用 `torch.no_grad()` 或 `torch.inference_mode()`。
7. 记录随机种子、数据版本、超参数、运行设备、最佳轮次和最终指标。
8. 不只保存模型文件；必须同时保存足以复现实验的配置和数据划分规则。
9. 先实现清晰、正确的普通训练循环，再学习高级 Trainer、分布式训练或复杂框架。
10. 每周至少完成一个可独立运行、可测试、可复现的小项目。

## 进入本阶段前的检查

开始前应能完成下面的任务：

- 使用 NumPy 写出 `X @ W + b`
- 解释 Softmax、交叉熵、梯度下降和反向传播
- 区分训练集、验证集和测试集
- 解释数据泄漏和过拟合
- 使用 Python 类、模块、异常处理和 pytest
- 使用 Git 查看差异并提交阶段性成果

如果其中任何一项还不熟练，先回看第一阶段对应内容，不要靠记忆 PyTorch API 掩盖概念缺口。

## 环境准备

继续使用项目根目录中的虚拟环境：

~~~bash
cd /Users/chenchao/Desktop/LLM
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
~~~

PyTorch 的安装命令与操作系统、加速设备有关，优先使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)生成命令。macOS 或只使用 CPU 时，可以先安装：

~~~bash
python -m pip install torch torchvision
python -m pip install matplotlib pandas scikit-learn jupyterlab pytest
~~~

安装后检查环境：

~~~bash
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available()); print(torch.cuda.is_available())"
~~~

统一选择设备：

~~~python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
~~~

注意：

- 不要为了使用 GPU 复制不匹配本机驱动或 CUDA 版本的安装命令。
- `model`、输入 Tensor 和标签 Tensor 必须位于兼容的 device。
- 固定随机种子可以减少实验波动，但不同 PyTorch 版本、平台和设备之间不保证完全一致。
- `.venv/`、下载的数据、检查点和大型日志通常不提交 Git；小型配置、测试和最终指标应提交。

## 核心资料

主线使用 [PyTorch Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)，按顺序完成：

- [Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Build the Neural Network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Automatic Differentiation](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
- [Optimizing Model Parameters](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Save and Load the Model](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)

辅助资料：

- [《动手学深度学习》深度学习计算](https://zh.d2l.ai/chapter_deep-learning-computation/index.html)
- [PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch NLP from Scratch](https://pytorch.org/tutorials/intermediate/nlp_from_scratch_index.html)
- [PyTorch Embedding 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [PyTorch GRU 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)

阅读资料时要自己重写关键代码，并用本阶段统一的训练、评估和测试结构组织，不直接复制完整 Notebook。

---

# 第一周：Tensor、自动微分与最小训练闭环

## 第 22 天：Tensor、dtype、shape 与 device

详细讲义与练习：

- [第22天-Tensor与设备.md](./第22天-Tensor与设备.md)
- [第22天-Tensor与设备-练习.py](./第22天-Tensor与设备-练习.py)

```bash
python "第二阶段/第22天-Tensor与设备-练习.py"
```

学习内容：

- `torch.tensor`、`zeros`、`ones`、`randn` 和 `arange`
- 索引、切片、`reshape`、`view`、`squeeze` 和 `unsqueeze`
- 广播、逐元素运算、`matmul` 和批量矩阵乘法
- `float32`、`float64`、`long` 和 `bool`
- NumPy 与 Tensor 的共享内存和复制区别
- CPU、CUDA、MPS 与 `.to(device)`

实践任务：

1. 将第一阶段 NumPy 的广播和矩阵运算练习改写为 PyTorch。
2. 对形状为 `(batch, sequence, hidden)` 的 Tensor 做按维求和、均值和线性变换。
3. 故意制造 shape、dtype 和 device 错误，并写下报错原因。
4. 比较 `torch.from_numpy`、`torch.tensor` 和 `.numpy()` 的行为。

验收标准：看到一段 Tensor 代码时，能在运行前写出关键中间结果的 shape、dtype 和 device。

## 第 23 天：计算图与自动微分

详细讲义与练习：

- [第23天-自动微分与计算图.md](./第23天-自动微分与计算图.md)
- [第23天-自动微分与计算图-练习.py](./第23天-自动微分与计算图-练习.py)

```bash
python "第二阶段/第23天-自动微分与计算图-练习.py"
```

学习内容：

- `requires_grad`、`grad_fn`、叶子张量和计算图
- `loss.backward()` 如何应用链式法则
- 标量输出与非标量输出的反向传播
- 梯度默认累积的原因
- `detach()`、`no_grad()` 和 `inference_mode()`
- 原地操作可能破坏计算图的原因

实践任务：

1. 用 PyTorch 重写第一阶段线性回归的前向传播。
2. 比较自动微分、手写解析梯度和中心差分数值梯度。
3. 连续调用两次 `backward()`，观察梯度累积。
4. 在训练循环中忘记清零梯度，记录损失与参数更新的异常。
5. 画出一个两层网络的张量 shape 与梯度流向。

验收标准：能解释为什么参数的 `.grad` 会累积，以及 `detach` 与 `no_grad` 分别切断什么。

## 第 24 天：使用 nn.Module 构建模型

详细讲义与练习：

- [第24天-nn.Module与模型构建.md](./第24天-nn.Module与模型构建.md)
- [第24天-nn.Module与模型构建-练习.py](./第24天-nn.Module与模型构建-练习.py)

```bash
python "第二阶段/第24天-nn.Module与模型构建-练习.py"
```

学习内容：

- `nn.Module`、`forward` 和 `__call__`
- `nn.Parameter` 与普通 Tensor 的区别
- `nn.Linear`、`nn.ReLU`、`nn.Sequential`
- 参数注册、子模块注册和 `state_dict`
- 参数初始化和模型结构打印

实践任务：

1. 实现 `LinearModel`，并与第一阶段 NumPy 线性模型对齐输出。
2. 实现 `MLP`：`Linear → ReLU → Linear`。
3. 打印所有参数的名称、shape 和是否需要梯度。
4. 写一个测试，验证 batch 大小改变时输出 shape 仍正确。
5. 故意把层放进普通 Python 容器，再与 `ModuleList` 的参数注册结果比较。

验收标准：能解释为什么把层赋值为模块属性后会被优化器发现，以及 `state_dict` 中保存了什么。

## 第 25 天：损失函数、优化器与训练循环

详细讲义与练习：

- [第25天-损失优化器与训练循环.md](./第25天-损失优化器与训练循环.md)
- [第25天-损失优化器与训练循环-练习.py](./第25天-损失优化器与训练循环-练习.py)

```bash
python "第二阶段/第25天-损失优化器与训练循环-练习.py"
```

学习内容：

- 回归的 MSE 与分类的交叉熵
- logits、概率、类别索引和 one-hot 标签
- SGD、Momentum 和 Adam 的基本区别
- `zero_grad → forward → loss → backward → step`
- epoch、batch、iteration 与全量平均损失

最小训练步骤：

~~~python
optimizer.zero_grad(set_to_none=True)
logits = model(inputs)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()
~~~

实践任务：

1. 用 `nn.Linear` 和 SGD 重写第一阶段线性回归。
2. 用 `nn.CrossEntropyLoss` 重写 Softmax 分类器。
3. 验证不手动做 Softmax 时分类损失仍能正确训练。
4. 分别记录“最后一个 batch 的 loss”和“按样本加权的整轮平均 loss”，解释差异。
5. 在一个小 batch 上训练到接近零损失。

验收标准：能从空文件开始写出一个正确的最小训练循环，不依赖复制现成模板。

## 第 26 天：Dataset 与 DataLoader

详细讲义与练习：

- [第26天-Dataset与DataLoader.md](./第26天-Dataset与DataLoader.md)
- [第26天-Dataset与DataLoader-练习.py](./第26天-Dataset与DataLoader-练习.py)

```bash
python "第二阶段/第26天-Dataset与DataLoader-练习.py"
```

学习内容：

- map-style `Dataset` 的 `__len__` 和 `__getitem__`
- `TensorDataset` 与自定义 `Dataset`
- `DataLoader` 的 `batch_size`、`shuffle` 和 `drop_last`
- batch 维、最后一个 batch 和数据类型
- 训练集 shuffle 与验证/测试集稳定顺序
- `num_workers` 的作用和平台差异

实践任务：

1. 为第一阶段 Iris 数据实现自定义 Dataset。
2. 分别创建训练、验证和测试 DataLoader。
3. 验证所有样本恰好被评估一次。
4. 检查分类标签是否为 `torch.long`。
5. 固定生成器种子，比较两次训练批次顺序。

验收标准：能够说明 Dataset 负责“单个样本”，DataLoader 负责“采样、组 batch 和迭代”。

## 第 27 天：评估、检查点与复现

详细讲义与练习：

- [第27天-评估检查点与复现.md](./第27天-评估检查点与复现.md)
- [第27天-评估检查点与复现-练习.py](./第27天-评估检查点与复现-练习.py)

```bash
python "第二阶段/第27天-评估检查点与复现-练习.py"
```

学习内容：

- `model.train()` 与 `model.eval()`
- `torch.no_grad()` 和 `torch.inference_mode()`
- `state_dict`、保存权重与保存完整对象的区别
- 最佳验证指标检查点和断点续训检查点
- 随机种子、配置记录和可复现边界

检查点至少包含：

~~~python
checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch": epoch,
    "best_val_metric": best_val_metric,
    "config": config,
}
~~~

实践任务：

1. 写 `evaluate` 函数，返回平均损失、预测和标签。
2. 保存最佳验证模型，重新创建模型后加载并验证输出一致。
3. 模拟中断，恢复优化器和下一轮训练。
4. 记录 Python、PyTorch 版本、设备和随机种子。
5. 用同一配置运行两次并比较曲线，同时说明结果不完全一致的可能原因。

## 第 28 天：第一周项目——FashionMNIST MLP

详细项目指南与练习骨架：

- [第28天-FashionMNIST-MLP项目.md](./第28天-FashionMNIST-MLP项目.md)
- [第28天-FashionMNIST-MLP项目-练习.py](./第28天-FashionMNIST-MLP项目-练习.py)

```bash
python "第二阶段/第28天-FashionMNIST-MLP项目-练习.py"
```

使用 FashionMNIST 完成一个不含卷积的 MLP 分类器。这个项目的目的不是追求图像任务高分，而是验证最小 PyTorch 训练闭环。

项目要求：

- 使用 `Dataset`、`DataLoader` 和 `nn.Module`
- 明确划分训练与验证数据，测试集只在最后使用
- 至少实现训练、验证、测试和预测四个入口
- 输出每轮平均损失与准确率
- 保存并重新加载最佳验证模型
- 绘制训练/验证损失和准确率曲线
- 对 16～64 个样本完成过拟合测试
- 至少为 shape、评估统计和检查点加载编写 5 个 pytest 测试

验收标准：在新进程中加载检查点即可评估；测试指标不是从最佳测试轮次挑选出来的。

---

# 第二周：训练工程、正则化与系统调试

## 第 29 天：重构可复用训练框架

学习内容：

- 将数据、模型、训练、评估和配置分离
- `train_one_epoch`、`evaluate` 和 `fit` 的职责
- 按样本累计 loss，避免 batch 大小导致统计偏差
- 使用 dataclass 或字典保存配置
- 使用返回值代替隐式全局状态

实践任务：

- 把第一周项目拆分成 `data.py`、`model.py`、`engine.py` 和 `train.py`。
- 让训练代码能通过配置切换隐藏层大小、优化器和学习率。
- 增加命令行参数、输出目录和运行摘要。
- 对关键纯函数编写单元测试。

验收标准：更换模型时不需要复制整套训练循环。

## 第 30 天：优化器与学习率

学习内容：

- SGD、Momentum、Adam 和 AdamW 的更新直觉
- 学习率过大、过小与损失震荡
- 学习率调度器的调用时机
- 固定训练预算下的公平比较
- batch size 与有效更新步数

实践任务：

1. 在相同初始化和数据顺序下比较 SGD、SGD + Momentum 和 AdamW。
2. 对每个优化器尝试至少 3 个学习率。
3. 同时画出横轴为 epoch 和 optimizer step 的曲线。
4. 记录最佳验证轮次，而不是只报告最后一轮。
5. 解释为什么不能从一次随机运行得出绝对结论。

## 第 31 天：初始化、梯度与数值诊断

学习内容：

- Xavier 与 He 初始化的适用场景
- 梯度消失、梯度爆炸和激活饱和
- 参数范数、梯度范数和激活分布
- `clip_grad_norm_` 的作用与局限
- NaN/Inf 的常见来源

实践任务：

1. 比较过大、过小和合理初始化。
2. 为每层记录参数范数和梯度范数。
3. 注入一个会产生 NaN 的错误并定位最早出现异常的位置。
4. 使用 `torch.isfinite` 给损失和关键张量增加断言。
5. 说明梯度裁剪为什么不能修复错误标签或错误损失。

## 第 32 天：权重衰减与 Dropout

学习内容：

- 训练误差与泛化误差
- L2 正则化与优化器中的 weight decay
- Dropout 的训练和评估行为
- 模型容量、正则化强度和数据量
- 早停的正确数据边界

实践任务：

1. 构造一个容易过拟合的小训练集。
2. 比较无正则、weight decay、Dropout 和两者组合。
3. 验证 `train()` 下两次前向输出可能不同，`eval()` 下应稳定。
4. 只根据验证集选择超参数。
5. 报告正则化前后的训练—验证差距。

## 第 33 天：归一化

学习内容：

- 特征标准化与网络内部归一化的区别
- BatchNorm 的 batch 统计量与运行统计量
- LayerNorm 对最后若干维度归一化
- `train()`/`eval()` 对 BatchNorm 的影响
- 为什么 Transformer 更常见 LayerNorm

实践任务：

1. 手工计算一个小 Tensor 的 LayerNorm，并与 `nn.LayerNorm` 对齐。
2. 比较 MLP 加 BatchNorm、LayerNorm 和不归一化。
3. 改变 batch size，观察 BatchNorm 的稳定性。
4. 检查归一化层可学习参数的 shape。

验收标准：能根据输入 shape 解释归一化统计发生在哪些维度。

## 第 34 天：深度学习调试与测试

学习内容：

- 小数据过拟合测试
- shape、dtype、数值范围和标签范围断言
- 基线、随机标签和打乱标签测试
- 梯度是否存在、是否有限、是否更新
- 数据泄漏、重复样本和划分重叠检查

推荐调试顺序：

1. 检查一个原始样本及标签。
2. 检查一个 batch 的 shape、dtype、device 和范围。
3. 检查前向输出、损失和梯度是否有限。
4. 尝试过拟合一个小 batch。
5. 再扩大数据并检查验证流程。
6. 最后才调整模型大小和训练时长。

实践任务：

- 为数据划分无重叠、模型输出 shape、损失有限、参数更新和评估确定性编写测试。
- 故意引入五类错误，并为每类错误保留最小复现与修复记录。
- 确认模型参数更新前后的 `state_dict` 确实不同。

## 第 35 天：第二周项目——可靠训练模板

将 FashionMNIST MLP 项目整理成后续两个文本项目都能复用的训练模板。

核心交付：

- 配置驱动的模型和优化器创建
- 训练、验证、早停、最佳检查点和断点续训
- CPU/CUDA/MPS 设备选择
- 随机种子和运行环境记录
- CSV 或 JSON 格式的 epoch 指标
- 损失、指标和学习率曲线
- 小 batch 过拟合命令
- 不少于 10 个自动化测试
- README 中写明运行命令、结果、限制和排错清单

非目标：

- 不需要分布式训练
- 不需要混合精度
- 不需要第三方实验追踪平台
- 不需要抽象成通用深度学习框架

---

# 第三周：文本分类器

## 第 36 天：文本分类任务与数据协议

推荐项目数据为 [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)，也可以使用来源明确、许可允许的本地小型文本分类数据。

学习内容：

- 文本、标签、类别分布和重复样本
- 分层划分训练/验证/测试集
- 多数类基线和简单关键词基线
- 只从训练集学习词表和预处理规则
- 类别不平衡下的准确率局限

实践任务：

1. 输出数据量、类别比例、文本长度分布和重复情况。
2. 固定随机种子完成分层划分，并验证集合无重叠。
3. 建立多数类基线。
4. 保存划分索引或可复现的划分配置。
5. 明确原始数据、处理中间数据和模型输入的边界。

## 第 37 天：Tokenization、词表与数值化

学习内容：

- 文本规范化与分词规则
- 训练集词表、最小词频和未知词
- `<pad>` 与 `<unk>` 的不同职责
- 序列截断、填充和长度分布
- 词表泄漏与数据预处理测试

实践任务：

1. 实现一个可测试的基础 tokenizer。
2. 只在训练文本上构建 `token_to_id`。
3. 实现 encode/decode，并测试未知词。
4. 统计不同最大长度下的截断比例。
5. 把词表与配置保存为 JSON。

验收标准：验证集或测试集出现的新词必须映射到 `<unk>`，不能重新扩充词表。

## 第 38 天：变长序列、collate_fn 与 Embedding

学习内容：

- 自定义 `collate_fn`
- batch 内动态 padding
- attention mask 在此阶段作为“有效 token mask”的含义
- `nn.Embedding` 的输入、输出和 padding index
- masked mean pooling

核心 shape：

~~~text
token_ids:  (batch, sequence)
mask:       (batch, sequence)
embeddings: (batch, sequence, embedding_dim)
pooled:     (batch, embedding_dim)
logits:     (batch, num_classes)
~~~

实践任务：

1. 实现 batch 内动态 padding。
2. 返回 `token_ids`、`mask`、`lengths` 和 `labels`。
3. 实现 masked mean pooling，确保 pad 不参与平均。
4. 用一个手算样例验证池化结果。
5. 测试全长、短序列、未知词和 batch size 为 1 的情况。

## 第 39 天：Embedding 与池化文本分类器

先实现简单可靠的模型：

`Token IDs → Embedding → Masked Mean Pooling → Linear → Logits`

学习内容：

- 稠密 one-hot 与 Embedding 查表
- 词袋模型的表达能力与顺序信息缺失
- logits 与二分类/多分类损失
- 参数量计算

实践任务：

1. 实现 `MeanPoolTextClassifier`。
2. 打印每一步的 shape。
3. 在 32 个样本上过拟合。
4. 与多数类基线比较。
5. 检查预测时是否使用 `eval()` 和 `inference_mode()`。

## 第 40 天：训练、指标与阈值

学习内容：

- 准确率、精确率、召回率、F1 和混淆矩阵
- macro、micro 和 weighted 平均
- 分类阈值与错误代价
- 验证集选阈值、测试集最终报告
- 概率校准只做概念了解

实践任务：

1. 每轮记录训练/验证损失和 F1。
2. 根据任务目标选择主指标，并说明理由。
3. 在验证集比较不同阈值。
4. 最终只在测试集运行一次固定方案。
5. 输出混淆矩阵和典型假阳性/假阴性。

## 第 41 天：正则化、消融与错误分析

学习内容：

- 词表大小、embedding 维度和最大长度
- Dropout、weight decay 和早停
- 消融实验与单变量比较
- 数据问题、模型问题和指标问题
- 词袋模型无法利用词序的结构性限制

至少完成下面三组实验：

- 不同最小词频
- 不同 embedding 维度
- 有无 Dropout 或 weight decay

每组实验固定其他条件，报告参数量、最佳验证指标、最佳轮次和训练时间。

错误分析至少覆盖：

- 未登录词过多
- 过度截断
- 拼写变化或缩写
- 否定词与词序
- 标签疑似错误或文本重复

## 第 42 天：第三周项目——文本分类器交付

核心交付：

- 从原始数据到训练输入的可复现流水线
- 训练集构建的 tokenizer 和词表
- 自定义 Dataset、collate_fn 和动态 padding
- Embedding + masked mean pooling 分类器
- 多数类基线与至少 3 组消融
- 最佳验证检查点和一次最终测试评估
- 混淆矩阵、典型错误和局限
- 单样本与批量预测命令
- 至少 12 个覆盖数据、模型和评估的测试

验收标准：

- pad token 不影响池化结果
- 测试数据没有参与建词表、调参、选轮次或选阈值
- 新进程可加载模型、词表和配置并得到一致预测
- README 能让他人在新虚拟环境中复现实验

---

# 第四周：字符级语言模型

## 第 43 天：语言模型目标与字符词表

使用来源明确的本地纯文本或公共领域小语料。先从几百 KB 到几 MB 的数据开始，不追求大规模训练。

学习内容：

- 自回归语言模型与下一个 token 预测
- 字符级 tokenization 的优点和局限
- BOS/EOS/UNK 是否需要由任务决定
- 输入序列与目标序列错一位对齐
- 训练/验证/测试按连续文本切分

示例：

~~~text
文本:   hello
输入:   hell
目标:   ello
~~~

实践任务：

1. 构建字符到 ID、ID 到字符的映射。
2. 实现 encode/decode 往返测试。
3. 按连续区间切分语料，避免随机窗口跨集合泄漏。
4. 统计字符频率和未知字符策略。
5. 实现字符频率基线或 bigram 基线。

## 第 44 天：序列窗口与语言模型 DataLoader

学习内容：

- 固定长度窗口和 stride
- 输入 `x` 与标签 `y` 的逐位置对齐
- batch-first 形状
- 随机窗口与顺序批次
- 短语料和边界条件

核心 shape：

~~~text
x:      (batch, sequence)
y:      (batch, sequence)
logits: (batch, sequence, vocab_size)
~~~

实践任务：

1. 实现 `CharLanguageModelDataset`。
2. 检查任意样本是否满足 `y[t] == x[t + 1]`。
3. 验证最后一个合法窗口没有越界。
4. 用很短的可手算字符串编写测试。
5. 比较不同 sequence length 的样本数量与训练成本。

## 第 45 天：RNN/GRU 与隐藏状态

学习内容：

- 循环神经网络的时间步与隐藏状态
- RNN 的梯度问题和 GRU 的门控直觉
- `nn.GRU` 的输入输出 shape
- 单层/多层、单向/双向与 dropout 参数
- Embedding、GRU 和输出投影

模型结构：

`Character IDs → Embedding → GRU → Linear → Logits`

实践任务：

1. 实现 `CharGRULanguageModel`。
2. 打印 embedding、GRU 输出、hidden state 和 logits 的 shape。
3. 测试不同 batch size 与 sequence length。
4. 计算模型参数量。
5. 在一个极小字符串上过拟合，确认能重复生成该模式。

## 第 46 天：序列损失、反向传播与梯度裁剪

学习内容：

- 将 `(B, T, V)` logits 和 `(B, T)` 标签展平计算交叉熵
- 截断反向传播的直觉
- 隐藏状态何时重置、传递和 `detach`
- 梯度裁剪与梯度爆炸
- token 平均损失与困惑度

实践任务：

1. 正确对齐 logits 与 targets。
2. 记录裁剪前后的梯度范数。
3. 对比每个 batch 重置 hidden 与跨 batch 传递 hidden。
4. 检查训练和验证 token 数是否正确。
5. 用 `exp(loss)` 计算困惑度时处理过大数值并解释其含义。

## 第 47 天：文本生成与采样

学习内容：

- 自回归生成循环
- greedy、随机采样、temperature 和 top-k
- `softmax(logits / temperature)`
- 随机种子与生成多样性
- 生成质量不能只靠主观挑选

实践任务：

1. 实现给定 prompt 的逐字符生成。
2. 比较 temperature 为 0.5、1.0 和 1.5 的结果。
3. 比较 greedy、随机采样和 top-k。
4. 处理空 prompt、未知字符和最大生成长度。
5. 保存固定 prompts，作为不同检查点的统一定性评测集。

## 第 48 天：实验、消融与错误分析

至少比较：

- 字符频率或 bigram 基线与 GRU
- 不同 sequence length
- 不同 hidden size
- 有无梯度裁剪

每组实验记录：

- 参数量
- 训练 token 数
- 最佳验证损失/困惑度
- 最佳轮次
- 训练时间和设备
- 固定 prompts 的生成样例

错误分析重点：

- 重复循环
- 长期一致性差
- 标点和空白异常
- 训练文本记忆
- 温度过低导致单调、过高导致混乱

## 第 49 天：第四周项目——字符级语言模型与阶段整理

核心交付：

- 可复现的语料读取、连续切分和字符词表
- 正确对齐的序列 Dataset 与 DataLoader
- bigram 基线
- Embedding + GRU + Linear 字符语言模型
- 训练、验证、梯度裁剪、最佳检查点和断点续训
- 困惑度曲线与至少 4 组对比实验
- 支持 prompt、temperature、top-k 和长度的生成脚本
- 固定 prompts 的生成对比
- 不少于 15 个自动化测试
- README 中写清数据来源、运行方式、结果、失败模式和限制

验收标准：

- 模型能在极小语料上过拟合
- 训练损失和验证损失的统计按 token 数加权
- 加载检查点后验证指标与生成行为符合预期
- 最终测试集没有参与模型或超参数选择
- 能解释 GRU 的隐藏状态与下一阶段 Transformer 上下文表示的差异

---

# 阶段项目结构建议

~~~text
第二阶段/
├── loadmap.md
├── week1_pytorch_basics/
│   ├── README.md
│   ├── tensors.py
│   ├── autograd_demo.py
│   ├── model.py
│   └── tests/
├── week2_training_template/
│   ├── README.md
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── engine.py
│   ├── checkpoint.py
│   ├── train.py
│   └── tests/
├── week3_text_classification/
│   ├── README.md
│   ├── data/
│   ├── artifacts/
│   │   ├── vocab.json
│   │   └── config.json
│   ├── dataset.py
│   ├── model.py
│   ├── engine.py
│   ├── train.py
│   ├── predict.py
│   └── tests/
└── week4_char_lm/
    ├── README.md
    ├── data/
    ├── artifacts/
    ├── dataset.py
    ├── bigram.py
    ├── model.py
    ├── engine.py
    ├── train.py
    ├── generate.py
    └── tests/
~~~

建议把大型数据和模型权重放在被 Git 忽略的目录中，只提交下载说明、小型样例、配置、代码、测试、图表和指标摘要。

# 阶段验收标准

## 知识验收

完成第二阶段时，应当能够独立回答：

- Tensor 与 NumPy 数组的主要相同点和差异是什么？
- `requires_grad`、叶子张量和 `grad_fn` 分别表示什么？
- 为什么每个训练 step 都要清理梯度？
- `model.train()` 与 `model.eval()` 会影响哪些常见层？
- 为什么分类模型通常输出 logits，而不是先输出 Softmax 概率？
- Dataset 与 DataLoader 的职责有什么区别？
- SGD、Momentum、AdamW 和学习率分别如何影响训练？
- weight decay、Dropout 和早停分别解决什么问题？
- BatchNorm 与 LayerNorm 的归一化维度和运行行为有什么区别？
- 为什么必须先做小 batch 过拟合测试？
- 文本词表为什么只能用训练集构建？
- padding 为什么不能参与 mean pooling？
- 语言模型的输入与目标为什么要错开一个 token？
- GRU hidden state 的 shape 由哪些因素决定？
- temperature 和 top-k 如何影响生成结果？
- 检查点为了断点续训至少应保存哪些内容？

## 代码验收

- 所有 pytest 测试通过。
- 项目能在新的虚拟环境中按 README 重新安装和运行。
- 同一套训练引擎能够支持至少两个模型。
- CPU 可以跑通 smoke test；有加速设备时能正确迁移模型与数据。
- 每个模型都能过拟合一个极小 batch。
- 训练、验证、测试统计按样本数或 token 数正确加权。
- 最佳模型只根据验证集选择，测试集只用于最终报告。
- 保存并恢复模型后，固定输入的推理结果一致或在合理容差内一致。
- 文本分类项目保存词表、标签映射、配置和模型权重。
- 字符语言模型保存字符映射、配置、模型权重和固定生成 prompts。
- 实验报告包含基线、曲线、对比实验、错误分析和局限。
- Git 历史能体现实现、测试、实验与修复过程。

## 常见失败与排查

| 现象 | 优先检查 |
| --- | --- |
| Loss 完全不下降 | 标签、损失输入、参数是否在优化器中、梯度是否存在、学习率 |
| Loss 变成 NaN | 输入范围、除零、log、学习率、梯度范数、有限性断言 |
| 训练很好但验证很差 | 划分、重复数据、模型容量、正则化、预处理泄漏 |
| 验证结果每次变化 | 是否调用 `eval()`、Dropout/BatchNorm、数据顺序、随机种子 |
| CPU 能跑但 GPU/MPS 报错 | 模型、输入、标签和新建 Tensor 的 device |
| 分类损失异常 | 是否把概率传给 CrossEntropy、标签 dtype/范围、输出类别数 |
| 文本池化结果异常 | pad mask、长度为零、padding index、求和维度 |
| 字符模型不学习 | x/y 对齐、词表大小、hidden state、梯度裁剪、序列长度 |
| 断点续训后曲线突变 | 优化器、调度器、轮次、随机状态和配置是否恢复 |

# 3 周压缩方案

如果只能投入 3 周，按下面方式压缩：

1. 第 22～27 天合并为 4 天，但必须保留自动微分、最小训练循环、DataLoader 和检查点。
2. 第 29～34 天合并为 3 天，优先保留训练模板、正则化和调试测试。
3. 文本分类器和字符级语言模型各保留完整 7 天项目周期。
4. FashionMNIST 只做 smoke test，不做大量调参。
5. 不压缩数据划分、基线、小 batch 过拟合、最终测试和错误分析。

# 进入第三阶段的判断

满足以下条件后再开始 Transformer：

1. 能从空文件独立写出 Dataset、DataLoader、nn.Module、训练和评估循环。
2. 能根据代码推断 `(batch, sequence, hidden)` 各维的含义。
3. 能解释自动微分如何把 loss 的梯度传回 Embedding 和 Linear 参数。
4. 遇到 Loss 不下降时，能按数据、shape、损失、梯度、学习率和模式切换逐项排查。
5. 能正确处理变长文本、padding mask 和 masked pooling。
6. 能实现并评估一个字符级自回归语言模型。
7. 能保存、恢复和复现实验，而不是只保留一次成功运行的 Notebook。
8. 能区分优化问题、泛化问题、数据问题和模型表达能力限制。

完成以上内容后，进入第三阶段：Tokenization、Embedding、Self-Attention、Multi-Head Attention、位置编码、Causal Mask、残差连接、LayerNorm 和 Mini GPT。

# 第 28 天：FashionMNIST MLP 项目

## 项目目标

本项目把第 22～27 天的内容组合成第一个完整 PyTorch 工程。任务是使用不含卷积的多层感知机，对 FashionMNIST 服饰图片进行 10 类分类。

项目重点不是追求图像任务的最高准确率，而是证明你能独立完成：

- Tensor shape、dtype 和 device 管理
- Dataset、Subset 和 DataLoader
- `nn.Module` 模型
- CrossEntropyLoss 和优化器
- 训练、验证、测试与推理
- 小 batch 过拟合
- 最佳验证检查点和断点续训
- 曲线、混淆矩阵和错误分析
- 自动化测试和 README

运行配套练习：

~~~bash
python "第二阶段/第28天-FashionMNIST-MLP项目-练习.py"
~~~

配套练习使用合成数据检查项目骨架，不会自动下载 FashionMNIST。完成练习后，再按本文创建正式项目目录。

## 1. 时间安排

如果完整项目无法在一天内完成，可以拆成两天：

### 第一天

1. 30 分钟：创建目录、配置和数据加载。
2. 30 分钟：实现模型与 shape 测试。
3. 45 分钟：实现训练和验证循环。
4. 30 分钟：小 batch 过拟合。
5. 30 分钟：完成一次短训练。

### 第二天

1. 30 分钟：最佳检查点与恢复。
2. 30 分钟：正式训练和最终测试。
3. 40 分钟：曲线、混淆矩阵和错误分析。
4. 40 分钟：pytest 与 README。
5. 20 分钟：从新进程复现。

## 2. 环境与依赖

~~~bash
cd /Users/chenchao/Desktop/LLM
source .venv/bin/activate
python -m pip install torch torchvision matplotlib pytest
~~~

验证：

~~~bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
~~~

如果安装失败，使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)生成与当前系统和设备匹配的命令。

## 3. 项目目录

~~~text
第二阶段/
└── week1_pytorch_basics/
    ├── __init__.py
    ├── README.md
    ├── config.py
    ├── data.py
    ├── model.py
    ├── engine.py
    ├── checkpoint.py
    ├── train.py
    ├── evaluate.py
    ├── predict.py
    ├── artifacts/
    │   ├── config.json
    │   ├── metrics.json
    │   ├── history.json
    │   ├── best_model.pt
    │   └── figures/
    └── tests/
        ├── test_data.py
        ├── test_model.py
        ├── test_engine.py
        └── test_checkpoint.py
~~~

数据下载目录可以是：

~~~text
第二阶段/data/fashion_mnist/
~~~

建议在 `.gitignore` 中忽略：

~~~gitignore
第二阶段/data/
第二阶段/**/artifacts/*.pt
第二阶段/**/__pycache__/
~~~

不要忽略小型 JSON 配置、指标、测试和最终图表。

## 4. 数据集

FashionMNIST 包含 28×28 灰度服饰图片，共 10 个类别。TorchVision 提供官方训练集和测试集对象。

类别：

~~~python
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
~~~

输入经过 `ToTensor` 后：

~~~text
单样本图片： (1, 28, 28)，float32
单样本标签： ()，long
一个 batch： (batch, 1, 28, 28)
~~~

MLP 会把图片展平：

~~~text
(batch, 1, 28, 28) → (batch, 784)
~~~

## 5. 数据划分协议

不要使用官方测试集调参。

推荐：

~~~text
官方训练集 60,000
├── 训练集 50,000
└── 验证集 10,000

官方测试集 10,000
└── 最终测试，训练过程中不查看
~~~

代码：

~~~python
from torch.utils.data import random_split
from torchvision import datasets, transforms

transform = transforms.ToTensor()

full_train = datasets.FashionMNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform,
)

test_set = datasets.FashionMNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=transform,
)

generator = torch.Generator().manual_seed(seed)
train_set, valid_set = random_split(
    full_train,
    lengths=[50_000, 10_000],
    generator=generator,
)
~~~

保存或记录：

- 划分种子
- 训练/验证数量
- TorchVision 数据集版本
- 使用的 transform

如果以后增加随机数据增强，需要让训练集和验证集使用不同 transform；两个 Subset 共享同一个底层 Dataset 时要特别小心。

## 6. DataLoader

~~~python
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)

valid_loader = DataLoader(
    valid_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)
~~~

先使用 `num_workers=0`。确认功能正确后，再测量增加 worker 是否真的更快。

数据验收：

~~~python
images, labels = next(iter(train_loader))

assert images.ndim == 4
assert images.shape[1:] == (1, 28, 28)
assert images.dtype == torch.float32
assert labels.ndim == 1
assert labels.dtype == torch.long
assert labels.min() >= 0
assert labels.max() < 10
~~~

至少显示一个 batch 中的 16 张图片和标签，人工确认没有错位。

## 7. 配置

建议使用 dataclass：

~~~python
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 42
    batch_size: int = 128
    hidden_dim: int = 128
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    num_workers: int = 0
~~~

保存为 JSON：

~~~python
import json

with config_path.open("w", encoding="utf-8") as file:
    json.dump(asdict(config), file, ensure_ascii=False, indent=2)
~~~

不要把关键超参数散落在不同文件的魔法数字中。

## 8. 模型

基线 MLP：

~~~python
class FashionMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)
~~~

shape：

~~~text
images   (B, 1, 28, 28)
flatten  (B, 784)
hidden   (B, hidden_dim)
logits   (B, 10)
~~~

最后不做 Softmax。

参数量：

~~~text
784×hidden_dim + hidden_dim
+ hidden_dim×10 + 10
~~~

当 `hidden_dim=128` 时总参数量为 101,770。

模型测试：

~~~python
model = FashionMLP(hidden_dim=128)

for batch_size in [1, 8, 31]:
    images = torch.randn(batch_size, 1, 28, 28)
    logits = model(images)
    assert logits.shape == (batch_size, 10)
    assert torch.isfinite(logits).all()
~~~

## 9. 基线

FashionMNIST 类别基本平衡，多数类或均匀随机分类器准确率大约在 10% 附近。项目报告至少给出：

- 10 类随机基线的理论准确率
- 训练集多数类基线的实际准确率
- MLP 的验证和测试准确率

基线的目的不是凑一个很低的数字，而是确认模型确实学到了数据与标签的关系。

## 10. 训练组件

~~~python
model = FashionMLP(
    hidden_dim=config.hidden_dim,
    dropout=config.dropout,
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)
~~~

训练 step：

~~~python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
~~~

训练指标：

~~~python
predictions = logits.argmax(dim=1)
correct = (predictions == labels).sum().item()
~~~

按样本数聚合：

~~~python
total_loss += loss.item() * batch_size
total_correct += correct
total_samples += batch_size
~~~

## 11. 训练函数接口

推荐：

~~~python
def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
) -> dict[str, float]:
    ...
~~~

返回：

~~~python
{
    "loss": epoch_loss,
    "accuracy": epoch_accuracy,
}
~~~

验证：

~~~python
def evaluate(
    model,
    data_loader,
    criterion,
    device,
) -> dict[str, float]:
    ...
~~~

保持两个函数返回结构一致，方便记录：

~~~python
history.append({
    "epoch": epoch,
    "train_loss": train_metrics["loss"],
    "train_accuracy": train_metrics["accuracy"],
    "valid_loss": valid_metrics["loss"],
    "valid_accuracy": valid_metrics["accuracy"],
})
~~~

## 12. 必做：小 batch 过拟合

从训练集固定取 32 个样本：

~~~python
tiny_images, tiny_labels = next(iter(
    DataLoader(train_set, batch_size=32, shuffle=False)
))
~~~

在同一 batch 上重复训练：

~~~python
for step in range(500):
    ...
~~~

目标：

- 损失显著下降
- 训练准确率接近 100%

如果做不到，不要直接开始完整训练。优先检查：

- 输入是否是 float32
- 标签是否是 long 且范围 0～9
- logits 是否为 `(32, 10)`
- 是否错误地做了 Softmax
- optimizer 是否拿到模型参数
- 是否执行 `zero_grad/backward/step`
- Dropout 是否过大
- 学习率是否不合适

完成后要重新创建正式模型和优化器，不能用已经记住小 batch 的参数开始正式实验。

## 13. Smoke test

完整训练前先做：

- 只用 2 个 batch
- 只训练 1 个 epoch
- 在 CPU 上运行
- 确认能保存检查点
- 确认能加载并评估

Smoke test 的目标是几分钟内发现工程错误，而不是得到好指标。

建议命令：

~~~bash
python -m week1_pytorch_basics.train --epochs 1 --max-train-batches 2
~~~

## 14. 完整训练

推荐起点：

~~~text
batch_size:    128
hidden_dim:    128
dropout:       0.2
optimizer:     AdamW
learning_rate: 1e-3
weight_decay:  1e-4
epochs:        10
seed:          42
~~~

这只是起点，不是保证最优的答案。

每轮：

1. 训练一轮。
2. 验证一轮。
3. 记录指标。
4. 如果验证损失改善，保存最佳检查点。
5. 打印学习率、用时和设备。

不要根据测试集决定 hidden size、Dropout、学习率或 epoch。

## 15. 检查点

~~~python
checkpoint = {
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch": epoch,
    "best_val_loss": best_val_loss,
    "config": asdict(config),
    "class_names": CLASS_NAMES,
}
~~~

保存：

~~~python
torch.save(checkpoint, checkpoint_path)
~~~

正式测试前：

1. 重新创建模型。
2. 加载最佳验证检查点。
3. 调用 `eval()`。
4. 对测试集运行一次。

## 16. 测试集评估

报告：

- 测试平均损失
- 测试准确率
- 每类准确率
- 10×10 混淆矩阵
- 至少 20 个典型错误样本

混淆矩阵可以用 scikit-learn：

~~~python
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(targets, predictions, labels=list(range(10)))
~~~

每类准确率：

~~~text
class_accuracy[c] =
    真实类别为 c 且预测正确的数量
    /
    真实类别为 c 的数量
~~~

观察容易混淆的类别，例如 Shirt、T-shirt/top、Pullover、Coat。不要只写“模型分不清”，要结合图像边界、纹理和类别相似性描述。

## 17. 曲线

至少绘制：

1. 训练损失与验证损失
2. 训练准确率与验证准确率

图表要求：

- 标题明确
- 横轴为 epoch
- 纵轴标明指标
- 训练与验证有图例
- 从 JSON 记录生成，而不是训练时临时手填

分析问题：

- 训练损失是否持续下降？
- 验证损失何时达到最好？
- 后期是否出现训练继续变好但验证变差？
- 最佳验证 epoch 与最后 epoch 是否相同？

## 18. 对比实验

至少完成两组单变量对比：

### 隐藏层大小

~~~text
hidden_dim ∈ {64, 128, 256}
~~~

### 正则化

~~~text
dropout ∈ {0.0, 0.2, 0.5}
~~~

固定其他配置，记录：

| hidden_dim | dropout | 参数量 | 最佳验证损失 | 最佳验证准确率 | 最佳 epoch | 时间 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.2 | TODO | TODO | TODO | TODO | TODO |
| 128 | 0.2 | TODO | TODO | TODO | TODO | TODO |
| 256 | 0.2 | TODO | TODO | TODO | TODO | TODO |

如果时间有限，先完成可靠单模型流程，再做对比；不要为了多跑实验省略测试与复现。

## 19. 自动化测试

至少 10 个测试：

### 数据

- 训练、验证、测试数量正确
- 训练与验证索引无重叠
- 图片 shape、dtype、数值有限
- 标签 dtype 和范围正确
- 验证/测试 DataLoader 覆盖全部样本

### 模型

- batch size 1、8、31 的输出 shape 正确
- logits 全部有限
- 参数数量符合公式
- 模型可以移动到选定 device

### 训练与评估

- 一个 train step 后至少一个参数变化
- 评估不产生梯度
- 评估模式下输出稳定
- epoch loss 按样本数加权
- 小 batch 可以过拟合

### 检查点

- 保存与加载后固定输入 logits 一致
- 优化器状态恢复
- 下一轮编号正确

## 20. README

README 必须包含：

1. 项目目标
2. 数据来源和划分
3. 环境与安装命令
4. 目录结构
5. 训练命令
6. Smoke test 命令
7. 小 batch 过拟合命令
8. 评估与预测命令
9. 模型结构和参数量
10. 最终配置
11. 基线、验证和测试结果
12. 曲线和混淆矩阵
13. 错误分析
14. 已知限制
15. 复现说明

## 21. 推荐命令接口

~~~bash
# 从第二阶段目录运行包命令
cd "/Users/chenchao/Desktop/LLM/第二阶段"

# 自动测试
pytest -q "week1_pytorch_basics/tests"

# 小 batch 过拟合
python -m week1_pytorch_basics.train --overfit-small-batch

# Smoke test
python -m week1_pytorch_basics.train --epochs 1 --max-train-batches 2

# 正式训练
python -m week1_pytorch_basics.train --config configs/baseline.json

# 最终测试
python -m week1_pytorch_basics.evaluate --checkpoint artifacts/best_model.pt --split test

# 单张图片预测
python -m week1_pytorch_basics.predict --checkpoint artifacts/best_model.pt --index 0
~~~

## 22. Git 提交建议

至少产生这些有意义的提交：

~~~text
feat: add reproducible FashionMNIST data split
feat: implement FashionMLP and shape tests
feat: add training and evaluation loops
test: add tiny-batch overfit check
feat: save and resume best checkpoint
docs: report experiments and error analysis
~~~

不要等项目全部完成后只提交一次。

## 23. 验收标准

### 必须完成

- [ ] 训练、验证、测试集合边界正确
- [ ] 模型返回 `(batch, 10)` logits
- [ ] CrossEntropyLoss 直接接收 logits
- [ ] 训练与验证指标按样本数加权
- [ ] 小 batch 能过拟合
- [ ] 保存最佳验证检查点
- [ ] 新进程能加载检查点
- [ ] 只在方案冻结后评估测试集
- [ ] 至少 10 个自动化测试
- [ ] README 可复现

### 结果质量

- [ ] 模型明显超过约 10% 的简单基线
- [ ] 训练/验证曲线合理
- [ ] 至少完成 2 组单变量对比
- [ ] 给出混淆矩阵和典型错误
- [ ] 解释 MLP 忽略二维局部结构的限制

准确率目标只能作为参考，不能替代流程正确性。不同版本、设备、随机种子和训练预算会造成结果差异。

## 24. 项目复盘问题

完成后独立回答：

1. 为什么官方测试集不能用来选择最佳 epoch？
2. 为什么模型最后一层不做 Softmax？
3. 为什么要先做小 batch 过拟合？
4. 为什么训练和验证 loss 不能简单平均每个 batch？
5. Dropout 在训练和评估时有什么不同？
6. 断点续训为什么要保存 optimizer state？
7. MLP 为什么丢失图片二维空间结构？
8. 如果训练准确率高而验证准确率低，下一步如何验证原因？
9. 如果 loss 完全不下降，应按什么顺序检查？
10. 如何证明重新加载的检查点与保存时模型一致？

## 25. 完成后的衔接

完成本项目后，应该已经具备一个可靠的 PyTorch 最小闭环。第二周会在这个模板上增加：

- 配置驱动的训练引擎
- 初始化、梯度范数和 NaN 诊断
- SGD、Momentum、AdamW 与学习率比较
- weight decay、Dropout、BatchNorm 与 LayerNorm
- 更系统的测试与错误注入

不要急着把项目改成 CNN。本阶段主线是训练工程与文本模型；图像任务只用于快速验证 PyTorch 基础。

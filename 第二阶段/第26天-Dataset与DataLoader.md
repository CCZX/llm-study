# 第 26 天：Dataset 与 DataLoader

## 今日目标

完成本节后，应当能够：

- 解释 `Dataset` 与 `DataLoader` 的职责边界
- 使用 `TensorDataset` 封装内存中的 Tensor
- 实现自定义 map-style Dataset
- 使用 batch、shuffle 和 `drop_last`
- 固定随机生成器以复现数据划分和 batch 顺序
- 正确创建训练、验证和测试 DataLoader
- 检查最后一个 batch、标签 dtype 和样本覆盖
- 理解 `collate_fn` 与变长数据批处理的作用

## 1. 今日安排

建议总用时为 2～3 小时：

1. 30 分钟：Dataset 与 DataLoader 职责。
2. 35 分钟：TensorDataset 和自定义 Dataset。
3. 40 分钟：batch、shuffle、drop_last 与采样。
4. 30 分钟：数据划分与可复现性。
5. 30 分钟：完成练习和数据管线检查。

运行练习：

~~~bash
python "第二阶段/第26天-Dataset与DataLoader-练习.py"
~~~

## 2. 两个对象的职责

### Dataset

Dataset 负责：

- 数据有多少个样本
- 给定索引时如何返回一个样本
- 单样本级的读取与转换

### DataLoader

DataLoader 负责：

- 按什么顺序取索引
- 是否打乱
- 一次组合多少个样本
- 如何把样本拼成 batch
- 是否使用多个 worker 读取

可以记成：

~~~text
Dataset：如何取得一个样本
DataLoader：如何迭代许多样本并组成 batch
~~~

## 3. TensorDataset

如果所有数据已经在 Tensor 中：

~~~python
import torch
from torch.utils.data import DataLoader, TensorDataset

features = torch.randn(100, 4)
labels = torch.randint(0, 3, size=(100,))

dataset = TensorDataset(features, labels)
~~~

~~~python
print(len(dataset))       # 100
sample_x, sample_y = dataset[0]
print(sample_x.shape)     # (4,)
print(sample_y.shape)     # 标量 Tensor
~~~

`TensorDataset` 要求所有输入 Tensor 的第 0 维长度相同。

## 4. 自定义 map-style Dataset

~~~python
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        if len(features) != len(labels):
            raise ValueError("特征和标签数量必须一致")
        self.features = features.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]
~~~

重点：

- `__len__` 返回样本数
- `__getitem__` 返回单个样本，不是完整 batch
- 分类标签通常转换为 `long`
- 不要在 `__getitem__` 中做每次都重复的昂贵全局计算

可以返回元组、字典或自定义结构：

~~~python
def __getitem__(self, index):
    return {
        "features": self.features[index],
        "label": self.labels[index],
        "sample_id": index,
    }
~~~

DataLoader 会递归拼接常见容器中的 Tensor。

## 5. 创建 DataLoader

~~~python
train_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    drop_last=False,
    num_workers=0,
)
~~~

读取一个 batch：

~~~python
features, labels = next(iter(train_loader))

print(features.shape)  # (16, 4)
print(labels.shape)    # (16,)
~~~

### batch_size

`batch_size=16` 表示最多组合 16 个样本。数据量不是 16 的整数倍时，最后一批更小。

~~~text
样本数 100，batch_size 16
完整 batch 6 个，共 96 个样本
最后 batch 1 个，共 4 个样本
总 batch 数 7
~~~

### shuffle

训练集通常使用：

~~~python
shuffle=True
~~~

每个 epoch 重新打乱有助于减少固定顺序产生的偏差。

验证和测试通常：

~~~python
shuffle=False
~~~

指标本身一般与顺序无关，但稳定顺序更方便对应样本 ID、保存预测和错误分析。

### drop_last

`drop_last=True` 会丢弃不足一个 batch 的最后样本。

训练时在 BatchNorm 等特定场景可能有用，但默认不要丢数据。验证和测试必须特别谨慎，通常应设为 False，确保每个样本都被评估。

## 6. 一个 epoch 的迭代

~~~python
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)
~~~

每次重新对 DataLoader 进行 `for` 循环就是一个新的 epoch 迭代。

DataLoader 自身不负责：

- 把数据自动移到 GPU/MPS
- 调用模型
- 计算损失
- 反向传播

训练循环中仍需：

~~~python
inputs = inputs.to(device)
targets = targets.to(device)
~~~

通常 Dataset 保存 CPU 数据，batch 取出后才移动设备。把完整数据集预先放入 GPU 可能占满显存，也会让多 worker 加载变复杂。

## 7. 数据划分

先创建互不重叠的训练、验证、测试索引，再分别构造 Dataset 或 Subset。

~~~python
from torch.utils.data import random_split

generator = torch.Generator().manual_seed(42)
train_set, valid_set, test_set = random_split(
    dataset,
    lengths=[70, 15, 15],
    generator=generator,
)
~~~

`random_split` 按数量划分，不保证分类任务中每类比例完全一致。类别明显不平衡时，后续文本分类项目应使用分层划分。

划分原则：

- 数据划分只执行一次并可复现
- 同一原始样本不能跨集合重复
- 预处理统计和词表只从训练集拟合
- 测试集不用于选择模型、轮次或阈值

检查索引：

~~~python
train_indices = set(train_set.indices)
valid_indices = set(valid_set.indices)
test_indices = set(test_set.indices)

assert train_indices.isdisjoint(valid_indices)
assert train_indices.isdisjoint(test_indices)
assert valid_indices.isdisjoint(test_indices)
~~~

## 8. 可复现的 shuffle

~~~python
generator = torch.Generator()
generator.manual_seed(42)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    generator=generator,
)
~~~

两个独立 DataLoader 使用两个同种子 Generator，第一次迭代顺序应一致：

~~~python
g1 = torch.Generator().manual_seed(42)
g2 = torch.Generator().manual_seed(42)
~~~

不要让两个 DataLoader 共享同一个 Generator 后再期待它们顺序一致，因为第一个迭代会推进生成器状态。

完整复现还可能涉及：

- Python `random`
- NumPy 随机数
- PyTorch 随机数
- 多 worker 种子
- 加速设备的非确定性算法

## 9. num_workers

~~~python
DataLoader(dataset, num_workers=0)
~~~

`num_workers=0` 表示在主进程加载，最容易调试，适合学习和小数据。

增加 worker 可能提高 I/O 吞吐，但：

- 会创建额外进程
- Dataset 必须能被 worker 正确序列化
- Notebook 和不同操作系统行为可能不同
- 小数据可能反而更慢
- 错误堆栈更难读

建议：

1. 先用 `num_workers=0` 验证正确性。
2. 数据加载明显成为瓶颈时再测试 2、4 等值。
3. 用实际吞吐测量，不凭经验盲调。

## 10. collate_fn

默认 collate 会把形状相同的单样本 Tensor 沿第 0 维堆叠：

~~~text
单样本 feature: (4,)
batch feature:  (batch, 4)
~~~

变长文本不能直接堆叠：

~~~text
样本 A token_ids: (5,)
样本 B token_ids: (8,)
~~~

需要自定义 `collate_fn` 在 batch 内 padding：

~~~python
def collate_batch(samples):
    # 计算 batch 内最大长度
    # padding
    # 生成 mask
    # 返回 token_ids、mask、labels
    ...
~~~

第 38 天会完整实现动态 padding。今天只需理解：Dataset 返回自然长度的单样本，collate_fn 决定怎样组 batch。

## 11. Dataset 中的转换

单样本转换可以放在 `__getitem__`：

~~~python
class TransformDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.features[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[index]
~~~

需要区分：

- 随机数据增强：训练集可以每次不同
- 确定性预处理：训练、验证、测试使用同一已拟合规则
- 统计量拟合：只能使用训练集

不要在每次 `__getitem__` 中重新计算全数据均值和标准差。

## 12. 检查数据管线

### 12.1 检查一个原始样本

~~~python
x, y = dataset[0]
print(x, x.shape, x.dtype)
print(y, y.shape, y.dtype)
~~~

### 12.2 检查一个 batch

~~~python
inputs, targets = next(iter(train_loader))

assert inputs.ndim == 2
assert inputs.dtype == torch.float32
assert targets.ndim == 1
assert targets.dtype == torch.long
assert inputs.shape[0] == targets.shape[0]
~~~

### 12.3 检查样本覆盖

验证/测试 DataLoader 应满足：

~~~python
seen = sum(len(targets) for _, targets in valid_loader)
assert seen == len(valid_loader.dataset)
~~~

### 12.4 检查标签范围

~~~python
assert targets.min() >= 0
assert targets.max() < num_classes
~~~

## 13. WeightedRandomSampler 只做概念了解

类别不平衡时，可以通过采样器改变训练样本被抽到的概率：

~~~python
from torch.utils.data import WeightedRandomSampler
~~~

但要注意：

- 采样器改变的是训练分布
- 验证和测试不应使用平衡采样来伪装真实指标
- `shuffle` 与显式 sampler 通常不能同时设置
- 是否使用采样器应通过验证集比较

第三周文本分类项目先建立类别不平衡指标，再决定是否需要加权损失或采样。

## 14. 常见错误

### 14.1 Dataset 返回完整数据集

`__getitem__` 应返回一个样本。返回完整 Tensor 会让 DataLoader 多套一层 batch，并重复大量数据。

### 14.2 标签仍是字符串或 float

在 Dataset 或明确的预处理阶段建立标签映射，并根据损失要求转换 dtype。

### 14.3 验证集 shuffle 并非严重错误，但不利于追踪

固定顺序更容易让预测与 sample ID 对齐。

### 14.4 验证/测试使用 drop_last=True

会静默漏掉部分样本，指标和预测文件不完整。

### 14.5 Dataset 内把数据放到 GPU

多 worker、显存和序列化更复杂。基础项目在训练循环中移动 batch。

### 14.6 划分后又重新随机划分

不同实验的数据集合变化，比较不公平。保存索引或固定划分代码和种子。

## 15. 自测题

1. Dataset 和 DataLoader 各负责什么？
2. 为什么 `__getitem__` 返回单样本而不是 batch？
3. 100 个样本、batch size 16、`drop_last=False` 时有几个 batch？
4. 验证集为什么通常 `shuffle=False`、`drop_last=False`？
5. 两个 DataLoader 如何得到相同的第一次 shuffle 顺序？
6. 什么时候需要 `collate_fn`？
7. 为什么完整数据集通常不直接放在 GPU？

答案：

~~~text
1. Dataset 定义单样本访问；DataLoader 负责采样、组 batch 和迭代。
2. batch 策略属于 DataLoader，保持 Dataset 可复用。
3. 7 个，最后一个含 4 个样本。
4. 保证稳定顺序和完整评估所有样本。
5. 分别传入两个使用相同种子初始化的 Generator。
6. 默认堆叠无法处理变长或复杂样本结构时。
7. 容易占满显存，并使多 worker 和设备管理复杂。
~~~

## 16. 今日验收

- 能实现包含 `__len__` 与 `__getitem__` 的 Dataset
- 能解释训练与验证 DataLoader 配置差异
- 能预测最后一个 batch 的大小
- 能固定 shuffle 的 Generator
- 能创建互不重叠的数据划分
- 能检查样本覆盖和标签范围
- 能说明 `collate_fn` 的作用
- 能把 batch 而不是整个 Dataset 移到 device

下一天将实现严格的评估流程、最佳检查点与断点续训状态。

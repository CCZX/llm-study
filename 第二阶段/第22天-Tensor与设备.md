# 第 22 天：Tensor、dtype、shape 与 device

## 今日目标

完成本节后，应当能够：

- 创建和检查 PyTorch Tensor
- 根据代码推断 `shape`、`dtype` 和 `device`
- 使用索引、切片、`reshape`、`view`、`squeeze` 和 `unsqueeze`
- 区分逐元素乘法、点积和矩阵乘法
- 对 `(batch, sequence, hidden)` Tensor 正确聚合和线性变换
- 理解 Tensor 与 NumPy 数组的共享内存关系
- 编写能在 CPU、CUDA 或 MPS 上运行的设备无关代码

## 1. 今日安排

建议总用时为 2～3 小时：

1. 30 分钟：创建 Tensor，检查属性与 dtype。
2. 45 分钟：索引、形状变换和连续内存。
3. 40 分钟：广播、聚合与矩阵乘法。
4. 30 分钟：NumPy 互操作和 device。
5. 30 分钟：完成练习并复盘错误。

运行练习：

~~~bash
python "第二阶段/第22天-Tensor与设备-练习.py"
~~~

## 2. Tensor 是什么

Tensor 是 PyTorch 中保存数值的基本数据结构。它与 NumPy 的 `ndarray` 很相似，但多了两个对深度学习非常重要的能力：

- 可以放到 GPU 或 Apple Silicon 的 MPS 上计算
- 可以参与自动微分，记录计算图并得到梯度

~~~python
import torch

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

print(x)
print(x.shape)   # torch.Size([2, 3])
print(x.ndim)    # 2
print(x.numel()) # 6
print(x.dtype)   # torch.float32
print(x.device)  # cpu
~~~

`torch.Size` 的行为接近元组：

~~~python
assert tuple(x.shape) == (2, 3)
assert x.ndim == len(x.shape)
assert x.numel() == 2 * 3
~~~

## 3. 创建 Tensor

### 3.1 从 Python 数据创建

~~~python
integers = torch.tensor([1, 2, 3])
floats = torch.tensor([1.0, 2.0, 3.0])
booleans = torch.tensor([True, False, True])

print(integers.dtype)  # 通常为 torch.int64
print(floats.dtype)    # 通常为 torch.float32
print(booleans.dtype)  # torch.bool
~~~

不要过度依赖默认 dtype。模型输入、标签和索引通常有明确要求：

~~~python
features = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
class_ids = torch.tensor([0, 2], dtype=torch.long)
~~~

常见约定：

| 用途 | 常见 dtype |
| --- | --- |
| 神经网络输入与参数 | `torch.float32` |
| 类别编号、Token ID、Embedding 索引 | `torch.long` |
| Mask | `torch.bool` |
| 梯度检查 | `torch.float64` |

### 3.2 使用工厂函数

~~~python
torch.zeros((2, 3))
torch.ones((2, 3))
torch.full((2, 3), 7.0)
torch.arange(0, 10, 2)
torch.linspace(0, 1, 5)
torch.randn((2, 3))
~~~

创建与已有 Tensor 属性一致的新 Tensor：

~~~python
x = torch.randn(2, 3, dtype=torch.float32)
same_kind = torch.zeros_like(x)
same_shape_random = torch.randn_like(x)
~~~

`zeros_like`、`ones_like` 和 `randn_like` 默认继承 shape、dtype 和 device，能减少设备不一致错误。

## 4. 索引与切片

~~~python
x = torch.arange(24).reshape(2, 3, 4)
~~~

形状推断：

~~~python
x[0].shape          # (3, 4)
x[0:1].shape        # (1, 3, 4)
x[:, 1, :].shape    # (2, 4)
x[:, :, 0].shape    # (2, 3)
x[..., :2].shape    # (2, 3, 2)
~~~

规则与 NumPy 相同：

- 使用整数索引会移除对应维度
- 使用切片会保留对应维度
- `...` 表示补齐若干个完整维度

布尔索引：

~~~python
scores = torch.tensor([0.2, 0.8, 0.6, 0.1])
selected = scores[scores >= 0.5]
print(selected)  # tensor([0.8000, 0.6000])
~~~

高级索引通常返回新 Tensor；普通切片往往与原 Tensor 共享存储。需要独立数据时使用：

~~~python
copy = x[0].clone()
~~~

## 5. 形状变换

### 5.1 reshape 与 view

~~~python
x = torch.arange(24)

a = x.reshape(2, 3, 4)
b = x.view(6, 4)
c = x.reshape(3, -1)
~~~

`reshape` 和 `view` 都不改变元素总数。区别是：

- `view` 要求底层内存布局兼容
- `reshape` 会尽量返回视图，必要时自动复制

### 5.2 transpose、permute 与 contiguous

~~~python
x = torch.randn(2, 3, 4)

swapped = x.transpose(1, 2)  # (2, 4, 3)
permuted = x.permute(2, 0, 1)  # (4, 2, 3)
~~~

转置后 Tensor 可能不连续：

~~~python
print(swapped.is_contiguous())  # 通常为 False
~~~

如果之后必须使用 `view`：

~~~python
flat = swapped.contiguous().view(2, -1)
~~~

学习初期优先使用 `reshape`，但必须理解 `contiguous`，因为模型代码经常先交换维度再展平。

### 5.3 squeeze 与 unsqueeze

~~~python
x = torch.zeros(1, 8, 1, 16)

x.squeeze().shape       # (8, 16)
x.squeeze(0).shape      # (8, 1, 16)
x.unsqueeze(2).shape    # (1, 8, 1, 1, 16)
~~~

不带维度的 `squeeze()` 会删除所有长度为 1 的维度，batch size 为 1 时容易误删批次维。更安全的做法是显式指定维度。

## 6. 广播

PyTorch 的广播规则与 NumPy 基本一致：从最右侧维度向左比较，每一对维度必须相等，或其中一个等于 1，或其中一方不存在。

~~~python
x = torch.randn(4, 8, 16)
bias = torch.randn(16)

y = x + bias
print(y.shape)  # (4, 8, 16)
~~~

`bias` 被视为 `(1, 1, 16)`，沿 batch 和 sequence 维广播。

为每个样本减去自己的特征均值：

~~~python
x = torch.randn(4, 16)
row_mean = x.mean(dim=1, keepdim=True)
centered = x - row_mean
~~~

如果不使用 `keepdim=True`：

~~~text
x         (4, 16)
row_mean  (4,)
~~~

从右对齐后，`4` 与 `16` 不兼容，因此不能广播。

## 7. 聚合与 dim

~~~python
x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
~~~

~~~python
x.sum()                 # 标量
x.mean(dim=0).shape     # (3, 4)
x.mean(dim=1).shape     # (2, 4)
x.mean(dim=2).shape     # (2, 3)
x.mean(dim=1, keepdim=True).shape  # (2, 1, 4)
~~~

可靠理解：

> `dim=n` 表示沿第 n 维聚合，默认从结果中移除这一维。

同时聚合多个维度：

~~~python
sample_means = x.mean(dim=(1, 2))
print(sample_means.shape)  # (2,)
~~~

## 8. 逐元素乘法与矩阵乘法

### 8.1 逐元素乘法

~~~python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

a * b  # tensor([4., 10., 18.])
~~~

### 8.2 点积

~~~python
torch.dot(a, b)  # 标量 32
a @ b           # 标量 32
~~~

### 8.3 矩阵乘法

~~~python
x = torch.randn(32, 128)
weight = torch.randn(128, 64)
output = x @ weight
print(output.shape)  # (32, 64)
~~~

带序列维的批量线性变换：

~~~python
hidden = torch.randn(8, 20, 64)
weight = torch.randn(64, 128)
bias = torch.randn(128)

projected = hidden @ weight + bias
print(projected.shape)  # (8, 20, 128)
~~~

矩阵乘法只作用于最后两个维度，前面的维度作为批次维保留或广播。

## 9. NumPy 与 Tensor 互操作

~~~python
import numpy as np
import torch

array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
tensor = torch.from_numpy(array)
~~~

`from_numpy` 通常共享 CPU 内存：

~~~python
tensor[0] = 99.0
print(array[0])  # 99.0
~~~

需要独立副本：

~~~python
tensor_copy = torch.tensor(array)
array_copy = tensor.detach().cpu().numpy().copy()
~~~

有梯度或位于加速设备上的 Tensor 不能直接调用 `numpy()`：

~~~python
array = tensor.detach().cpu().numpy()
~~~

## 10. device

~~~python
def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = choose_device()
x = torch.randn(2, 3, device=device)
y = torch.ones_like(x)
z = x + y
~~~

常见错误：

~~~python
x = torch.randn(2, 3, device=device)
y = torch.ones(2, 3)  # 默认在 CPU
z = x + y             # x 不在 CPU 时会报 device 不一致
~~~

修复：

~~~python
y = torch.ones_like(x)
# 或
y = torch.ones(2, 3, device=x.device, dtype=x.dtype)
~~~

`.to(device)` 不保证原地修改：

~~~python
x = x.to(device)
~~~

忘记接收返回值，`x` 仍可能留在原设备。

## 11. 大模型常见形状

设：

~~~python
hidden_states = torch.randn(4, 12, 32)
~~~

三个维度为：

- batch = 4：一次处理 4 段文本
- sequence = 12：每段文本 12 个 Token
- hidden = 32：每个 Token 的表示维度

常见操作：

~~~python
first_sample = hidden_states[0]         # (12, 32)
first_tokens = hidden_states[:, 0, :]   # (4, 32)
mean_pool = hidden_states.mean(dim=1)   # (4, 32)
flattened = hidden_states.reshape(-1, 32)  # (48, 32)
~~~

线性投影：

~~~python
weight = torch.randn(32, 64)
bias = torch.randn(64)
output = hidden_states @ weight + bias
print(output.shape)  # (4, 12, 64)
~~~

## 12. 常见错误

### 12.1 分类标签使用 float

~~~python
labels = torch.tensor([0, 1, 2], dtype=torch.float32)
~~~

`CrossEntropyLoss` 的类别编号通常需要 `torch.long`。

### 12.2 错误删除 batch 维

~~~python
logits = torch.randn(1, 10)
bad = logits.squeeze()   # (10,)
good = logits.squeeze(0) # (10,)，这里是明确知道要删第 0 维
~~~

更常见的做法是让模型始终保留 batch 维，不随意 squeeze。

### 12.3 用 view 处理非连续 Tensor

~~~python
x = torch.randn(2, 3, 4).transpose(1, 2)
x.view(2, -1)  # 可能报错
~~~

使用 `reshape`，或先 `contiguous()`。

### 12.4 新建 Tensor 时忘记 device

优先使用 `zeros_like`、`ones_like` 或从已有 Tensor 读取 `device` 和 `dtype`。

## 13. 自测题

先预测，再运行：

~~~python
x = torch.zeros(2, 3, 4)
w = torch.zeros(4, 5)
b = torch.zeros(5)
~~~

1. `x[:, 1].shape` 是什么？
2. `x[:, 1:2].shape` 是什么？
3. `x.mean(dim=1).shape` 是什么？
4. `x.mean(dim=1, keepdim=True).shape` 是什么？
5. `(x @ w + b).shape` 是什么？
6. `x.permute(1, 0, 2).shape` 是什么？
7. 为什么 `torch.tensor([0, 1])` 可以做类别标签，而 `torch.tensor([0.0, 1.0])` 通常不行？

答案：

~~~text
1. (2, 4)
2. (2, 1, 4)
3. (2, 4)
4. (2, 1, 4)
5. (2, 3, 5)
6. (3, 2, 4)
7. 前者通常是 int64/long 类别编号；后者是浮点数，不符合 CrossEntropyLoss 的索引标签要求。
~~~

## 14. 今日验收

- 能写出至少 5 种 Tensor 创建方式
- 能解释整数索引与切片对 shape 的不同影响
- 能解释 `reshape`、`view` 和 `contiguous` 的关系
- 能判断广播是否合法
- 能区分 `*` 与 `@`
- 能对 `(batch, sequence, hidden)` 做正确聚合和线性投影
- 能安全地在 Tensor 与 NumPy 之间转换
- 能编写 CPU/CUDA/MPS 兼容的设备选择代码

下一天将学习计算图与自动微分，把第一阶段手写的梯度交给 PyTorch 自动计算。

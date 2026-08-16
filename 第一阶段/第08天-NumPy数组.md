# 第 8 天：NumPy 数组

## 今日目标

完成本节后，应当能够：

- 理解 `ndarray`、`shape`、`ndim`、`size` 和 `dtype`
- 创建、索引和切片多维数组
- 使用 `reshape` 改变数组形状
- 使用 `sum`、`mean`、`max` 完成聚合计算
- 根据 `axis` 推断计算结果的形状
- 理解 `(批次, 序列长度, 隐藏维度)` 这种大模型常见数据形状

## 1. 启动环境

```bash
cd /Users/chenchao/Desktop/LLM
source .venv/bin/activate
jupyter lab
```

在 Notebook 中导入 NumPy：

```python
import numpy as np

print(np.__version__)
```

## 2. ndarray 的核心属性

`ndarray` 是 NumPy 的 N 维数组。数组中的元素通常具有相同的数据类型，因此适合批量数值计算。

```python
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

print(a.ndim)   # 2：数组有两个维度
print(a.shape)  # (2, 3)：第0维长度2，第1维长度3
print(a.size)   # 6：元素总数
print(a.dtype)  # 整数类型，例如 int64
```

需要记住：

```python
a.ndim == len(a.shape)
a.size == np.prod(a.shape)
```

常见数据类型：

- `int64`：整数、标签或 Token ID
- `float32`：常规模型计算
- `float16`、`bfloat16`：节省显存的模型计算
- `int8`、`int4`：模型量化中常见

修改数据类型：

```python
a_float = a.astype(np.float32)
```

## 3. 创建数组

```python
np.array([1, 2, 3])             # 从列表创建
np.zeros((2, 3))                # 全零数组
np.ones((2, 3))                 # 全一数组
np.full((2, 3), 7)              # 全部填充为7
np.arange(0, 10, 2)             # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)            # 0到1之间的5个等距数
```

推荐使用可复现的随机数生成器：

```python
rng = np.random.default_rng(42)
x = rng.standard_normal((2, 3)).astype(np.float32)
```

固定随机种子后，每次运行可以得到相同的随机数据，方便复现实验。

## 4. 索引与切片

```python
a = np.array([
    [10, 20, 30],
    [40, 50, 60],
])

a[0]       # 第0行，形状为 (3,)
a[1, 2]    # 第1行第2列，结果为 60
a[:, 1]    # 所有行的第1列
a[:1, :]   # 保留前1行，形状为 (1, 3)
```

关键区别：

```python
a[0].shape    # (3,)：整数索引会移除一个维度
a[0:1].shape  # (1, 3)：切片会保留该维度
```

普通切片通常是原数组的视图，修改切片可能影响原数组。需要独立数据时使用：

```python
part = a[0].copy()
```

## 5. reshape

`reshape` 只能改变形状，不能改变元素总数。

```python
a = np.arange(12)

a.reshape(3, 4)     # 合法：3 × 4 = 12
a.reshape(2, 2, 3)  # 合法：2 × 2 × 3 = 12
a.reshape(3, -1)    # NumPy 自动推断为 (3, 4)
```

下面的操作不合法：

```python
a.reshape(5, 3)  # 5 × 3 != 12
```

大模型中经常把批次和序列维度合并：

```python
x = np.zeros((32, 128, 768))
tokens = x.reshape(32 * 128, 768)

print(tokens.shape)  # (4096, 768)
```

## 6. 聚合运算与 axis

```python
a = np.array([
    [1, 2, 3],
    [4, 5, 6],
])

a.sum()          # 21
a.mean()         # 3.5
a.max()          # 6
a.mean(axis=0)   # [2.5, 3.5, 4.5]
a.mean(axis=1)   # [2.0, 5.0]
```

理解 `axis` 的可靠方法：

> `axis=n` 表示沿第 n 个维度计算，并把该维度从结果形状中移除。

对于形状 `(2, 3)`：

- `axis=0`：移除第0维，结果形状为 `(3,)`
- `axis=1`：移除第1维，结果形状为 `(2,)`

使用 `keepdims=True` 可以保留长度为1的维度：

```python
a.mean(axis=0, keepdims=True).shape  # (1, 3)
a.mean(axis=1, keepdims=True).shape  # (2, 1)
```

## 7. 大模型数据形状练习

```python
rng = np.random.default_rng(42)

hidden_states = rng.standard_normal(
    (32, 128, 768)
).astype(np.float32)
```

三个维度分别表示：

- `32`：批次大小，一次处理32段文本
- `128`：序列长度，每段文本包含128个 Token
- `768`：隐藏维度，每个 Token 由768个数字表示

### 沿批次维度求均值

```python
mean_batch = hidden_states.mean(axis=0)
print(mean_batch.shape)  # (128, 768)
```

含义：对32个样本求平均，保留每个序列位置的768维表示。

### 沿序列维度求均值

```python
mean_sequence = hidden_states.mean(axis=1)
print(mean_sequence.shape)  # (32, 768)
```

含义：把每段文本的128个 Token 聚合成一个768维文本向量，类似 Mean Pooling。

实际文本通常包含 Padding Token，真正做平均池化时需要结合 `attention_mask` 排除填充位置。

### 沿隐藏维度求均值

```python
mean_hidden = hidden_states.mean(axis=2)
print(mean_hidden.shape)  # (32, 128)
```

含义：每个 Token 的768个隐藏特征被压缩为一个数。

### 保留维度

```python
hidden_states.mean(axis=0, keepdims=True).shape  # (1, 128, 768)
hidden_states.mean(axis=1, keepdims=True).shape  # (32, 1, 768)
hidden_states.mean(axis=2, keepdims=True).shape  # (32, 128, 1)
```

## 8. 索引形状练习

```python
x = np.zeros((32, 128, 768))

x[0].shape          # (128, 768)：第一个样本
x[0:1].shape        # (1, 128, 768)：保留批次维度
x[0, 9].shape       # (768,)：第一个样本的第十个Token
x[:, 0, :].shape    # (32, 768)：所有样本的第一个Token
x[:, :, 0].shape    # (32, 128)：所有Token的第一个隐藏特征
x[:, :, :100].shape # (32, 128, 100)：前100个隐藏特征
```

## 9. 自测题

```python
x = np.arange(24).reshape(2, 3, 4)
```

先不运行代码，预测以下结果：

```python
x.ndim
x.size
x[0].shape
x[:, 1, :].shape
x[:, :, 0].shape
x.mean(axis=0).shape
x.mean(axis=1).shape
x.mean(axis=2).shape
```

答案：

```text
x.ndim               -> 3
x.size               -> 24
x[0].shape           -> (3, 4)
x[:, 1, :].shape     -> (2, 4)
x[:, :, 0].shape     -> (2, 3)
x.mean(axis=0).shape -> (3, 4)
x.mean(axis=1).shape -> (2, 4)
x.mean(axis=2).shape -> (2, 3)
```

## 10. 易错点

1. `shape` 是各维度长度组成的元组，`ndim` 是维度数量。
2. `reshape` 前后的元素总数必须相同。
3. 整数索引通常会移除维度，切片通常会保留维度。
4. `axis` 表示被聚合掉的维度，而不是简单地记作“按行”或“按列”。
5. NumPy 切片通常是视图，需要独立副本时使用 `.copy()`。
6. 大模型张量通常按照 `(batch, sequence, hidden)` 理解。

## 11. 今日验收清单

- [ ] 能解释 `shape`、`ndim`、`size`、`dtype`
- [ ] 能创建全零、全一、序列和随机数组
- [ ] 能对二维和三维数组进行索引、切片
- [ ] 能使用 `reshape`，并提前判断是否合法
- [ ] 能根据 `axis` 直接推断结果形状
- [ ] 能解释 `(32, 128, 768)` 三个维度的含义
- [ ] 能解释沿三个轴求均值后的三个结果

## 12. 官方参考文档

- [NumPy 初学者教程](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy 数组索引](https://numpy.org/doc/stable/user/basics.indexing.html)
- [numpy.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)

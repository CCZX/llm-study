# 第 20 天：从零实现 Softmax 分类器

## 今日目标

完成本节后，应当能够：

- 使用类别编号和 one-hot 表示多分类标签
- 实现固定随机种子的分层数据划分
- 实现 `logits = X @ W + b`
- 实现数值稳定的 Softmax 与交叉熵
- 推导并实现 Softmax 分类器参数梯度
- 使用梯度下降训练多分类模型
- 计算准确率和多分类混淆矩阵
- 在 Iris 数据集上与多数类基线比较

## 1. 项目任务

使用 [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris) 的 4 个特征预测 3 种鸢尾花：

```text
输入 X：(samples, 4)
标签 y：(samples,)，取值为 0、1、2
权重 W：(4, 3)
偏置 b：(3,)
logits：(samples, 3)
```

建议目标：测试准确率达到 85% 以上。数据集很小，这只是实现正确性的粗略检查，不是严肃模型基准。

## 2. 今日安排

1. 30 分钟：下载、读取、编码和分层划分。
2. 40 分钟：实现 Softmax、one-hot 和交叉熵。
3. 50 分钟：推导梯度并完成梯度检查。
4. 40 分钟：训练、调参和绘制曲线。
5. 20 分钟：混淆矩阵、基线和项目记录。

运行合成数据练习：

```bash
python "第一阶段/第20天-从零实现Softmax分类器-练习.py"
```

## 3. 读取 Iris 数据

从 UCI 页面下载数据，保存到 `week3_numpy_ml/data/raw/`。原始数据通常没有表头：

```python
columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]

iris = pd.read_csv(data_path, names=columns)
iris = iris.dropna().copy()
```

标签编码：

```python
label_names = sorted(iris["species"].unique())
label_to_id = {name: index for index, name in enumerate(label_names)}

y = iris["species"].map(label_to_id).to_numpy(dtype=np.int64)
x = iris[columns[:-1]].to_numpy(dtype=np.float64)
```

记录 `label_to_id`，否则混淆矩阵中的类别编号无法还原成人类可读名称。

## 4. 为什么需要分层划分

普通随机划分在小数据集上可能使某个集合缺少某一类别。分层划分分别打乱每个类别，再按相同比例分配：

```python
def stratified_split(labels, train_ratio=0.7, valid_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    train_parts = []
    valid_parts = []
    test_parts = []

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)

        train_end = int(len(class_indices) * train_ratio)
        valid_end = train_end + int(len(class_indices) * valid_ratio)

        train_parts.append(class_indices[:train_end])
        valid_parts.append(class_indices[train_end:valid_end])
        test_parts.append(class_indices[valid_end:])

    train = np.concatenate(train_parts)
    valid = np.concatenate(valid_parts)
    test = np.concatenate(test_parts)
    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test
```

划分后打印每个集合的类别计数：

```python
np.bincount(y[train_indices], minlength=3)
```

## 5. 标准化

只使用训练集统计量：

```python
mean = x_train.mean(axis=0, keepdims=True)
std = x_train.std(axis=0, keepdims=True)
safe_std = np.where(std == 0, 1.0, std)

x_train = (x_train - mean) / safe_std
x_valid = (x_valid - mean) / safe_std
x_test = (x_test - mean) / safe_std
```

## 6. one-hot 编码

类别编号：

```text
y = [0, 2, 1]
```

one-hot：

```text
[[1, 0, 0],
 [0, 0, 1],
 [0, 1, 0]]
```

```python
def one_hot(labels, class_count):
    result = np.zeros((len(labels), class_count), dtype=np.float64)
    result[np.arange(len(labels)), labels] = 1.0
    return result
```

one-hot 便于推导梯度，真实框架通常直接接收类别编号以节省内存。

## 7. 模型前向传播

```text
logits = X @ W + b
```

```python
weight = np.zeros((feature_count, class_count), dtype=np.float64)
bias = np.zeros(class_count, dtype=np.float64)

logits = x @ weight + bias
```

偏置 `(classes,)` 会广播到每一个样本。

## 8. 数值稳定的 Softmax

```python
def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)
```

检查：

```python
probabilities = softmax(logits)
assert probabilities.shape == logits.shape
assert np.allclose(probabilities.sum(axis=1), 1.0)
assert np.isfinite(probabilities).all()
```

## 9. 交叉熵

类别编号版本：

```python
def cross_entropy(labels, probabilities):
    correct = probabilities[np.arange(len(labels)), labels]
    safe = np.clip(correct, 1e-12, 1.0)
    return float(-np.log(safe).mean())
```

训练时可以从 logits 直接计算稳定交叉熵；当前实现先显式计算概率，便于观察中间结果。

## 10. 关键梯度

令：

```text
Y      one-hot 标签
P      Softmax 概率
N      样本数量
```

Softmax 与交叉熵组合后，对 logits 的梯度有简洁形式：

```text
dlogits = (P - Y) / N
```

再通过 `logits = X @ W + b`：

```text
dW = Xᵀ @ dlogits
db = sum(dlogits, axis=0)
```

形状：

```text
X.T       (features, samples)
dlogits   (samples, classes)
dW        (features, classes)，与 W 相同
db        (classes,)，与 b 相同
```

实现：

```python
def loss_and_gradients(x, labels, weight, bias):
    logits = x @ weight + bias
    probabilities = softmax(logits)
    loss = cross_entropy(labels, probabilities)

    targets = one_hot(labels, weight.shape[1])
    grad_logits = (probabilities - targets) / len(x)
    grad_weight = x.T @ grad_logits
    grad_bias = grad_logits.sum(axis=0)
    return loss, grad_weight, grad_bias
```

## 11. 梯度检查

使用很小的 `X`、`W`，对几个权重做中心差分：

```text
relative_error = ||analytic-numeric|| / (||analytic||+||numeric||+1e-12)
```

相对误差应足够小，例如小于 `1e-5`。若不一致，检查：

- 是否除以批次大小
- one-hot 类别顺序是否正确
- `X.T` 是否遗漏
- `db` 是否沿样本轴求和
- 数值梯度是否恢复了参数

## 12. 训练循环

```python
for epoch in range(epochs):
    loss, grad_weight, grad_bias = loss_and_gradients(
        x_train, y_train, weight, bias
    )

    weight -= learning_rate * grad_weight
    bias -= learning_rate * grad_bias

    train_loss_history.append(loss)
    valid_accuracy_history.append(
        accuracy(y_valid, predict(x_valid, weight, bias))
    )
```

先尝试：

```text
learning_rate = 0.1
epochs = 1000
```

然后只根据验证集比较至少两种学习率。

## 13. 预测和准确率

```python
def predict(x, weight, bias):
    logits = x @ weight + bias
    return np.argmax(logits, axis=1)

def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))
```

`argmax(logits)` 与 `argmax(softmax(logits))` 相同，因此预测类别时不必额外计算 Softmax。

## 14. 多分类混淆矩阵

```python
def confusion_matrix(y_true, y_pred, class_count):
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix
```

约定：行是真实类别，列是预测类别。对角线是正确预测，非对角线显示最常混淆的类别对。

## 15. 基线

多数类基线：

```python
majority_class = np.argmax(np.bincount(y_train))
baseline_predictions = np.full_like(y_test, majority_class)
```

Iris 类别大致平衡，多数类基线约为三分之一。模型达到 85% 才明显高于这个简单基线。

## 16. 决策边界的限制

Softmax 分类器仍然是线性分类器。任意两个类别之间的分界由线性方程决定，因此它只能学习线性决策边界。

如果特征空间中的类别关系是明显非线性的，需要特征变换或隐藏层。第 21 天将增加 ReLU 隐藏层。

## 17. 最终报告

至少记录：

- 标签编码映射
- 三个集合的类别数量
- 训练集标准化统计量
- 学习率和训练轮数比较
- 最佳验证准确率
- 最终测试准确率
- 混淆矩阵
- 多数类基线准确率
- 训练曲线和已知局限

## 18. 易错点

1. 在划分前标准化。
2. 普通随机划分导致小集合缺少类别。
3. 忘记 Softmax 减最大值。
4. one-hot 类别数量或顺序错误。
5. `dlogits` 忘记除以样本数。
6. `db` 沿错误轴求和。
7. 使用测试集选择学习率。
8. 混淆矩阵行列含义没有说明。
9. 只报告准确率，没有多数类基线。

## 19. 今日验收清单

- [ ] 完成固定种子的分层划分
- [ ] 只使用训练集统计量标准化
- [ ] one-hot 编码形状和类别顺序正确
- [ ] Softmax 对极端输入保持有限
- [ ] 交叉熵与梯度通过小样本测试
- [ ] 解析梯度通过数值检查
- [ ] 训练损失明显下降
- [ ] 测试准确率明显超过多数类基线
- [ ] 输出并解释混淆矩阵
- [ ] 配套练习全部显示 `[通过]`


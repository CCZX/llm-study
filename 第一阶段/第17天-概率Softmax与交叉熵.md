# 第 17 天：概率、Softmax 与交叉熵

## 今日目标

完成本节后，应当能够：

- 区分随机试验、随机变量、事件和概率分布
- 计算离散分布的期望与方差
- 计算联合概率和条件概率
- 解释独立性与“互斥”的区别
- 通过模拟理解大数定律
- 从最大似然角度理解交叉熵
- 连接 logits、Softmax、正确类别概率与交叉熵

## 1. 今日安排

建议总用时为 2～3 小时：

1. 35 分钟：随机变量与概率分布。
2. 40 分钟：条件概率、独立性、期望和方差。
3. 35 分钟：模拟实验与大数定律。
4. 50 分钟：最大似然、Softmax 和交叉熵。

运行练习：

```bash
python "第一阶段/第17天-概率Softmax与交叉熵-练习.py"
```

## 2. 随机试验、样本空间与事件

投掷一次六面骰子：

- 随机试验：投掷骰子。
- 样本空间：`{1, 2, 3, 4, 5, 6}`。
- 一个结果：例如 `4`。
- 事件：一个或多个结果组成的集合，例如“点数为偶数”是 `{2, 4, 6}`。

公平骰子中：

```text
P(点数为偶数) = 3/6 = 0.5
```

概率必须满足：

```text
0 ≤ P(A) ≤ 1
P(样本空间) = 1
```

## 3. 随机变量和概率分布

随机变量把随机结果映射成数字。骰子点数本身可以记为随机变量 `X`。

公平骰子的概率质量函数：

| `x` | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `P(X=x)` | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |

```python
values = np.arange(1, 7)
probabilities = np.full(6, 1.0 / 6.0)

assert np.all(probabilities >= 0)
assert np.isclose(probabilities.sum(), 1.0)
```

机器学习模型输出的类别概率也是离散概率分布。例如三分类输出 `[0.7, 0.2, 0.1]`。

## 4. 期望

离散随机变量的期望：

```text
E[X] = Σₓ x P(X=x)
```

公平骰子：

```text
E[X] = (1+2+3+4+5+6)/6 = 3.5
```

```python
expected_value = np.sum(values * probabilities)
```

期望是长期平均，不要求某次试验真的取到这个值。骰子不会掷出 `3.5`，但大量投掷的平均点数会靠近 `3.5`。

## 5. 方差

```text
Var(X) = E[(X-E[X])²]
```

等价公式：

```text
Var(X) = E[X²] - E[X]²
```

```python
mean = np.sum(values * probabilities)
variance = np.sum((values - mean) ** 2 * probabilities)
```

期望描述中心位置，方差描述围绕期望的波动程度。

## 6. 联合概率与条件概率

条件概率：在事件 `B` 已发生的前提下，事件 `A` 发生的概率：

```text
P(A | B) = P(A ∩ B) / P(B)
```

骰子例子：

- `A`：点数大于 3，即 `{4,5,6}`。
- `B`：点数为偶数，即 `{2,4,6}`。
- `A∩B`：`{4,6}`。

```text
P(A | B) = (2/6) / (3/6) = 2/3
```

数据中的条件频率：

```python
scores = np.array([1, 2, 3, 4, 5, 6])
event_a = scores > 3
event_b = scores % 2 == 0

conditional = np.mean(event_a & event_b) / np.mean(event_b)
```

## 7. 独立性与互斥不是一回事

事件 `A`、`B` 独立：

```text
P(A ∩ B) = P(A)P(B)
```

等价地，在 `P(B)>0` 时：

```text
P(A | B) = P(A)
```

独立意味着知道 `B` 是否发生，不会改变对 `A` 的概率判断。

互斥意味着两个事件不能同时发生：

```text
P(A ∩ B) = 0
```

两个概率都大于 0 的互斥事件不可能独立，因为 `P(A)P(B)>0`。

例如一次掷骰子，“点数为 1”和“点数为 2”互斥，但不独立。

## 8. 经验频率与大数定律

```python
rng = np.random.default_rng(42)

for sample_count in [10, 100, 10_000, 1_000_000]:
    rolls = rng.integers(1, 7, size=sample_count)
    frequency_of_six = np.mean(rolls == 6)
    print(sample_count, frequency_of_six)
```

样本较少时频率波动很大；样本增加时，经验频率通常逐渐靠近理论概率 `1/6`。

大数定律描述的是长期收敛趋势，不保证样本数每增加一次，误差都会单调减小。

固定随机种子用于复现实验，但不要把某个种子的偶然结果误认为普遍规律。

## 9. 概率、似然与最大似然

概率和似然使用同一个数学表达式，但关注对象不同：

- 概率：参数固定，讨论数据出现的可能性。
- 似然：数据已经观察到，比较不同参数对数据的解释能力。

抛硬币 10 次，观察到 7 次正面。若正面概率为 `p`，这组具体结果的似然包含：

```text
L(p) ∝ p⁷(1-p)³
```

最大似然估计会选择使观察数据最可能的 `p`，这里得到 `p=7/10=0.7`。

直接连乘许多小概率容易下溢，因此取对数：

```text
log L(p) = 7 log(p) + 3 log(1-p)
```

对数是单调函数，最大化似然与最大化对数似然得到相同参数。

## 10. 从最大似然到交叉熵

分类模型给每个样本的正确类别一个概率：

```text
p₁, p₂, ..., pₙ
```

假设样本独立，整批正确标签的似然是：

```text
L = p₁ × p₂ × ... × pₙ
```

最大化对数似然：

```text
log L = Σ log(pᵢ)
```

训练通常最小化损失，因此加负号并取平均：

```text
loss = -(1/N) Σ log(pᵢ)
```

这就是 one-hot 分类标签下的平均交叉熵。因此：

> 最小化交叉熵，等价于最大化训练标签的对数似然。

## 11. logits 与 Softmax

模型通常输出任意实数 logits：

```python
logits = np.array([[2.0, 1.0, -1.0]])
```

Softmax 转成概率：

```python
def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)
```

Softmax 保持相对顺序，并让所有类别概率非负、总和为 1。

平移不变性：

```text
softmax(logits + c) = softmax(logits)
```

因为所有指数项都乘上 `exp(c)`，分子和分母中的共同因子会约掉。减去每行最大值正是利用这个性质提高数值稳定性。

## 12. 交叉熵复习

one-hot 标签：

```text
H(y, p) = -Σ yᵢ log(pᵢ)
```

因为只有正确类别的 `yᵢ=1`：

```text
H(y, p) = -log(正确类别概率)
```

```python
probabilities = np.array([0.9, 0.5, 0.1])
losses = -np.log(probabilities)

# [0.105, 0.693, 2.303]
```

正确类别概率越低，损失增长越快。详细数值稳定实现见第 10 天讲义。

## 13. 信息量的直觉

事件越罕见，发生时带来的信息越大：

```text
information(x) = -log P(x)
```

- 概率为 `1` 的必然事件，信息量为 `0`。
- 概率很小的意外事件，信息量很大。

交叉熵可以理解为：真实数据按照预测分布编码时，平均需要多少信息量。若使用自然对数，单位是 nat；若使用以 2 为底的对数，单位是 bit。

## 14. 在大模型中的联系

语言模型训练数据提供正确的下一个 Token。模型输出整个词表的概率分布，交叉熵计算正确 Token 的负对数概率。

```text
输入：  今天天气很
标签：  好
模型给“好”的概率：0.6
该位置损失：-log(0.6) ≈ 0.511
```

一段文本中每个有效 Token 都产生一个损失，再对批次中的有效位置求平均。Padding 和不参与训练的位置通常通过 Mask 忽略。

困惑度：

```text
Perplexity = exp(平均交叉熵)
```

困惑度只能在相同 Tokenizer、数据集和计算方式下比较。

## 15. 易错点

1. 把互斥事件当作独立事件。
2. 条件概率忘记除以条件事件概率。
3. 把期望理解为一次试验必然得到的值。
4. 认为经验频率会单调靠近理论概率。
5. 直接连乘大量小概率，导致下溢。
6. 把 logits 当作概率。
7. Softmax 没有沿类别维度计算。
8. 直接对极端 logits 求指数。
9. 把相关或条件频率解释为因果关系。

## 16. 今日验收清单

- [ ] 能解释随机变量、事件和概率分布
- [ ] 能计算离散分布的期望与方差
- [ ] 能使用条件概率公式
- [ ] 能区分独立与互斥
- [ ] 能通过模拟解释大数定律
- [ ] 能解释概率与似然关注对象的不同
- [ ] 能解释最大化似然与最小化交叉熵的关系
- [ ] 能验证 Softmax 平移不变性
- [ ] 配套练习全部显示 `[通过]`

## 17. 参考资料

- [动手学深度学习：概率](https://zh.d2l.ai/chapter_preliminaries/probability.html)
- [Seeing Theory：概率可视化](https://seeing-theory.brown.edu/basic-probability/index.html)
- [NumPy 随机数生成](https://numpy.org/doc/stable/reference/random/index.html)

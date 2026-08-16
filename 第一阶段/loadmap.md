# 第一阶段学习路线：编程、数学与机器学习基础

## 阶段定位

第一阶段的任务是为后续学习 PyTorch、Transformer、RAG 和模型微调建立共同基础。

建议周期为 **3 周，每周学习 6 天，每天 2～3 小时**。如果完全没有编程基础，可以把每天的内容拆成两天，总周期延长到 5～6 周。

完成本阶段后，应当能够：

- 使用 Python 编写结构清晰的脚本和小型项目
- 使用虚拟环境、命令行、Git、JupyterLab 和 pytest
- 使用 NumPy、Pandas 和 Matplotlib 完成数据处理与可视化
- 理解向量、矩阵、导数、梯度、条件概率、期望和交叉熵
- 理解训练集、验证集、测试集、损失函数、梯度下降、过拟合和数据泄漏
- 不依赖 PyTorch，使用 NumPy 实现线性回归和 Softmax 分类器
- 能够阅读并解释简单的两层神经网络代码

## 学习原则

每天建议按照下面的比例安排时间：

- 30～45 分钟：课程、文档或教材
- 60～90 分钟：独立编码和实验
- 15～30 分钟：测试、整理笔记和复盘错误

执行时遵守以下原则：

1. 每个概念都必须配一段自己写的代码。
2. 先预测运行结果，再执行代码验证。
3. 不复制完整答案；可以查文档，但应自己重新实现。
4. 每周至少完成一个能独立运行的项目。
5. Git 提交应反映开发过程，不要等全部完成后只提交一次。
6. 第一阶段只学习支撑后续大模型课程所需的数学，不追求完整数学体系。

## 环境准备

在项目根目录执行：

```bash
cd /Users/chenchao/Desktop/LLM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib jupyterlab pytest
```

常用命令：

```bash
jupyter lab
pytest -q
git status
git diff
git log --oneline
```

需要理解：

- `.venv` 用于隔离项目依赖，不提交到 Git。
- `requirements.txt` 用于记录可重新安装的项目依赖。
- Notebook 适合实验和展示，正式逻辑应逐步整理到 `.py` 文件中。
- 固定随机种子可以让数据划分和训练结果更容易复现。

参考资料：

- [Python 虚拟环境官方文档](https://docs.python.org/3/library/venv.html)
- [JupyterLab 安装说明](https://jupyter.org/install)
- [pytest 入门教程](https://docs.pytest.org/en/stable/getting-started.html)

## 核心课程

### Python

主课程使用 [Harvard CS50P：Python 编程导论](https://cs50.harvard.edu/python/)，重点学习：

- Week 0：Functions, Variables
- Week 1：Conditionals
- Week 2：Loops
- Week 3：Exceptions
- Week 4：Libraries
- Week 5：Unit Tests
- Week 6：File I/O
- Week 8：Object-Oriented Programming
- Week 9：Et Cetera 中与迭代器、生成器有关的内容

已经掌握其他编程语言时，可以把 [Python 官方教程](https://docs.python.org/3/tutorial/)作为速查资料，重点阅读第 4～9 章和第 12 章。

### 命令行与 Git

使用 [MIT Missing Semester](https://missing.csail.mit.edu/)，重点学习：

- Introduction to the Shell
- Development Environment and Tools
- Debugging and Profiling
- Version Control and Git
- Packaging and Shipping Code

Git 的补充资料为 [Pro Git](https://git-scm.com/book/en/v2.html)，第一阶段只需学习 Getting Started、Git Basics 和 Git Branching 的前半部分。

### NumPy、Pandas 与可视化

- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy 初学者资源](https://numpy.org/learn/)
- [10 Minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [pandas 入门教程](https://pandas.pydata.org/docs/getting_started/index.html)
- [Matplotlib Pyplot 教程](https://matplotlib.org/stable/tutorials/pyplot.html)

### 数学基础

中文主线使用《动手学深度学习》的[预备知识](https://zh.d2l.ai/chapter_preliminaries/index.html)：

- [线性代数](https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html)
- [微积分](https://zh.d2l.ai/chapter_preliminaries/calculus.html)
- [概率](https://zh.d2l.ai/chapter_preliminaries/probability.html)

辅助理解资料：

- [3Blue1Brown：线性代数的本质](https://www.3blue1brown.com/topics/linear-algebra)
- [3Blue1Brown：神经网络、梯度下降与反向传播](https://www.3blue1brown.com/topics/neural-networks)
- [Seeing Theory：概率可视化](https://seeing-theory.brown.edu/basic-probability/index.html)
- [MIT 线性代数](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/)，只用于补充薄弱知识，不要求完整刷完
- [MIT 单变量微积分](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/)，重点查看导数和链式法则

### 机器学习基础

使用 [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)，重点学习：

- Introduction to Machine Learning
- Linear Regression
- Logistic Regression
- Classification
- Datasets, Generalization, and Overfitting
- Neural Networks

完成课程中的[交互与编程练习](https://developers.google.com/machine-learning/crash-course/exercises)，重点观察学习率、损失曲线、分类阈值和过拟合现象。

---

# 第一周：Python 与工程基础

## 第 1 天：环境和命令行

学习内容：

- 目录、绝对路径和相对路径
- `pwd`、`ls`、`cd`、`mkdir`、`cp`、`mv`
- Python 解释器、脚本和 Jupyter Notebook 的区别
- 虚拟环境、pip 和依赖隔离
- Git 仓库的基本概念

实践任务：

1. 创建并激活 `.venv`。
2. 安装第一阶段所需依赖。
3. 创建 `hello.py` 并从命令行运行。
4. 启动 JupyterLab，创建一个 Notebook 并运行代码。
5. 执行 `git status`，确认 `.venv` 未被追踪。

验收标准：能够解释为什么每个项目都应有独立虚拟环境，以及 Notebook 和 Python 脚本分别适合什么场景。

## 第 2 天：变量、条件与循环

学习内容：

- `str`、`int`、`float`、`bool`
- `list`、`tuple`、`dict`、`set`
- `if`、`for`、`while`
- 字符串处理、切片和推导式

实践任务：

- 统计一段文字的字符数、单词数和行数。
- 删除重复单词。
- 统计单词出现次数并按频率排序。
- 输出长度最长的 10 个单词。

验收标准：不查答案独立完成词频统计，并能说明列表、集合和字典的使用场景。

## 第 3 天：函数与模块

学习内容：

- 参数、返回值和作用域
- 默认参数和关键字参数
- 类型标注和 Docstring
- 模块、包和 `import`
- 单一职责原则

实践任务：

```python
def tokenize(text: str) -> list[str]:
    ...

def count_words(tokens: list[str]) -> dict[str, int]:
    ...

def top_k_words(
    counts: dict[str, int],
    k: int,
) -> list[tuple[str, int]]:
    ...
```

验收标准：每个函数只承担一项职责；主程序只负责组织调用，不堆积全部业务逻辑。

## 第 4 天：文件、异常与命令行参数

学习内容：

- 使用 `pathlib` 操作路径
- 读写 TXT、CSV 和 JSON
- `try`、`except`、`raise`
- 使用 `argparse` 接收命令行参数
- `if __name__ == "__main__"`

实践任务：让程序支持下面的调用方式：

```bash
python text_stats.py article.txt --top-k 20 --output result.json
```

程序必须正确处理：

- 文件不存在
- 文件为空
- 文件编码错误
- `top-k` 不是正整数
- 输出目录不存在

## 第 5 天：类、迭代器、生成器与装饰器

学习内容：

- 类、实例、属性和方法
- `__init__` 和 `__repr__`
- 可迭代对象和迭代器
- 使用 `yield` 创建生成器
- 装饰器的基本结构

实践任务：

- 创建一个 `TextDocument` 类保存路径和统计结果。
- 使用生成器逐行读取大文件，避免一次载入全部内容。
- 编写一个简单的计时装饰器，输出函数运行时间。

第一阶段不需要深入元类、多重继承和复杂装饰器。

## 第 6 天：pytest、调试与 Git

学习内容：

- `assert` 和 pytest 测试发现规则
- 正常输入、边界输入和异常输入
- 使用 `pytest.approx` 比较浮点数
- 使用断点、日志和最小复现定位错误
- `git add`、`commit`、`diff`、`log`、`switch`

实践任务：

- 为文本处理函数编写至少 8 个测试。
- 创建一个功能分支并完成一次合并。
- 故意制造一个错误，使用测试和调试器定位它。

## 第 7 天：第一周项目——文本语料分析器

输入一个或多个 TXT/Markdown 文件，输出：

- 文件数、总字符数、总行数和总词数
- 唯一词数量
- Top-K 高频词
- 平均词长
- 指定关键词出现次数
- JSON 或 CSV 格式的统计结果

项目要求：

- 至少拆分为 4 个函数
- 使用 `argparse`、`pathlib` 和异常处理
- 使用生成器读取文件
- 至少包含 8 个 pytest 测试
- 提供 README 和运行示例
- 至少产生 5 次有意义的 Git 提交

项目与 LLM 的联系：文本读取、清洗、统计和批量处理是后续语料准备、Tokenization 和 RAG 文档处理的基础。

---

# 第二周：NumPy、Pandas 与数据分析

## 第 8 天：NumPy 数组

详细讲义和练习已放在当前目录：

- [第08天-NumPy数组.md](./第08天-NumPy数组.md)
- [第08天-NumPy数组-练习.py](./第08天-NumPy数组-练习.py)

必须掌握：

- `ndarray`、`shape`、`ndim`、`size` 和 `dtype`
- 数组创建、索引、切片和 `reshape`
- `sum`、`mean`、`max` 和 `axis`
- `(batch, sequence, hidden)` 的形状含义

完成练习后运行：

```bash
python "第一阶段/第08天-NumPy数组-练习.py"
```

## 第 9 天：广播、向量化与矩阵运算

学习内容：

- Element-wise 运算
- Broadcasting 规则
- 点积、矩阵乘法、转置
- `@`、`np.dot` 和 `np.matmul`
- 使用向量化代替 Python 循环

实践任务：

- 对矩阵的每一列进行标准化。
- 手写余弦相似度函数。
- 一次计算一个查询向量与 100 个候选向量的相似度。
- 比较循环版本和向量化版本的运行时间。

验收标准：看到矩阵运算前能先推断结果形状，并能解释广播为什么合法或不合法。

## 第 10 天：数值函数和数值稳定性

学习内容：

- 均值、方差和范数
- `exp`、`log` 和浮点数范围
- 上溢、下溢和除零问题
- Softmax 减去最大值的原因

必须实现：

```python
def sigmoid(x):
    ...

def softmax(x):
    ...

def mse(y_true, y_pred):
    ...

def cross_entropy(y_true, probabilities):
    ...
```

测试要求：

- Softmax 每行之和约等于 1。
- 极大或极小输入不会轻易产生 `NaN`。
- MSE 在预测完全正确时等于 0。
- 正确类别概率越低，交叉熵越大。

## 第 11 天：Pandas 基础

学习内容：

- `Series` 和 `DataFrame`
- 读取 CSV
- `head`、`info`、`describe`
- 行列选择、布尔过滤和排序
- 创建新列和修改数据类型

实践任务：

- 载入一个 CSV 数据集。
- 查看字段、数据类型和样本数量。
- 找出缺失值和重复数据。
- 根据多个条件过滤样本。

## 第 12 天：数据清洗、分组统计与可视化

学习内容：

- 缺失值处理和重复值处理
- `groupby`、聚合、连接和合并
- 直方图、散点图和箱线图
- 相关系数的含义和局限
- 训练数据泄漏的基本概念

实践任务：

- 按类别或数值区间分组统计。
- 绘制三个有明确问题导向的图表。
- 为图表添加标题、坐标名称和单位。
- 用文字说明每张图能支持什么结论、不能支持什么结论。

## 第 13～14 天：第二周项目——葡萄酒质量数据报告

数据集：[UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine%2Bquality)

创建 `wine_analysis.ipynb`，完成：

1. 数据读取与字段说明。
2. 缺失值、重复值和异常值检查。
3. 描述性统计。
4. 目标变量分布分析。
5. 特征直方图、散点图和箱线图。
6. 特征与质量评分的相关性分析。
7. 使用 NumPy 完成特征标准化。
8. 使用固定随机种子划分训练集、验证集和测试集。
9. 用文字总结至少 5 条发现和 2 个局限。

验收标准：

- Notebook 能从头到尾无报错运行。
- 图表标题和坐标完整。
- 标准化均值和标准差只能从训练集计算。
- 能解释为什么先对完整数据标准化会产生数据泄漏。
- README 写清数据来源、运行方法和结论。

---

# 第三周：数学与 NumPy 机器学习

## 第 15 天：线性代数

必须掌握：

- 标量、向量、矩阵和张量
- 形状与维度
- 点积和矩阵乘法
- 转置和范数
- 线性变换的直观含义

自测题：

1. 为什么 `(32, 768) @ (768, 4)` 的结果是 `(32, 4)`？
2. 为什么两个长度为 768 的向量点积得到一个标量？
3. 余弦相似度为什么要除以两个向量的范数？
4. 大模型中的 Embedding 查表结果为什么通常是三维张量？

特征值和特征向量在本阶段只要求形成直觉，不要求熟练推导。

## 第 16 天：微积分与梯度下降

必须掌握：

- 导数表示局部变化率
- 偏导数和梯度
- 链式法则
- 梯度下降
- 学习率

实践任务：

- 手算 `f(x) = x²` 和 `f(x) = (x - 3)²` 的导数。
- 使用有限差分近似导数。
- 比较解析梯度与数值梯度。
- 绘制梯度下降优化 `f(x) = (x - 3)²` 的轨迹。
- 比较过大、合适和过小的三种学习率。

## 第 17 天：概率、Softmax 与交叉熵

必须掌握：

- 随机变量和概率分布
- 条件概率
- 期望和方差
- 独立性的含义
- 对数的基本性质
- 最大似然的直觉
- Softmax 和交叉熵

实践任务：

- 模拟投骰子，观察样本数增大时频率如何接近理论概率。
- 使用 NumPy 计算均值、方差和条件频率。
- 比较正确类别概率为 `0.9`、`0.5` 和 `0.1` 时的交叉熵。
- 验证为所有 logits 加上同一个常数不会改变 Softmax 输出。

## 第 18 天：机器学习基本流程

学习内容：

- 样本、特征、标签、模型和参数
- 训练集、验证集和测试集
- 损失函数和评价指标
- 参数与超参数
- 欠拟合和过拟合
- 优化与泛化
- 基线模型
- 准确率、精确率、召回率和混淆矩阵

自测要求：能够解释：

- 为什么测试集不能参与调参？
- 为什么训练损失下降不代表模型一定更好？
- 损失函数和评价指标有什么区别？
- 为什么需要一个简单基线？

## 第 19 天：从零实现线性回归

使用 Wine Quality 数据集预测质量评分，实现：

- 训练、验证和测试划分
- 仅使用训练集统计量标准化
- 线性模型 `y_hat = X @ w + b`
- MSE 和解析梯度
- 批量梯度下降
- 训练损失与验证损失曲线
- 测试集 RMSE

基线模型：始终预测训练集目标均值。最终模型应与这个基线比较，而不是只报告单独的 RMSE。

## 第 20 天：从零实现 Softmax 分类器

数据集：[UCI Iris](https://archive.ics.uci.edu/dataset/53/iris)

实现：

- 固定随机种子的分层数据划分
- One-hot 编码
- `logits = X @ W + b`
- 数值稳定的 Softmax
- 交叉熵
- 参数梯度和梯度下降
- 准确率和混淆矩阵

建议目标：测试准确率达到 85% 以上。该数据集很小，这一目标只用于检查实现是否大致正确，不作为严肃模型基准。

## 第 21 天：两层神经网络与项目整理

在 Softmax 分类器上增加一个隐藏层：

```text
输入 → Linear → ReLU → Linear → Softmax
```

完成版要求：

- 只使用 NumPy 完成前向传播和反向传播
- 尝试至少两种隐藏层大小
- 尝试至少两种学习率
- 绘制训练损失和验证准确率曲线
- 使用数值梯度抽查部分参数梯度

如果三周内时间不足，优先保证线性回归和 Softmax 分类器完整、可测试；两层神经网络可增加 1～2 天完成，但必须在进入 PyTorch 阶段前读懂其前向传播和梯度流向。

---

# 阶段项目结构建议

```text
第一阶段/
├── loadmap.md
├── 第08天-NumPy数组.md
├── 第08天-NumPy数组-练习.py
├── week1_text_stats/
│   ├── README.md
│   ├── text_stats.py
│   └── test_text_stats.py
├── week2_data_analysis/
│   ├── README.md
│   ├── wine_analysis.ipynb
│   └── figures/
└── week3_numpy_ml/
    ├── README.md
    ├── data.py
    ├── losses.py
    ├── linear_regression.py
    ├── softmax_classifier.py
    ├── mlp.py
    ├── train.py
    └── tests/
```

# 阶段验收标准

## 知识验收

完成第一阶段时，应当能够独立回答：

- Python 列表和 NumPy 数组有什么区别？
- 什么是数组形状、广播和向量化？
- 矩阵乘法和逐元素相乘有什么区别？
- 导数、偏导数和梯度分别表示什么？
- 学习率过大和过小会分别产生什么现象？
- 为什么要划分训练集、验证集和测试集？
- 什么是数据泄漏？
- MSE 和交叉熵分别适合什么任务？
- Softmax 为什么要先减去每行最大值？
- 训练损失下降为什么不代表泛化能力一定提高？
- 参数与超参数有什么区别？

## 代码验收

- 所有 pytest 测试通过。
- 项目可在新虚拟环境中重新安装并运行。
- Notebook 可以从头到尾无报错执行。
- NumPy 模型的训练损失明显下降。
- 解析梯度和数值梯度基本一致。
- 数据标准化和模型选择没有使用测试集信息。
- 每个项目都有 README、运行命令、结果和已知问题。
- Git 历史能够体现逐步实现、测试和修复过程。

## 进入第二阶段的判断

满足以下条件后再开始 PyTorch：

1. 能独立完成一次数据读取、清洗、划分、训练和评测流程。
2. 能根据代码推断主要数组的形状。
3. 能解释线性回归和 Softmax 的前向计算、损失与参数更新。
4. 遇到 Loss 不下降时，知道检查数据、形状、梯度、学习率和数值稳定性。
5. 能用测试验证关键数学函数，而不是只凭肉眼查看输出。

完成以上内容后，进入第二阶段：PyTorch Tensor、自动微分、`Dataset`、`DataLoader`、`nn.Module` 和完整训练循环。

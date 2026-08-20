# 第 11 天：Pandas 基础

## 今日目标

完成本节后，应当能够：

- 理解 `Series` 和 `DataFrame` 的结构
- 从 CSV 读取数据并进行基础检查
- 使用列选择、`loc`、`iloc` 和布尔条件选择数据
- 按一列或多列排序
- 创建新列并修改数据类型
- 统计缺失值与重复行
- 避免链式赋值和原地修改带来的常见问题

## 1. 今日安排

建议总用时为 2～3 小时：

1. 30 分钟：Series、DataFrame 和索引。
2. 40 分钟：读取与检查 CSV。
3. 50 分钟：选择、过滤、排序与新列。
4. 30 分钟：数据类型、缺失值、重复值和练习。

运行配套练习：

```bash
cd /Users/chenchao/Desktop/LLM
source .venv/bin/activate
python "第一阶段/第11天-Pandas基础-练习.py"
```

练习数据位于：

```text
第一阶段/data/第11天-模型实验.csv
```

## 2. Series 与 DataFrame

### 2.1 Series

`Series` 是带索引的一维数据：

```python
import pandas as pd

scores = pd.Series(
    [0.82, 0.88, 0.91],
    index=["TinyLM", "SmallLM", "BaseLM"],
    name="accuracy",
)

print(scores)
print(scores.index)
print(scores.dtype)
```

### 2.2 DataFrame

`DataFrame` 是带行索引和列名的二维表格：

```python
experiments = pd.DataFrame(
    {
        "model": ["TinyLM", "SmallLM", "BaseLM"],
        "accuracy": [0.82, 0.88, 0.91],
        "latency_ms": [120, 95, 180],
    }
)

print(experiments.shape)   # (3, 3)
print(experiments.columns)
print(experiments.index)
```

理解方式：

- 每一列通常代表一个变量或特征。
- 每一行通常代表一个样本、记录或实验。
- 每列可以有自己的 dtype。
- 行索引是数据定位标签，不等同于普通数据列。

## 3. 读取 CSV

使用相对脚本位置构造路径，避免依赖当前终端目录：

```python
from pathlib import Path

DATA_PATH = Path("第一阶段/data/第11天-模型实验.csv")
df = pd.read_csv(DATA_PATH)
```

常用参数：

```python
pd.read_csv(
    DATA_PATH,
    sep=",",               # 分隔符
    encoding="utf-8",      # 文件编码
    na_values=["", "NA"], # 额外的缺失值标记
)
```

不同数据文件可能使用逗号、分号或制表符分隔。第 13～14 天使用的 Wine Quality CSV 使用分号 `;`。

## 4. 读取后先做基础检查

不要读取后立刻画图或建模。先回答下面的问题：

```python
print(df.shape)       # 有多少行、多少列
print(df.columns)     # 列名是否符合预期
print(df.head())      # 前 5 行
print(df.tail())      # 后 5 行
print(df.dtypes)      # 每列类型
df.info()             # 非空数量、类型和内存概况
```

数值列的描述统计：

```python
print(df.describe())
```

同时包含非数值列：

```python
print(df.describe(include="all"))
```

重点检查：

- 行列数是否符合数据说明
- 列名是否有空格或拼写问题
- 数字是否错误地读取成字符串
- 关键列是否存在缺失
- 类别列是否出现意外取值
- 唯一标识列是否真的唯一

## 5. 选择列

选择单列会得到 `Series`：

```python
accuracy = df["accuracy"]
print(type(accuracy))  # pandas.Series
```

选择多列会得到 `DataFrame`：

```python
metrics = df[["model", "accuracy", "latency_ms"]]
print(type(metrics))   # pandas.DataFrame
```

注意双层方括号：外层是 DataFrame 选择操作，内层是列名列表。

## 6. 使用 loc 与 iloc

### `loc`：按标签选择

```python
df.loc[0, "model"]
df.loc[0:2, ["model", "accuracy"]]
```

`loc[0:2]` 通常包含结束标签 `2`。

### `iloc`：按整数位置选择

```python
df.iloc[0, 1]
df.iloc[0:3, 1:4]
```

`iloc[0:3]` 与 Python 切片一样，不包含位置 `3`。

简单记忆：

- `loc` 看标签和布尔条件
- `iloc` 看第几行、第几列

## 7. 布尔过滤

筛选准确率至少为 `0.85` 的实验：

```python
high_accuracy = df.loc[df["accuracy"] >= 0.85]
```

组合多个条件时，每个条件都加括号：

```python
selected = df.loc[
    (df["accuracy"] >= 0.85)
    & (df["latency_ms"] < 100)
]
```

常用操作符：

- `&`：并且
- `|`：或者
- `~`：取反
- `.isin([...])`：属于指定集合
- `.between(left, right)`：落在区间内
- `.notna()`：不是缺失值

```python
dev_models = df.loc[df["dataset"].isin(["dev", "test"])]
valid_accuracy = df.loc[df["accuracy"].notna()]
```

不要在 Series 条件中使用 Python 的 `and`、`or`。

## 8. 排序

按准确率从高到低：

```python
ranked = df.sort_values("accuracy", ascending=False)
```

多列排序：准确率降序，相同时延迟升序：

```python
ranked = df.sort_values(
    ["accuracy", "latency_ms"],
    ascending=[False, True],
    na_position="last",
)
```

`sort_values` 默认返回新 DataFrame，不会修改原对象。

## 9. 创建和修改列

创建延迟秒数：

```python
df_with_seconds = df.assign(
    latency_seconds=df["latency_ms"] / 1000.0
)
```

创建布尔标记：

```python
df_with_flag = df.assign(
    meets_target=(df["accuracy"] >= 0.85) & (df["latency_ms"] < 100)
)
```

`assign` 返回新 DataFrame，适合练习中保留原始数据。直接赋值也可以：

```python
df = df.copy()
df["accuracy_pct"] = df["accuracy"] * 100
```

## 10. 修改数据类型

```python
df["experiment_id"] = df["experiment_id"].astype("string")
df["model"] = df["model"].astype("category")
df["quantized"] = df["quantized"].astype("bool")
```

常见类型：

- `int64`、`float64`：数值
- `bool`：真假值
- `string`：文本
- `category`：重复较多的有限类别
- `datetime64[ns]`：日期时间

如果一列可能包含无法转换的文本，可以先检查或使用：

```python
pd.to_numeric(df["accuracy"], errors="coerce")
pd.to_datetime(df["created_at"], errors="coerce")
```

`errors="coerce"` 会把无法解析的值转成缺失值，因此转换后必须重新统计缺失值。

## 11. 查找缺失值和重复数据

每列缺失数量：

```python
missing_count = df.isna().sum()
missing_rate = df.isna().mean()
```

找到包含缺失值的行：

```python
rows_with_missing = df.loc[df.isna().any(axis=1)]
```

重复行数量：

```python
duplicate_count = df.duplicated().sum()
duplicate_rows = df.loc[df.duplicated(keep=False)]
```

今天只要求识别问题。删除、填充及其依据放到第 12 天。

## 12. 避免链式赋值

下面的写法意图不清楚：

```python
subset = df[df["accuracy"] >= 0.85]
subset["level"] = "high"
```

如果要得到独立子集，显式复制：

```python
subset = df.loc[df["accuracy"] >= 0.85].copy()
subset["level"] = "high"
```

如果要修改原 DataFrame，使用一次 `loc`：

```python
df.loc[df["accuracy"] >= 0.85, "level"] = "high"
```

## 13. 与大模型工程的联系

Pandas 常用于：

- 检查训练语料元数据和标签分布
- 分析 Prompt、模型版本、延迟、成本和评测得分
- 清洗指令微调数据和人工偏好数据
- 汇总 RAG 检索命中率与回答正确率
- 对实验结果进行筛选、排序和分组

Pandas 适合中小规模表格分析。超大数据集通常需要数据库、流式处理或分布式计算工具。

## 14. 自测题

1. `df["model"]` 和 `df[["model"]]` 的返回类型有什么区别？
2. `loc` 与 `iloc` 分别按什么定位？
3. 为什么多个布尔条件需要分别加括号？
4. 为什么读取 CSV 后不能只看 `head()`？
5. `errors="coerce"` 有什么风险？

参考答案：

1. 前者是 Series，后者是只有一列的 DataFrame。
2. `loc` 按标签，`iloc` 按整数位置。
3. Pandas 条件组合需要明确运算优先级。
4. 前几行无法暴露完整的类型、缺失、重复和分布问题。
5. 无法转换的数据会变成缺失值，可能掩盖原始脏数据。

## 15. 易错点

1. 选择多列时忘记使用列名列表。
2. 使用 `and`、`or` 组合 Series 条件。
3. 忘记给每个布尔条件加括号。
4. 把行索引当作业务字段。
5. 依赖 `inplace=True`，导致数据流难以追踪。
6. 修改过滤结果却没有显式 `.copy()`。
7. 转换 dtype 后不检查新增缺失值。
8. 排序后以为原 DataFrame 已被修改。

## 16. 今日验收清单

- [ ] 能解释 Series、DataFrame、行索引和列名
- [ ] 能读取 CSV 并检查形状、列名、类型和描述统计
- [ ] 能使用 `loc`、`iloc` 和布尔条件选择数据
- [ ] 能按多列和不同方向排序
- [ ] 能创建计算列并转换数据类型
- [ ] 能统计缺失值和重复行
- [ ] 能解释为什么要避免链式赋值
- [ ] 配套练习全部显示 `[通过]`

## 17. 官方参考文档

- [Pandas 10 分钟入门](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas 数据选择](https://pandas.pydata.org/docs/user_guide/indexing.html)
- [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
- [Pandas 缺失数据](https://pandas.pydata.org/docs/user_guide/missing_data.html)

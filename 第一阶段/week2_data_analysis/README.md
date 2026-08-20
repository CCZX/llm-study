# 葡萄酒质量数据报告

> 这是第 13～14 天项目的 README 模板。完成项目时请删除本提示，并替换所有 `TODO`。

## 项目目标

TODO：用 2～3 句话说明要分析的问题、数据对象和交付结果。

## 数据来源

- 数据集：[UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine%2Bquality)
- 下载日期：TODO
- 原始文件：`winequality-red.csv`、`winequality-white.csv`
- 原始文件修改：TODO（建议保持未修改）

## 项目结构

```text
week2_data_analysis/
├── README.md
├── wine_analysis.ipynb
├── data/
│   ├── raw/
│   └── processed/
└── figures/
```

## 环境与运行

在仓库根目录执行：

```bash
source .venv/bin/activate
cd 第一阶段/week2_data_analysis
jupyter lab
```

打开 `wine_analysis.ipynb`，重启内核并运行全部单元格。

依赖：

- Python 3.11+
- NumPy
- Pandas
- Matplotlib
- JupyterLab

## 分析流程

1. TODO：数据读取与合并。
2. TODO：数据质量检查与清洗规则。
3. TODO：描述统计与可视化。
4. TODO：训练、验证和测试划分。
5. TODO：只使用训练集统计量标准化。

## 主要发现

1. TODO：发现、证据和边界。
2. TODO：发现、证据和边界。
3. TODO：发现、证据和边界。
4. TODO：发现、证据和边界。
5. TODO：发现、证据和边界。

## 局限

1. TODO：说明一项数据或方法局限及其影响。
2. TODO：说明另一项局限及其影响。

## 可复现性

- 随机种子：`42`
- 数据划分比例：训练集 TODO / 验证集 TODO / 测试集 TODO
- 标准化统计量来源：仅训练集
- Notebook 从头运行状态：TODO

## 参考资料

- [第 13～14 天项目指南](../第13至14天-葡萄酒质量数据报告.md)
- [UCI Wine Quality 数据集](https://archive.ics.uci.edu/dataset/186/wine%2Bquality)

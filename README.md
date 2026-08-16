# LLM 学习笔记

这是一个面向大模型学习的 Python 实践项目。当前内容聚焦 NumPy 数组基础，并结合大模型中常见的 `(batch, sequence, hidden)` 数据形状，练习数组创建、索引、切片、变形和聚合运算。

## 项目内容

```text
.
├── 第一阶段/
│   ├── 第08天-NumPy数组.md          # NumPy 数组讲义
│   └── 第08天-NumPy数组-练习.py     # 带自动检查的练习
└── Untitled.ipynb                   # NumPy 实验 Notebook
```

## 环境要求

- Python 3.11 或更高版本
- NumPy
- JupyterLab（运行 Notebook 时需要）

## 快速开始

在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy jupyterlab
```

Windows PowerShell 使用下面的命令激活虚拟环境：

```powershell
.venv\Scripts\Activate.ps1
```

## 学习方式

建议按以下顺序学习：

1. 阅读 [`第一阶段/第08天-NumPy数组.md`](第一阶段/第08天-NumPy数组.md)，理解 NumPy 数组的核心概念。
2. 打开 `Untitled.ipynb`，运行示例并尝试修改数组形状和索引方式。
3. 完成练习文件中的 `TODO`，将赋值右侧的 `...` 替换为 NumPy 表达式。
4. 运行练习脚本，根据自动检查结果修正答案。

启动 JupyterLab：

```bash
jupyter lab
```

运行练习：

```bash
python "第一阶段/第08天-NumPy数组-练习.py"
```

脚本会为每道题输出 `[通过]`、`[未通过]` 或 `[待完成]`，未填写的题目不会中断后续检查。

## 当前学习目标

- 理解 `ndarray`、`shape`、`ndim`、`size` 和 `dtype`
- 掌握多维数组的创建、索引与切片
- 使用 `reshape` 调整数组形状
- 使用 `sum`、`mean` 和 `max` 完成聚合计算
- 根据 `axis` 推断计算结果的形状
- 理解大模型中常见的 `(batch, sequence, hidden)` 张量结构

## 参考资料

- [NumPy 初学者教程](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy 数组索引](https://numpy.org/doc/stable/user/basics.indexing.html)


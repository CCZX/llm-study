# 第三周 NumPy 机器学习项目

> 这是第 19～21 天项目模板。完成后替换所有 `TODO`。

## 项目目标

只使用 NumPy 实现：

1. Wine Quality 线性回归。
2. Iris Softmax 多分类器。
3. 带一个 ReLU 隐藏层的两层神经网络。

## 项目结构

```text
week3_numpy_ml/
├── README.md
├── data.py
├── losses.py
├── linear_regression.py
├── softmax_classifier.py
├── mlp.py
├── train.py
├── figures/
└── tests/
```

## 环境与运行

```bash
source .venv/bin/activate
pytest -q 第一阶段/week3_numpy_ml/tests
```

TODO：补充三个模型的运行命令。

## 数据与划分

- Wine Quality 来源：TODO
- Iris 来源：TODO
- 随机种子：`42`
- 训练/验证/测试比例：TODO
- 分层策略：TODO
- 标准化统计量来源：仅训练集

## 线性回归

- 训练集均值基线测试 RMSE：TODO
- 模型测试 RMSE：TODO
- 最佳学习率与 epoch：TODO
- 梯度检查相对误差：TODO

## Softmax 分类器

- 多数类基线准确率：TODO
- 最佳验证准确率：TODO
- 测试准确率：TODO
- 混淆矩阵：TODO
- 梯度检查相对误差：TODO

## 两层神经网络

| hidden_dim | learning_rate | 最佳验证准确率 | 最佳 epoch |
| ---: | ---: | ---: | ---: |
| 4 | 0.01 | TODO | TODO |
| 4 | 0.1 | TODO | TODO |
| 16 | 0.01 | TODO | TODO |
| 16 | 0.1 | TODO | TODO |

- 最终测试准确率：TODO
- 梯度检查最大相对误差：TODO

## 主要发现

1. TODO
2. TODO
3. TODO

## 局限与下一步

1. TODO
2. TODO

下一阶段将使用 PyTorch 的 Tensor、自动微分和 `nn.Module` 重写这些模型，并比较手写梯度与自动微分。

## 对应讲义

- [第 19 天：从零实现线性回归](../第19天-从零实现线性回归.md)
- [第 20 天：从零实现 Softmax 分类器](../第20天-从零实现Softmax分类器.md)
- [第 21 天：两层神经网络与项目整理](../第21天-两层神经网络与项目整理.md)

"""第 20 天：从零实现 Softmax 分类器练习。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def check(name: str, actual: Any, expected: Any, atol: float = 1e-7) -> None:
    if actual is Ellipsis:
        print(f"[待完成] {name}")
        return

    try:
        if isinstance(expected, np.ndarray):
            passed = isinstance(actual, np.ndarray) and np.allclose(actual, expected, atol=atol)
        elif isinstance(expected, float):
            passed = bool(np.isclose(actual, expected, atol=atol))
        else:
            passed = actual == expected
    except (TypeError, ValueError):
        passed = False

    print(f"[{'通过' if passed else '未通过'}] {name}")
    if not passed:
        print(f"  你的结果：{actual}")
        print(f"  预期结果：{expected}")


def safe_call(name: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except Exception as error:
        print(f"[未通过] {name} 执行异常：{type(error).__name__}: {error}")
        return Ellipsis


def stratified_split_indices(
    labels: np.ndarray,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐类别划分并分别打乱三个集合。"""
    # TODO：替换下面的 ...。
    return ...


def one_hot(labels: np.ndarray, class_count: int) -> np.ndarray:
    """把类别编号转换成 one-hot 矩阵。"""
    # TODO：替换下面的 ...。
    return ...


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """沿最后一个维度计算稳定 Softmax。"""
    # TODO：替换下面的 ...。
    return ...


def loss_and_gradients(
    x: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """返回平均交叉熵、grad_weight 和 grad_bias。"""
    # TODO：替换下面的 ...。
    return ...


def predict(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """返回每个样本的预测类别编号。"""
    # TODO：无需计算 Softmax，直接利用 logits。
    return ...


def train_classifier(
    x: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    learning_rate: float = 0.1,
    epochs: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """零初始化训练，返回 weight、bias 和每轮更新前的损失。"""
    # TODO：替换下面的 ...。
    return ...


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_count: int,
) -> np.ndarray:
    """行是真实类别，列是预测类别。"""
    # TODO：替换下面的 ...。
    return ...


def numerical_gradient(
    function: Callable[[np.ndarray], float],
    parameters: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    result = np.zeros_like(parameters, dtype=np.float64)
    for index in np.ndindex(parameters.shape):
        positive = parameters.copy()
        negative = parameters.copy()
        positive[index] += h
        negative[index] -= h
        result[index] = (function(positive) - function(negative)) / (2.0 * h)
    return result


def exercise_1_stratified_split() -> None:
    print("\n=== 练习 1：分层划分 ===")
    labels = np.repeat(np.arange(3), 50)
    result = safe_call("分层划分", stratified_split_indices, labels)
    if result is Ellipsis:
        check("分层划分", Ellipsis, "每类按 35/7/8 划分")
        return

    train, valid, test = result
    check("训练集类别计数", np.bincount(labels[train], minlength=3), np.array([35, 35, 35]))
    check("验证集类别计数", np.bincount(labels[valid], minlength=3), np.array([7, 7, 7]))
    check("测试集类别计数", np.bincount(labels[test], minlength=3), np.array([8, 8, 8]))
    check("划分无重复", len(np.unique(np.concatenate([train, valid, test]))), 150)


def exercise_2_one_hot_softmax() -> None:
    print("\n=== 练习 2：one-hot 与 Softmax ===")
    labels = np.array([0, 2, 1])
    targets = safe_call("one-hot", one_hot, labels, 3)
    check("one-hot 编码", targets, np.eye(3)[labels])

    logits = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
    probabilities = safe_call("稳定 Softmax", stable_softmax, logits)
    if probabilities is not Ellipsis:
        check("Softmax 概率和", probabilities.sum(axis=1), np.ones(2))
        check("Softmax 结果有限", bool(np.isfinite(probabilities).all()), True)


def exercise_3_gradients() -> None:
    print("\n=== 练习 3：分类器梯度 ===")
    x = np.array([[1.0, -1.0], [0.5, 2.0], [-1.5, 0.2]])
    labels = np.array([0, 2, 1])
    weight = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, 0.05]])
    bias = np.array([0.01, -0.02, 0.03])

    result = safe_call("Softmax 分类器梯度", loss_and_gradients, x, labels, weight, bias)
    if result is Ellipsis:
        check("Softmax 分类器梯度", Ellipsis, "通过数值检查")
        return

    loss, grad_weight, grad_bias = result
    check("损失为有限正数", bool(np.isfinite(loss) and loss > 0), True)
    check("权重梯度形状", grad_weight.shape, weight.shape)
    check("偏置梯度形状", grad_bias.shape, bias.shape)

    loss_for_weight = lambda candidate: loss_and_gradients(x, labels, candidate, bias)[0]
    numeric = numerical_gradient(loss_for_weight, weight)
    check("权重解析梯度通过数值检查", grad_weight, numeric, atol=1e-5)


def make_cluster_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    centers = np.array([[-2.0, -2.0], [2.0, -1.0], [0.0, 2.5]])
    parts = [center + rng.normal(scale=0.45, size=(80, 2)) for center in centers]
    x = np.vstack(parts)
    labels = np.repeat(np.arange(3), 80)
    order = rng.permutation(len(x))
    return x[order], labels[order]


def exercise_4_training() -> None:
    print("\n=== 练习 4：训练三分类模型 ===")
    x, labels = make_cluster_data()
    result = safe_call(
        "训练 Softmax 分类器",
        train_classifier,
        x,
        labels,
        3,
        learning_rate=0.1,
        epochs=500,
    )
    if result is Ellipsis:
        check("训练 Softmax 分类器", Ellipsis, "准确率超过 95%")
        return

    weight, bias, history = result
    predictions = safe_call("分类预测", predict, x, weight, bias)
    if predictions is Ellipsis:
        check("合成数据准确率", Ellipsis, True)
        return

    accuracy = float(np.mean(predictions == labels))
    check("训练损失下降", bool(history[-1] < history[0]), True)
    check("合成数据准确率", bool(accuracy > 0.95), True)

    matrix = safe_call("混淆矩阵", confusion_matrix, labels, predictions, 3)
    if matrix is Ellipsis:
        check("混淆矩阵", Ellipsis, np.zeros((3, 3), dtype=np.int64))
        return

    check("混淆矩阵形状", matrix.shape, (3, 3))
    check("混淆矩阵样本总数", int(matrix.sum()), len(labels))


def main() -> None:
    print("第 20 天：从零实现 Softmax 分类器练习")
    exercise_1_stratified_split()
    exercise_2_one_hot_softmax()
    exercise_3_gradients()
    exercise_4_training()


if __name__ == "__main__":
    main()

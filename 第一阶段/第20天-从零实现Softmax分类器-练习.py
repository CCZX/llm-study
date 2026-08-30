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
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels 必须是一维数组")
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("train_ratio 必须在 [0, 1] 范围内")
    if not 0.0 <= valid_ratio <= 1.0:
        raise ValueError("valid_ratio 必须在 [0, 1] 范围内")
    if train_ratio + valid_ratio > 1.0:
        raise ValueError("train_ratio 与 valid_ratio 之和不能超过 1")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)

        train_end = int(len(class_indices) * train_ratio)
        valid_end = train_end + int(len(class_indices) * valid_ratio)
        train_parts.append(class_indices[:train_end])
        valid_parts.append(class_indices[train_end:valid_end])
        test_parts.append(class_indices[valid_end:])

    if train_parts:
        train = np.concatenate(train_parts)
        valid = np.concatenate(valid_parts)
        test = np.concatenate(test_parts)
    else:
        train = np.empty(0, dtype=np.int64)
        valid = np.empty(0, dtype=np.int64)
        test = np.empty(0, dtype=np.int64)

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test


def one_hot(labels: np.ndarray, class_count: int) -> np.ndarray:
    """把类别编号转换成 one-hot 矩阵。"""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels 必须是一维数组")
    if not isinstance(class_count, (int, np.integer)) or class_count <= 0:
        raise ValueError("class_count 必须是正整数")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels 必须包含整数类别编号")
    if labels.size and (labels.min() < 0 or labels.max() >= class_count):
        raise ValueError("labels 中的类别编号超出有效范围")

    result = np.zeros((len(labels), class_count), dtype=np.float64)
    result[np.arange(len(labels)), labels] = 1.0
    return result


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """沿最后一个维度计算稳定 Softmax。"""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 0 or logits.shape[-1] == 0:
        raise ValueError("logits 的最后一个维度必须包含至少一个类别")

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def loss_and_gradients(
    x: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """返回平均交叉熵、grad_weight 和 grad_bias。"""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)

    if x.ndim != 2 or weight.ndim != 2 or bias.ndim != 1:
        raise ValueError("x、weight 和 bias 的维度必须分别为 2、2 和 1")
    if len(x) == 0:
        raise ValueError("x 不能为空")
    if labels.ndim != 1 or len(labels) != len(x):
        raise ValueError("labels 必须是与 x 样本数相同的一维数组")
    if x.shape[1] != weight.shape[0] or weight.shape[1] != len(bias):
        raise ValueError("x、weight 与 bias 的形状不匹配")

    targets = one_hot(labels, weight.shape[1])
    logits = x @ weight + bias
    probabilities = stable_softmax(logits)

    # log(sum(exp(shifted))) - shifted[target] 比先取概率再取对数更稳定。
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_normalizers = np.log(np.exp(shifted).sum(axis=1))
    correct_logits = shifted[np.arange(len(labels)), labels]
    loss = float(np.mean(log_normalizers - correct_logits))

    grad_logits = (probabilities - targets) / len(x)
    grad_weight = x.T @ grad_logits
    grad_bias = np.sum(grad_logits, axis=0)
    return loss, grad_weight, grad_bias


def predict(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """返回每个样本的预测类别编号。"""
    logits = np.asarray(x) @ np.asarray(weight) + np.asarray(bias)
    return np.argmax(logits, axis=1)


def train_classifier(
    x: np.ndarray,
    labels: np.ndarray,
    class_count: int,
    learning_rate: float = 0.1,
    epochs: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """零初始化训练，返回 weight、bias 和每轮更新前的损失。"""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    if x.ndim != 2:
        raise ValueError("x 必须是二维数组")
    if labels.ndim != 1 or len(labels) != len(x):
        raise ValueError("labels 必须是与 x 样本数相同的一维数组")
    if len(x) == 0:
        raise ValueError("训练数据不能为空")
    if not isinstance(class_count, (int, np.integer)) or class_count <= 0:
        raise ValueError("class_count 必须是正整数")
    if not isinstance(epochs, (int, np.integer)) or epochs < 0:
        raise ValueError("epochs 必须是非负整数")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate 必须是有限正数")

    # 在训练开始前验证标签范围，避免部分更新后才报错。
    one_hot(labels, class_count)
    weight = np.zeros((x.shape[1], class_count), dtype=np.float64)
    bias = np.zeros(class_count, dtype=np.float64)
    history = np.empty(epochs, dtype=np.float64)

    for epoch in range(epochs):
        loss, grad_weight, grad_bias = loss_and_gradients(x, labels, weight, bias)
        history[epoch] = loss
        weight -= learning_rate * grad_weight
        bias -= learning_rate * grad_bias

    return weight, bias, history


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_count: int,
) -> np.ndarray:
    """行是真实类别，列是预测类别。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("y_true 与 y_pred 必须是等长的一维数组")
    if not isinstance(class_count, (int, np.integer)) or class_count <= 0:
        raise ValueError("class_count 必须是正整数")
    if not np.issubdtype(y_true.dtype, np.integer) or not np.issubdtype(
        y_pred.dtype, np.integer
    ):
        raise TypeError("y_true 与 y_pred 必须包含整数类别编号")
    if y_true.size and (
        y_true.min() < 0
        or y_pred.min() < 0
        or y_true.max() >= class_count
        or y_pred.max() >= class_count
    ):
        raise ValueError("类别编号超出有效范围")

    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


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

"""第 21 天：两层神经网络与项目整理练习。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


Parameters = dict[str, np.ndarray]


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


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU 激活。"""
    # TODO：替换下面的 ...。
    return ...


def relu_gradient(x: np.ndarray) -> np.ndarray:
    """ReLU 对输入的局部梯度，x==0 时取 0。"""
    # TODO：替换下面的 ...。
    return ...


def initialize_parameters(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int = 42,
) -> Parameters:
    """使用 He 缩放随机初始化 w1、w2，偏置初始化为零。"""
    # TODO：替换下面的 ...。
    return ...


def forward(x: np.ndarray, parameters: Parameters) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """返回 logits 和包含 x、z1、hidden 的缓存。"""
    # TODO：替换下面的 ...。
    return ...


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def loss_and_gradients(
    x: np.ndarray,
    labels: np.ndarray,
    parameters: Parameters,
) -> tuple[float, Parameters]:
    """返回平均交叉熵和 w1、b1、w2、b2 四组梯度。"""
    # TODO：替换下面的 ...。
    return ...


def predict(x: np.ndarray, parameters: Parameters) -> np.ndarray:
    """返回预测类别编号。"""
    # TODO：替换下面的 ...。
    return ...


def train_mlp(
    x: np.ndarray,
    labels: np.ndarray,
    hidden_dim: int = 8,
    learning_rate: float = 0.1,
    epochs: int = 2000,
    seed: int = 42,
) -> tuple[Parameters, np.ndarray]:
    """训练两层网络，返回参数和每轮更新前的损失。"""
    # TODO：替换下面的 ...。
    return ...


def numerical_parameter_gradient(
    loss_function: Callable[[Parameters], float],
    parameters: Parameters,
    parameter_name: str,
    h: float = 1e-5,
) -> np.ndarray:
    """对参数字典中的一个数组执行中心差分。"""
    target = parameters[parameter_name]
    result = np.zeros_like(target, dtype=np.float64)

    for index in np.ndindex(target.shape):
        positive = {name: value.copy() for name, value in parameters.items()}
        negative = {name: value.copy() for name, value in parameters.items()}
        positive[parameter_name][index] += h
        negative[parameter_name][index] -= h
        result[index] = (loss_function(positive) - loss_function(negative)) / (2.0 * h)

    return result


def exercise_1_relu() -> None:
    print("\n=== 练习 1：ReLU ===")
    x = np.array([-2.0, 0.0, 3.0])
    check("ReLU 前向", relu(x), np.array([0.0, 0.0, 3.0]))
    check("ReLU 梯度", relu_gradient(x), np.array([0.0, 0.0, 1.0]))


def exercise_2_initialization_and_forward() -> None:
    print("\n=== 练习 2：初始化与前向传播 ===")
    parameters = safe_call("参数初始化", initialize_parameters, 2, 4, 3, 42)
    if parameters is Ellipsis:
        check("参数初始化", Ellipsis, "四组正确形状参数")
        return

    expected_shapes = {"w1": (2, 4), "b1": (4,), "w2": (4, 3), "b2": (3,)}
    check("参数键", set(parameters), set(expected_shapes))
    for name, shape in expected_shapes.items():
        check(f"{name} 形状", parameters[name].shape, shape)
    check("权重不是全零", bool(np.any(parameters["w1"] != 0)), True)

    x = np.array([[1.0, -1.0], [0.5, 2.0], [-1.0, 0.3]])
    result = safe_call("前向传播", forward, x, parameters)
    if result is not Ellipsis:
        logits, cache = result
        check("logits 形状", logits.shape, (3, 3))
        check("隐藏层形状", cache["hidden"].shape, (3, 4))


def exercise_3_gradient_check() -> None:
    print("\n=== 练习 3：全部参数梯度检查 ===")
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 2)) + 0.13
    labels = np.array([0, 1, 2, 1])
    parameters = initialize_parameters(2, 3, 3, seed=9)

    if parameters is Ellipsis:
        check("全部参数梯度检查", Ellipsis, True)
        return

    result = safe_call("解析反向传播", loss_and_gradients, x, labels, parameters)
    if result is Ellipsis:
        check("全部参数梯度检查", Ellipsis, True)
        return

    loss, gradients = result
    check("损失有限", bool(np.isfinite(loss)), True)

    loss_function = lambda candidate: loss_and_gradients(x, labels, candidate)[0]
    for name in ["w1", "b1", "w2", "b2"]:
        numeric = numerical_parameter_gradient(loss_function, parameters, name)
        denominator = np.linalg.norm(gradients[name]) + np.linalg.norm(numeric) + 1e-12
        relative_error = np.linalg.norm(gradients[name] - numeric) / denominator
        check(f"{name} 相对误差小于 1e-5", bool(relative_error < 1e-5), True)


def make_xor_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    base_x = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    base_y = np.array([0, 1, 1, 0])
    x = np.repeat(base_x, 50, axis=0) + rng.normal(scale=0.12, size=(200, 2))
    labels = np.repeat(base_y, 50)
    order = rng.permutation(len(x))
    return x[order], labels[order]


def exercise_4_training_xor() -> None:
    print("\n=== 练习 4：拟合非线性 XOR ===")
    x, labels = make_xor_data()
    result = safe_call(
        "训练两层网络",
        train_mlp,
        x,
        labels,
        hidden_dim=8,
        learning_rate=0.1,
        epochs=2500,
        seed=42,
    )
    if result is Ellipsis:
        check("训练两层网络", Ellipsis, "XOR 准确率超过 95%")
        return

    parameters, history = result
    predictions = safe_call("MLP 分类预测", predict, x, parameters)
    if predictions is Ellipsis:
        check("XOR 训练准确率", Ellipsis, True)
        return

    accuracy = float(np.mean(predictions == labels))
    check("损失历史形状", history.shape, (2500,))
    check("训练损失显著下降", bool(history[-1] < history[0] * 0.25), True)
    check("XOR 训练准确率", bool(accuracy > 0.95), True)


def main() -> None:
    print("第 21 天：两层神经网络与项目整理练习")
    exercise_1_relu()
    exercise_2_initialization_and_forward()
    exercise_3_gradient_check()
    exercise_4_training_xor()


if __name__ == "__main__":
    main()

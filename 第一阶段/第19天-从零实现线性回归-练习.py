"""第 19 天：从零实现线性回归练习。"""

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


def predict(x: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    """线性模型前向计算。"""
    # TODO：替换下面的 ...。
    return ...


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方误差。"""
    # TODO：替换下面的 ...。
    return ...


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差。"""
    # TODO：调用 mse 并开平方。
    return ...


def gradients(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    bias: float,
) -> tuple[np.ndarray, float]:
    """返回 MSE 对 weight 和 bias 的解析梯度。"""
    # TODO：替换下面的 ...。
    return ...


def train_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    epochs: int = 500,
) -> tuple[np.ndarray, float, np.ndarray]:
    """从零初始化并训练，返回 weight、bias 和每轮更新后的 MSE。"""
    # TODO：替换下面的 ...。
    return ...


def numerical_gradient(
    function: Callable[[np.ndarray], float],
    parameters: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    result = np.zeros_like(parameters, dtype=np.float64)
    for index in range(parameters.size):
        positive = parameters.copy()
        negative = parameters.copy()
        positive[index] += h
        negative[index] -= h
        result[index] = (function(positive) - function(negative)) / (2.0 * h)
    return result


def exercise_1_forward_and_losses() -> None:
    print("\n=== 练习 1：前向计算与损失 ===")
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    weight = np.array([0.5, -1.0])
    bias = 2.0
    y = np.array([0.5, -0.5])

    predictions = safe_call("线性预测", predict, x, weight, bias)
    check("线性预测", predictions, np.array([0.5, -0.5]))
    if predictions is not Ellipsis:
        check("完美预测 MSE", mse(y, predictions), 0.0)
        check("完美预测 RMSE", rmse(y, predictions), 0.0)


def exercise_2_gradients() -> None:
    print("\n=== 练习 2：解析梯度 ===")
    x = np.array([[1.0, 2.0], [2.0, -1.0], [0.5, 3.0]])
    y = np.array([2.0, -1.0, 3.0])
    weight = np.array([0.2, -0.3])
    bias = 0.1

    result = safe_call("解析梯度", gradients, x, y, weight, bias)
    if result is Ellipsis:
        check("解析梯度", Ellipsis, np.zeros(2))
        return

    grad_weight, grad_bias = result
    errors = x @ weight + bias - y
    expected_weight = (2.0 / len(x)) * (x.T @ errors)
    expected_bias = 2.0 * errors.mean()
    check("权重梯度", grad_weight, expected_weight)
    check("偏置梯度", grad_bias, float(expected_bias))

    loss_for_weight = lambda candidate: float(np.mean((x @ candidate + bias - y) ** 2))
    numeric = numerical_gradient(loss_for_weight, weight)
    check("解析梯度通过数值检查", grad_weight, numeric, atol=1e-5)


def exercise_3_training() -> None:
    print("\n=== 练习 3：训练合成线性数据 ===")
    rng = np.random.default_rng(42)
    x = rng.standard_normal((200, 3))
    true_weight = np.array([2.0, -1.0, 0.5])
    true_bias = 1.5
    y = x @ true_weight + true_bias

    result = safe_call(
        "训练线性回归",
        train_linear_regression,
        x,
        y,
        learning_rate=0.05,
        epochs=500,
    )
    if result is Ellipsis:
        check("训练线性回归", Ellipsis, "恢复真实参数")
        return

    weight, bias, history = result
    check("损失历史形状", history.shape, (500,))
    check("损失显著下降", bool(history[-1] < history[0] * 1e-4), True)
    check("恢复权重", weight, true_weight, atol=1e-3)
    check("恢复偏置", bias, true_bias, atol=1e-3)


def exercise_4_baseline() -> None:
    print("\n=== 练习 4：与基线比较 ===")
    x_train = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    y_train = 3.0 * x_train[:, 0] + 2.0
    x_test = np.array([[-1.5], [0.5], [2.5]])
    y_test = 3.0 * x_test[:, 0] + 2.0

    result = safe_call(
        "训练基线对照模型",
        train_linear_regression,
        x_train,
        y_train,
        learning_rate=0.05,
        epochs=500,
    )
    if result is Ellipsis:
        check("模型超过基线", Ellipsis, True)
        return

    weight, bias, _ = result
    model_predictions = safe_call("测试集预测", predict, x_test, weight, bias)
    baseline_predictions = np.full_like(y_test, y_train.mean())

    if model_predictions is Ellipsis:
        check("模型测试 RMSE 小于基线", Ellipsis, True)
        return

    model_rmse = safe_call("模型测试 RMSE", rmse, y_test, model_predictions)
    baseline_rmse = safe_call("基线测试 RMSE", rmse, y_test, baseline_predictions)
    if model_rmse is Ellipsis or baseline_rmse is Ellipsis:
        check("模型测试 RMSE 小于基线", Ellipsis, True)
        return

    check("模型测试 RMSE 小于基线", model_rmse < baseline_rmse, True)


def main() -> None:
    print("第 19 天：从零实现线性回归练习")
    exercise_1_forward_and_losses()
    exercise_2_gradients()
    exercise_3_training()
    exercise_4_baseline()


if __name__ == "__main__":
    main()

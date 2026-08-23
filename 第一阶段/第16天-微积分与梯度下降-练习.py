"""第 16 天：微积分与梯度下降练习。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def check(name: str, actual: Any, expected: Any) -> None:
    if actual is Ellipsis:
        print(f"[待完成] {name}")
        return

    try:
        if isinstance(expected, np.ndarray):
            passed = isinstance(actual, np.ndarray) and np.allclose(
                actual, expected, rtol=1e-5, atol=1e-7
            )
        elif isinstance(expected, float):
            passed = bool(np.isclose(actual, expected, rtol=1e-5, atol=1e-7))
        else:
            passed = actual == expected
    except (TypeError, ValueError):
        passed = False

    print(f"[{'通过' if passed else '未通过'}] {name}")
    if not passed:
        print(f"  你的结果：{actual}")
        print(f"  预期结果：{expected}")


def safe_call(name: str, function: Callable[..., Any], *args: Any) -> Any:
    try:
        return function(*args)
    except Exception as error:
        print(f"[未通过] {name} 执行异常：{type(error).__name__}: {error}")
        return Ellipsis


def central_difference(
    function: Callable[[float], float],
    x: float,
    h: float = 1e-5,
) -> float:
    """使用中心差分近似一元函数在 x 处的导数。"""
    # TODO：替换下面的 ...。
    return ...


def gradient_descent(
    gradient_function: Callable[[float], float],
    initial_x: float,
    learning_rate: float,
    steps: int,
) -> tuple[float, np.ndarray]:
    """执行梯度下降，返回最终 x 和包含初始值的完整轨迹。"""
    # TODO：替换下面的 ...。
    return ...


def linear_mse_gradients(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    bias: float,
) -> tuple[np.ndarray, float]:
    """返回线性模型 MSE 对 weight 和 bias 的解析梯度。"""
    # TODO：替换下面的 ...。
    return ...


def numerical_gradient(
    function: Callable[[np.ndarray], float],
    parameters: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """用于自动检查的小规模数值梯度。"""
    result = np.zeros_like(parameters, dtype=np.float64)
    for index in range(parameters.size):
        positive = parameters.copy()
        negative = parameters.copy()
        positive[index] += h
        negative[index] -= h
        result[index] = (function(positive) - function(negative)) / (2.0 * h)
    return result


def exercise_1_derivatives() -> None:
    print("\n=== 练习 1：解析导数 ===")
    x = 2.0

    # TODO：计算 f(x)=x² 在 x=2 处的导数。
    square_gradient = ...
    # TODO：计算 f(x)=(3x+1)² 在 x=2 处的导数。
    chain_gradient = ...

    check("x² 的导数", square_gradient, 4.0)
    check("链式法则", chain_gradient, 42.0)


def exercise_2_numerical_derivative() -> None:
    print("\n=== 练习 2：数值导数 ===")
    function = lambda value: (value - 3.0) ** 2
    actual = safe_call("中心差分", central_difference, function, 5.0)
    check("中心差分接近解析导数", actual, 4.0)


def exercise_3_gradient_descent() -> None:
    print("\n=== 练习 3：梯度下降 ===")
    gradient = lambda value: 2.0 * (value - 3.0)
    result = safe_call("梯度下降", gradient_descent, gradient, 0.0, 0.1, 30)

    if result is Ellipsis:
        check("梯度下降到最小值附近", Ellipsis, 3.0)
        return

    final_x, history = result
    check("轨迹包含初始值和 30 次更新", history.shape, (31,))
    check("轨迹初始值", float(history[0]), 0.0)
    check("最终接近最小值", bool(abs(final_x - 3.0) < 0.01), True)


def exercise_4_learning_rates() -> None:
    print("\n=== 练习 4：学习率影响 ===")
    gradient = lambda value: 2.0 * (value - 3.0)
    small = safe_call("小学习率", gradient_descent, gradient, 0.0, 0.001, 20)
    good = safe_call("合适学习率", gradient_descent, gradient, 0.0, 0.1, 20)
    large = safe_call("大学习率", gradient_descent, gradient, 0.0, 1.1, 20)

    if Ellipsis in (small, good, large):
        check("学习率比较", Ellipsis, "合适学习率收敛最快")
        return

    small_x = small[0]
    good_x = good[0]
    large_x = large[0]
    check("合适学习率比小学习率更接近最小值", abs(good_x - 3) < abs(small_x - 3), True)
    check("过大学习率发散", abs(large_x - 3) > 3.0, True)


def exercise_5_linear_gradients() -> None:
    print("\n=== 练习 5：线性回归解析梯度 ===")
    x = np.array([[1.0, 2.0], [2.0, -1.0], [0.5, 3.0]])
    y = np.array([2.0, -1.0, 3.0])
    weight = np.array([0.2, -0.3])
    bias = 0.1

    result = safe_call("线性回归梯度", linear_mse_gradients, x, y, weight, bias)
    if result is Ellipsis:
        check("权重梯度", Ellipsis, np.zeros(2))
        return

    grad_weight, grad_bias = result
    errors = x @ weight + bias - y
    expected_weight = (2.0 / len(x)) * (x.T @ errors)
    expected_bias = 2.0 * errors.mean()
    check("权重梯度", grad_weight, expected_weight)
    check("偏置梯度", grad_bias, float(expected_bias))

    loss_for_weight = lambda candidate: float(np.mean((x @ candidate + bias - y) ** 2))
    numeric_weight = numerical_gradient(loss_for_weight, weight)
    check("解析梯度与数值梯度一致", grad_weight, numeric_weight)


def main() -> None:
    print("第 16 天：微积分与梯度下降练习")
    exercise_1_derivatives()
    exercise_2_numerical_derivative()
    exercise_3_gradient_descent()
    exercise_4_learning_rates()
    exercise_5_linear_gradients()


if __name__ == "__main__":
    main()

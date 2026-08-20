"""第 10 天：数值函数与数值稳定性练习。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def check(name: str, actual: Any, expected: Any) -> None:
    """比较结果并继续执行后续检查。"""
    if actual is Ellipsis:
        print(f"[待完成] {name}")
        return

    try:
        if isinstance(expected, np.ndarray):
            passed = isinstance(actual, np.ndarray) and np.allclose(
                actual, expected, rtol=1e-6, atol=1e-9
            )
        elif isinstance(expected, float):
            passed = bool(np.isclose(actual, expected, rtol=1e-6, atol=1e-9))
        else:
            passed = actual == expected
    except (TypeError, ValueError):
        passed = False

    print(f"[{'通过' if passed else '未通过'}] {name}")
    if not passed:
        print(f"  你的结果：{actual}")
        print(f"  预期结果：{expected}")


def safe_call(name: str, function: Callable[..., Any], *args: Any) -> Any:
    """未完成的函数返回 Ellipsis；其他异常会显示但不打断练习。"""
    try:
        result = function(*args)
    except Exception as error:  # 练习脚本需要继续检查后续函数
        print(f"[未通过] {name} 执行时发生异常：{type(error).__name__}: {error}")
        return Ellipsis
    return result


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """返回数值稳定的 Sigmoid 结果。"""
    # TODO：替换下面的 ...，极端输入不能产生 NaN。
    return ...


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """沿最后一个维度计算数值稳定的 Softmax。"""
    # TODO：替换下面的 ...；先逐行减最大值，再计算指数与概率。
    return ...


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算所有元素的均方误差。"""
    # TODO：替换下面的 ...。
    return ...


def cross_entropy(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """计算 one-hot 标签的平均交叉熵。"""
    # TODO：替换下面的 ...；使用 clip 避免 log(0)。
    return ...


def exercise_1_statistics() -> None:
    print("\n=== 练习 1：均值、方差和范数 ===")
    x = np.array([[3.0, 4.0], [5.0, 12.0]])

    # TODO：计算全部元素均值。
    mean = ...
    # TODO：沿 axis=1 计算每行的总体方差。
    row_variance = ...
    # TODO：沿 axis=1 计算每行的 L2 范数。
    row_norm = ...

    check("全部元素均值", mean, 6.0)
    check("每行总体方差", row_variance, np.array([0.25, 12.25]))
    check("每行 L2 范数", row_norm, np.array([5.0, 13.0]))


def exercise_2_sigmoid() -> None:
    print("\n=== 练习 2：稳定 Sigmoid ===")
    x = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
    actual = safe_call("稳定 Sigmoid", stable_sigmoid, x)

    expected = np.array([0.0, 0.11920292, 0.5, 0.88079708, 1.0])
    check("稳定 Sigmoid 数值", actual, expected)
    if isinstance(actual, np.ndarray):
        check("Sigmoid 输出全部有限", bool(np.isfinite(actual).all()), True)
        check("Sigmoid 输出单调不减", bool(np.all(np.diff(actual) >= 0)), True)


def exercise_3_softmax() -> None:
    print("\n=== 练习 3：稳定 Softmax ===")
    logits = np.array(
        [
            [1000.0, 1001.0, 1002.0],
            [-1000.0, -1001.0, -1002.0],
        ]
    )
    actual = safe_call("稳定 Softmax", stable_softmax, logits)

    expected = np.array(
        [
            [0.09003057, 0.24472847, 0.66524096],
            [0.66524096, 0.24472847, 0.09003057],
        ]
    )
    check("稳定 Softmax 数值", actual, expected)
    if isinstance(actual, np.ndarray):
        check("Softmax 每行之和", actual.sum(axis=-1), np.ones(2))
        shifted = safe_call("Softmax 平移测试", stable_softmax, logits + 12345.0)
        check("Softmax 平移不变性", shifted, actual)
        check("Softmax 输出全部有限", bool(np.isfinite(actual).all()), True)


def exercise_4_mse() -> None:
    print("\n=== 练习 4：MSE ===")
    y_true = np.array([1.0, 2.0, 3.0])
    perfect = safe_call("完全正确的 MSE", mse, y_true, y_true.copy())
    imperfect = safe_call("存在误差的 MSE", mse, y_true, np.array([0.0, 2.0, 5.0]))

    check("完全正确时 MSE 为 0", perfect, 0.0)
    check("均方误差计算", imperfect, 5.0 / 3.0)


def exercise_5_cross_entropy() -> None:
    print("\n=== 练习 5：交叉熵 ===")
    labels = np.array([[1.0, 0.0], [0.0, 1.0]])
    good_probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])
    bad_probabilities = np.array([[0.1, 0.9], [0.8, 0.2]])
    zero_probability = np.array([[0.0, 1.0], [1.0, 0.0]])

    good_loss = safe_call("较好预测的交叉熵", cross_entropy, labels, good_probabilities)
    bad_loss = safe_call("较差预测的交叉熵", cross_entropy, labels, bad_probabilities)
    finite_loss = safe_call("零概率交叉熵", cross_entropy, labels, zero_probability)

    check("较好预测的交叉熵", good_loss, float((-np.log(0.9) - np.log(0.8)) / 2))
    if good_loss is not Ellipsis and bad_loss is not Ellipsis:
        check("正确类别概率越低损失越大", bool(bad_loss > good_loss), True)
    if finite_loss is not Ellipsis:
        check("零概率经过保护后仍为有限值", bool(np.isfinite(finite_loss)), True)


def exercise_6_float_comparison() -> None:
    print("\n=== 练习 6：浮点比较 ===")
    value = 0.1 + 0.2

    # TODO：使用 NumPy 函数判断 value 是否近似等于 0.3。
    approximately_equal = ...

    check("浮点数近似比较", approximately_equal, True)


def main() -> None:
    print("第 10 天：数值函数与数值稳定性练习")
    print("完成所有 TODO 后，确认每项都显示 [通过]。")
    exercise_1_statistics()
    exercise_2_sigmoid()
    exercise_3_softmax()
    exercise_4_mse()
    exercise_5_cross_entropy()
    exercise_6_float_comparison()


if __name__ == "__main__":
    main()

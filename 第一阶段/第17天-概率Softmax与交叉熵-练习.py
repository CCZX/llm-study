"""第 17 天：概率、Softmax 与交叉熵练习。"""

from __future__ import annotations

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


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """沿最后一个维度计算数值稳定的 Softmax。"""
    # TODO：替换下面的 ...。
    return ...


def categorical_cross_entropy(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """labels 是类别编号，返回批次平均交叉熵。"""
    # TODO：取出正确类别概率，使用 clip 保护后计算负对数均值。
    return ...


def bernoulli_log_likelihood(successes: int, trials: int, probability: float) -> float:
    """忽略组合系数，返回伯努利观测的对数似然。"""
    # TODO：实现 successes*log(p) + failures*log(1-p)。
    return ...


def exercise_1_expectation_variance() -> None:
    print("\n=== 练习 1：期望与方差 ===")
    values = np.arange(1, 7, dtype=np.float64)
    probabilities = np.full(6, 1.0 / 6.0)

    # TODO：按概率加权计算期望。
    expectation = ...
    # TODO：按概率加权计算方差。
    variance = ...

    check("公平骰子期望", expectation, 3.5)
    check("公平骰子方差", variance, 35.0 / 12.0)


def exercise_2_conditional_probability() -> None:
    print("\n=== 练习 2：条件概率 ===")
    outcomes = np.arange(1, 7)
    greater_than_three = outcomes > 3
    even = outcomes % 2 == 0

    # TODO：计算 P(点数>3 | 点数为偶数)。
    conditional = ...
    # TODO：计算两个事件的联合概率。
    joint = ...

    check("条件概率", conditional, 2.0 / 3.0)
    check("联合概率", joint, 1.0 / 3.0)


def exercise_3_independence() -> None:
    print("\n=== 练习 3：独立性 ===")
    # 同时投掷两枚公平硬币，0 表示反面，1 表示正面。
    outcomes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    first_head = outcomes[:, 0] == 1
    second_head = outcomes[:, 1] == 1

    # TODO：验证 P(A∩B) 是否等于 P(A)P(B)，返回 bool。
    independent = ...
    check("两枚公平硬币结果独立", independent, True)


def exercise_4_simulation() -> None:
    print("\n=== 练习 4：骰子模拟 ===")
    rng = np.random.default_rng(42)

    # TODO：模拟 200000 次公平骰子，取值范围为 1 到 6。
    rolls = ...

    if rolls is Ellipsis:
        check("骰子经验频率", Ellipsis, np.full(6, 1 / 6))
        return

    frequencies = np.bincount(rolls, minlength=7)[1:] / len(rolls)
    check("模拟结果形状", rolls.shape, (200_000,))
    check("骰子经验频率接近理论值", frequencies, np.full(6, 1 / 6), atol=0.003)


def exercise_5_maximum_likelihood() -> None:
    print("\n=== 练习 5：最大似然 ===")
    candidate_probabilities = np.array([0.3, 0.5, 0.7, 0.9])

    losses = np.array([
        bernoulli_log_likelihood(7, 10, probability)
        for probability in candidate_probabilities
    ])

    if any(value is Ellipsis for value in losses):
        check("最大似然参数", Ellipsis, 0.7)
        return

    best_probability = candidate_probabilities[np.argmax(losses)]
    check("7 次成功/10 次试验的最大似然参数", float(best_probability), 0.7)


def exercise_6_softmax_cross_entropy() -> None:
    print("\n=== 练习 6：Softmax 与交叉熵 ===")
    logits = np.array([[1000.0, 1001.0, 1002.0], [-3.0, 0.0, 3.0]])
    labels = np.array([2, 1])

    probabilities = stable_softmax(logits)
    if probabilities is Ellipsis:
        check("稳定 Softmax", Ellipsis, np.zeros_like(logits))
        return

    loss = categorical_cross_entropy(labels, probabilities)
    shifted_probabilities = stable_softmax(logits + 12345.0)

    check("Softmax 每行概率和", probabilities.sum(axis=-1), np.ones(2))
    check("Softmax 平移不变性", shifted_probabilities, probabilities)
    expected_loss = float(-np.log(probabilities[np.arange(2), labels]).mean())
    check("类别编号交叉熵", loss, expected_loss)


def main() -> None:
    print("第 17 天：概率、Softmax 与交叉熵练习")
    exercise_1_expectation_variance()
    exercise_2_conditional_probability()
    exercise_3_independence()
    exercise_4_simulation()
    exercise_5_maximum_likelihood()
    exercise_6_softmax_cross_entropy()


if __name__ == "__main__":
    main()

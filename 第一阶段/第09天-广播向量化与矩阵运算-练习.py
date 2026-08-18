"""第 9 天：广播、向量化与矩阵运算练习。

使用方法：
1. 激活项目虚拟环境。
2. 将每道题中赋值右侧的 ``...`` 替换成 NumPy 表达式。
3. 运行本文件，程序会自动检查答案。

运行命令：
    python 第一阶段/第09天-广播向量化与矩阵运算-练习.py
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np


def check(name: str, actual: Any, expected: Any) -> None:
    """打印检查结果；未填写或形状错误时，继续检查后续题目。"""
    if actual is Ellipsis:
        print(f"[待完成] {name}")
        return

    try:
        if isinstance(expected, np.ndarray):
            passed = isinstance(actual, np.ndarray) and np.allclose(
                actual,
                expected,
                rtol=1e-6,
                atol=1e-7,
            )
        elif isinstance(expected, float):
            passed = bool(np.isclose(actual, expected, rtol=1e-6, atol=1e-7))
        else:
            passed = actual == expected
    except (TypeError, ValueError):
        passed = False

    if passed:
        print(f"[通过] {name}")
    else:
        print(f"[未通过] {name}")
        print(f"  你的结果：{actual}")
        print(f"  预期结果：{expected}")


def exercise_1_element_wise_operations() -> None:
    """练习 1：区分逐元素运算和点积。"""
    print("\n=== 练习 1：逐元素运算与点积 ===")

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])

    # TODO：计算 a 和 b 的逐元素乘积，结果形状应为 (3,)。
    element_wise_product = ...

    # TODO：使用 @ 计算 a 和 b 的点积，结果应为标量 140。
    dot_product = ...

    check("逐元素乘积", element_wise_product, np.array([10.0, 40.0, 90.0]))
    check("向量点积", dot_product, 140.0)


def exercise_2_broadcasting() -> None:
    """练习 2：使用广播分别调整矩阵的列和行。"""
    print("\n=== 练习 2：广播 ===")

    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    column_offset = np.array([10.0, 20.0, 30.0])
    row_offset = np.array([100.0, 200.0])

    # TODO：把 column_offset 加到 x 的每一行。
    add_to_columns = ...

    # TODO：为 row_offset 增加一个维度，再把它加到 x 的每一列。
    add_to_rows = ...

    check(
        "为每一列增加不同偏移",
        add_to_columns,
        np.array([[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]),
    )
    check(
        "为每一行增加不同偏移",
        add_to_rows,
        np.array([[101.0, 102.0, 103.0], [204.0, 205.0, 206.0]]),
    )


def exercise_3_standardization() -> None:
    """练习 3：使用广播对每一列标准化。"""
    print("\n=== 练习 3：逐列标准化 ===")

    x = np.array(
        [
            [1.0, 10.0, 5.0],
            [2.0, 20.0, 5.0],
            [3.0, 30.0, 5.0],
        ]
    )

    # TODO：沿 axis=0 计算每列均值，并用 keepdims=True 保留维度。
    column_mean = ...

    # TODO：沿 axis=0 计算每列标准差，并保留维度。
    column_std = ...

    # TODO：把标准差为 0 的位置替换成 1.0，其他位置保持不变。
    safe_std = ...

    # TODO：使用 column_mean 和 safe_std 对 x 的每一列进行标准化。
    standardized = ...

    check(
        "每列均值的形状",
        Ellipsis if column_mean is Ellipsis else column_mean.shape,
        (1, 3),
    )
    check(
        "每列标准差的形状",
        Ellipsis if column_std is Ellipsis else column_std.shape,
        (1, 3),
    )
    check(
        "安全标准差",
        safe_std,
        np.array([[np.sqrt(2.0 / 3.0), np.sqrt(200.0 / 3.0), 1.0]]),
    )

    expected = np.array(
        [
            [-1.22474487, -1.22474487, 0.0],
            [0.0, 0.0, 0.0],
            [1.22474487, 1.22474487, 0.0],
        ]
    )
    check("逐列标准化结果", standardized, expected)

    if isinstance(standardized, np.ndarray):
        check("标准化后的每列均值", standardized.mean(axis=0), np.zeros(3))


def exercise_4_matrix_multiplication() -> None:
    """练习 4：矩阵乘法、转置和偏置广播。"""
    print("\n=== 练习 4：矩阵乘法 ===")

    x = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    weight = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    bias = np.array([0.5, -0.5])

    # TODO：使用 @ 计算 x 与 weight 的矩阵乘法，再加上 bias。
    output = ...

    # TODO：交换 x 的行和列。
    transposed = ...

    check(
        "线性变换 XW+b",
        output,
        np.array([[22.5, 27.5], [49.5, 63.5]]),
    )
    check("二维矩阵转置", transposed, x.T)


def exercise_5_llm_linear_layer() -> None:
    """练习 5：对三维隐藏状态应用同一个线性变换。"""
    print("\n=== 练习 5：大模型线性层形状 ===")

    hidden_states = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    weight = np.arange(4 * 5, dtype=np.float64).reshape(4, 5)
    bias = np.arange(5, dtype=np.float64)

    # TODO：使用 @ 和广播计算线性层输出。
    output = ...

    check(
        "三维隐藏状态的线性变换",
        output,
        hidden_states @ weight + bias,
    )
    check(
        "线性变换后的形状",
        Ellipsis if output is Ellipsis else output.shape,
        (2, 3, 5),
    )


def exercise_6_cosine_similarity() -> None:
    """练习 6：实现单个向量的余弦相似度。"""
    print("\n=== 练习 6：余弦相似度 ===")

    a = np.array([1.0, 1.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])

    # TODO：计算分子，即 a 和 b 的点积。
    numerator = ...

    # TODO：计算分母，即 a 的范数乘以 b 的范数。
    denominator = ...

    # TODO：用分子除以分母，得到余弦相似度。
    similarity = ...

    check("余弦相似度分子", numerator, 1.0)
    check("余弦相似度分母", denominator, np.sqrt(2.0))
    check("单个向量余弦相似度", similarity, 1.0 / np.sqrt(2.0))


def exercise_7_batch_cosine_similarity() -> None:
    """练习 7：一次计算查询向量与全部候选向量的相似度。"""
    print("\n=== 练习 7：批量余弦相似度 ===")

    query = np.array([1.0, 1.0, 0.0])
    candidates = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # TODO：使用一次矩阵-向量乘法计算所有候选与 query 的点积。
    numerators = ...

    # TODO：沿 axis=1 计算每个候选向量的范数。
    candidate_norms = ...

    # TODO：计算 query 的范数。
    query_norm = ...

    # TODO：一次得到全部余弦相似度，禁止使用 Python 循环。
    similarities = ...

    expected = np.array(
        [
            1.0 / np.sqrt(2.0),
            1.0,
            -1.0 / np.sqrt(2.0),
            0.0,
        ]
    )

    check("批量点积分子", numerators, np.array([1.0, 2.0, -1.0, 0.0]))
    check(
        "候选向量范数",
        candidate_norms,
        np.array([1.0, np.sqrt(2.0), 1.0, 1.0]),
    )
    check("查询向量范数", query_norm, np.sqrt(2.0))
    check("批量余弦相似度", similarities, expected)


def exercise_8_vectorization() -> None:
    """练习 8：用向量化代替逐元素 Python 循环。"""
    print("\n=== 练习 8：向量化与计时 ===")

    rng = np.random.default_rng(42)
    x = rng.standard_normal(100_000)

    start = perf_counter()
    loop_result = np.array([value * value + 2.0 * value for value in x])
    loop_seconds = perf_counter() - start

    start = perf_counter()
    # TODO：不使用 Python 循环，一次计算所有元素的 value² + 2×value。
    vectorized_result = ...
    vectorized_seconds = perf_counter() - start

    check("向量化结果与循环一致", vectorized_result, loop_result)

    if isinstance(vectorized_result, np.ndarray):
        print(f"  循环版本：{loop_seconds:.6f} 秒")
        print(f"  向量化版本：{vectorized_seconds:.6f} 秒")
        if vectorized_seconds > 0:
            print(f"  本次速度比：{loop_seconds / vectorized_seconds:.1f} 倍")
        print("  提示：计时会波动，应该多次运行后再比较。")


def main() -> None:
    print("第 9 天：广播、向量化与矩阵运算练习")
    print("请把每道题中的 ... 替换为你的 NumPy 表达式。")

    exercise_1_element_wise_operations()
    exercise_2_broadcasting()
    exercise_3_standardization()
    exercise_4_matrix_multiplication()
    exercise_5_llm_linear_layer()
    exercise_6_cosine_similarity()
    exercise_7_batch_cosine_similarity()
    exercise_8_vectorization()


if __name__ == "__main__":
    main()

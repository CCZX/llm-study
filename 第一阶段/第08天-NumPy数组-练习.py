"""第 8 天：NumPy 数组练习。

使用方法：
1. 激活项目虚拟环境。
2. 将每道题中赋值右侧的 ``...`` 替换成 NumPy 表达式。
3. 运行本文件，程序会自动检查答案。

运行命令：
    python 第一阶段/第08天-NumPy数组-练习.py
"""

from __future__ import annotations

from typing import Any

import numpy as np


def check(name: str, actual: Any, expected: Any) -> None:
    """打印一道题的检查结果，未填写的题目不会导致程序退出。"""
    if actual is Ellipsis:
        print(f"[待完成] {name}")
        return

    if isinstance(expected, np.ndarray):
        passed = isinstance(actual, np.ndarray) and np.allclose(actual, expected)
    else:
        passed = actual == expected

    if passed:
        print(f"[通过] {name}")
    else:
        print(f"[未通过] {name}")
        print(f"  你的结果：{actual}")
        print(f"  预期结果：{expected}")


def exercise_1_array_attributes() -> None:
    """练习 1：ndim、shape、size 和 dtype。"""
    print("\n=== 练习 1：数组属性 ===")

    a = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    # TODO：分别取得 a 的维度数、形状和元素总数。
    ndim = ...
    shape = ...
    size = ...

    check("a 的维度数", ndim, 2)
    check("a 的形状", shape, (2, 3))
    check("a 的元素总数", size, 6)

    # TODO：把 a 转换为 float32 类型。
    a_float = ...

    if a_float is Ellipsis:
        print("[待完成] 转换为 float32")
    else:
        check("转换后的形状不变", a_float.shape, (2, 3))
        check("转换为 float32", a_float.dtype, np.dtype("float32"))


def exercise_2_create_arrays() -> None:
    """练习 2：创建数组。"""
    print("\n=== 练习 2：创建数组 ===")

    # TODO：创建形状为 (2, 3) 的全零数组。
    zeros = ...

    # TODO：创建 [0, 2, 4, 6, 8]。
    even_numbers = ...

    # TODO：创建从 0 到 1 的 5 个等间距数字。
    points = ...

    check("创建全零数组", zeros, np.zeros((2, 3)))
    check("创建偶数序列", even_numbers, np.array([0, 2, 4, 6, 8]))
    check("创建等间距数字", points, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))


def exercise_3_indexing_and_slicing() -> None:
    """练习 3：三维数组的索引和切片。"""
    print("\n=== 练习 3：索引与切片 ===")

    x = np.arange(24).reshape(2, 3, 4)

    # TODO：取得第一个二维块，结果形状应为 (3, 4)。
    first_block = ...

    # TODO：取得所有二维块的第 2 行，结果形状应为 (2, 4)。
    second_rows = ...

    # TODO：取得最后一个维度中的第一个特征，结果形状应为 (2, 3)。
    first_feature = ...

    # TODO：取得最后一个维度中的前两个特征，结果形状应为 (2, 3, 2)。
    first_two_features = ...

    check("第一个二维块", first_block, x[0])
    check("所有块的第2行", second_rows, x[:, 1, :])
    check("第一个特征", first_feature, x[:, :, 0])
    check("前两个特征", first_two_features, x[:, :, :2])


def exercise_4_reshape() -> None:
    """练习 4：reshape 和自动推断。"""
    print("\n=== 练习 4：reshape ===")

    a = np.arange(24)

    # TODO：把 a 改成形状 (2, 3, 4)。
    three_dimensional = ...

    # TODO：使用 -1，让 NumPy 自动推断第二个维度，结果形状应为 (6, 4)。
    inferred = ...

    check(
        "reshape 为 (2, 3, 4)",
        Ellipsis if three_dimensional is Ellipsis else three_dimensional.shape,
        (2, 3, 4),
    )
    check(
        "使用 -1 自动推断为 (6, 4)",
        Ellipsis if inferred is Ellipsis else inferred.shape,
        (6, 4),
    )


def exercise_5_aggregation() -> None:
    """练习 5：sum、mean、max 和 axis。"""
    print("\n=== 练习 5：聚合运算与 axis ===")

    a = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    # TODO：计算全部元素之和。
    total = ...

    # TODO：沿 axis=0 求平均值，结果应包含三个数字。
    mean_axis_0 = ...

    # TODO：沿 axis=1 求平均值，结果应包含两个数字。
    mean_axis_1 = ...

    # TODO：计算每一行的最大值。
    row_max = ...

    check("全部元素之和", total, 21)
    check("沿 axis=0 求均值", mean_axis_0, np.array([2.5, 3.5, 4.5]))
    check("沿 axis=1 求均值", mean_axis_1, np.array([2.0, 5.0]))
    check("每一行的最大值", row_max, np.array([3, 6]))


def exercise_6_llm_shapes() -> None:
    """练习 6：大模型常见的 batch、sequence、hidden 形状。"""
    print("\n=== 练习 6：大模型数据形状 ===")

    hidden_states = np.zeros((32, 128, 768), dtype=np.float32)

    # TODO：对批次维度求平均，并取得结果的 shape。
    mean_batch_shape = ...

    # TODO：对序列维度求平均，并取得结果的 shape。
    mean_sequence_shape = ...

    # TODO：对隐藏维度求平均，并取得结果的 shape。
    mean_hidden_shape = ...

    check("聚合批次维度后的形状", mean_batch_shape, (128, 768))
    check("聚合序列维度后的形状", mean_sequence_shape, (32, 768))
    check("聚合隐藏维度后的形状", mean_hidden_shape, (32, 128))

    # TODO：沿序列维度求平均，但使用 keepdims=True 保留该维度。
    keep_sequence_shape = ...

    check("keepdims 保留序列维度", keep_sequence_shape, (32, 1, 768))


def exercise_7_combined_axes() -> None:
    """练习 7：同时聚合多个维度。"""
    print("\n=== 练习 7：同时聚合多个维度 ===")

    rng = np.random.default_rng(42)
    x = rng.standard_normal((4, 8, 16))

    # TODO：对 axis=(1, 2) 求均值，为每个样本得到一个数字。
    sample_mean = ...

    # TODO：沿隐藏维度求最大值，为每个 Token 得到一个数字。
    token_max = ...

    check(
        "每个样本的平均值形状",
        Ellipsis if sample_mean is Ellipsis else sample_mean.shape,
        (4,),
    )
    check(
        "每个Token的最大值形状",
        Ellipsis if token_max is Ellipsis else token_max.shape,
        (4, 8),
    )


def main() -> None:
    print("第 8 天：NumPy 数组练习")
    print("请把每道题中的 ... 替换为你的 NumPy 表达式。")

    exercise_1_array_attributes()
    exercise_2_create_arrays()
    exercise_3_indexing_and_slicing()
    exercise_4_reshape()
    exercise_5_aggregation()
    exercise_6_llm_shapes()
    exercise_7_combined_axes()


if __name__ == "__main__":
    main()

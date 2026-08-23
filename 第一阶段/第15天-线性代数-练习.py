"""第 15 天：线性代数练习。"""

from __future__ import annotations

from typing import Any

import numpy as np


def check(name: str, actual: Any, expected: Any) -> None:
    if actual is Ellipsis:
        print(f"[待完成] {name}")
        return

    try:
        if isinstance(expected, np.ndarray):
            passed = isinstance(actual, np.ndarray) and np.allclose(actual, expected)
        elif isinstance(expected, float):
            passed = bool(np.isclose(actual, expected))
        else:
            passed = actual == expected
    except (TypeError, ValueError):
        passed = False

    print(f"[{'通过' if passed else '未通过'}] {name}")
    if not passed:
        print(f"  你的结果：{actual}")
        print(f"  预期结果：{expected}")


def exercise_1_shapes() -> None:
    print("\n=== 练习 1：对象与形状 ===")
    scalar = np.array(3.0)
    vector = np.array([1.0, 2.0, 3.0])
    matrix = np.zeros((2, 3))
    tensor = np.zeros((4, 8, 16))

    # TODO：分别填写四个数组的 ndim。
    dimensions = ...
    # TODO：把 vector 变成形状 (3, 1) 的二维列向量。
    column_vector = ...

    check("四类对象的维度数", dimensions, (0, 1, 2, 3))
    check(
        "一维向量转为列向量",
        Ellipsis if column_vector is Ellipsis else column_vector.shape,
        (3, 1),
    )


def exercise_2_dot_norm_cosine() -> None:
    print("\n=== 练习 2：点积、范数与余弦相似度 ===")
    a = np.array([3.0, 4.0])
    b = np.array([4.0, -3.0])

    # TODO：计算点积。
    dot = ...
    # TODO：计算 a 和 b 的 L2 范数。
    norms = ...
    # TODO：计算余弦相似度。
    cosine = ...

    check("向量点积", dot, 0.0)
    check("两个向量的范数", norms, np.array([5.0, 5.0]))
    check("正交向量的余弦相似度", cosine, 0.0)


def exercise_3_matrix_multiplication() -> None:
    print("\n=== 练习 3：矩阵乘法 ===")
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    weight = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    bias = np.array([0.5, -0.5])

    # TODO：计算 x @ weight + bias。
    output = ...
    # TODO：计算 x @ x.T，结果表示样本两两之间的点积。
    gram = ...

    check("仿射变换", output, np.array([[22.5, 27.5], [49.5, 63.5]]))
    check("样本 Gram 矩阵", gram, np.array([[14.0, 32.0], [32.0, 77.0]]))


def exercise_4_batch_linear_layer() -> None:
    print("\n=== 练习 4：批量线性层 ===")
    hidden = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    weight = np.arange(4 * 5, dtype=np.float64).reshape(4, 5)
    bias = np.arange(5, dtype=np.float64)

    # TODO：为所有 Token 应用相同的仿射变换。
    projected = ...

    check("批量线性层数值", projected, hidden @ weight + bias)
    check(
        "批量线性层形状",
        Ellipsis if projected is Ellipsis else projected.shape,
        (2, 3, 5),
    )


def exercise_5_embedding_lookup() -> None:
    print("\n=== 练习 5：Embedding 查表 ===")
    table = np.arange(10 * 4, dtype=np.float64).reshape(10, 4)
    token_ids = np.array([[2, 5, 1], [0, 4, 3]])

    # TODO：使用 token_ids 从 table 中取出所有 Token 向量。
    embeddings = ...

    check("Embedding 查表", embeddings, table[token_ids])
    check(
        "Embedding 结果形状",
        Ellipsis if embeddings is Ellipsis else embeddings.shape,
        (2, 3, 4),
    )


def exercise_6_transpose() -> None:
    print("\n=== 练习 6：转置与交换轴 ===")
    x = np.zeros((2, 3, 4))

    # TODO：只交换 x 的最后两个轴。
    swapped = ...

    check(
        "交换最后两个轴",
        Ellipsis if swapped is Ellipsis else swapped.shape,
        (2, 4, 3),
    )


def exercise_7_eigenvector() -> None:
    print("\n=== 练习 7：特征向量直觉 ===")
    matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
    vector = np.array([1.0, 0.0])

    # TODO：计算 matrix @ vector。
    transformed = ...
    # TODO：填写对应特征值，使 transformed == eigenvalue * vector。
    eigenvalue = ...

    check("特征向量经过变换", transformed, np.array([2.0, 0.0]))
    check("对应特征值", eigenvalue, 2.0)


def main() -> None:
    print("第 15 天：线性代数练习")
    exercise_1_shapes()
    exercise_2_dot_norm_cosine()
    exercise_3_matrix_multiplication()
    exercise_4_batch_linear_layer()
    exercise_5_embedding_lookup()
    exercise_6_transpose()
    exercise_7_eigenvector()


if __name__ == "__main__":
    main()

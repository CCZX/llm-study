"""第 18 天：机器学习基本流程练习。"""

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


def split_indices(
    sample_count: int,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """随机划分索引，剩余样本全部进入测试集。"""
    # TODO：替换下面的 ...。
    return ...


def fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """只使用训练集，返回 keepdims=True 的均值和安全标准差。"""
    # TODO：替换下面的 ...。
    return ...


def transform_standardizer(
    x: np.ndarray,
    mean: np.ndarray,
    safe_std: np.ndarray,
) -> np.ndarray:
    """使用已有训练统计量转换任意数据集。"""
    # TODO：替换下面的 ...。
    return ...


def regression_baseline(
    y_train: np.ndarray,
    y_evaluation: np.ndarray,
) -> tuple[np.ndarray, float]:
    """始终预测训练标签均值，返回预测和 RMSE。"""
    # TODO：替换下面的 ...。
    return ...


def binary_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    """返回 tp、tn、fp、fn、accuracy、precision 和 recall。"""
    # TODO：替换下面的 ...；分母为 0 时相应指标返回 0.0。
    return ...


def exercise_1_split() -> None:
    print("\n=== 练习 1：数据划分 ===")
    result = split_indices(100, seed=42)
    if result is Ellipsis:
        check("数据划分", Ellipsis, "70/15/15")
        return

    train, valid, test = result
    check("训练集大小", len(train), 70)
    check("验证集大小", len(valid), 15)
    check("测试集大小", len(test), 15)
    all_indices = np.concatenate([train, valid, test])
    check("索引无重复且覆盖全部样本", len(np.unique(all_indices)), 100)
    repeated = split_indices(100, seed=42)
    check("固定种子可复现", np.array_equal(train, repeated[0]), True)


def exercise_2_standardization() -> None:
    print("\n=== 练习 2：训练集拟合标准化 ===")
    x_train = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    x_valid = np.array([[4.0, 5.0], [5.0, 5.0]])

    statistics = fit_standardizer(x_train)
    if statistics is Ellipsis:
        check("标准化统计量", Ellipsis, "训练均值和安全标准差")
        return

    mean, safe_std = statistics
    scaled_train = transform_standardizer(x_train, mean, safe_std)
    scaled_valid = transform_standardizer(x_valid, mean, safe_std)

    if scaled_train is Ellipsis or scaled_valid is Ellipsis:
        check("应用训练集标准化统计量", Ellipsis, "训练集和验证集转换结果")
        return

    check("均值形状", mean.shape, (1, 2))
    check("常数列安全标准差", float(safe_std[0, 1]), 1.0)
    check("训练集第一列标准化均值", float(scaled_train[:, 0].mean()), 0.0)
    check("验证集使用训练均值", scaled_valid[:, 0], (x_valid[:, 0] - 2.0) / np.std([1, 2, 3]))


def exercise_3_baseline() -> None:
    print("\n=== 练习 3：回归基线 ===")
    y_train = np.array([4.0, 6.0, 8.0])
    y_valid = np.array([5.0, 9.0])
    result = regression_baseline(y_train, y_valid)

    if result is Ellipsis:
        check("回归基线", Ellipsis, "预测训练均值")
        return

    predictions, rmse = result
    check("基线预测", predictions, np.array([6.0, 6.0]))
    check("基线 RMSE", rmse, np.sqrt(5.0))


def exercise_4_classification_metrics() -> None:
    print("\n=== 练习 4：分类指标 ===")
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0])
    metrics = binary_classification_metrics(y_true, y_pred)

    if metrics is Ellipsis:
        check("二分类指标", Ellipsis, "混淆矩阵和三个指标")
        return

    check("TP", metrics["tp"], 2)
    check("TN", metrics["tn"], 2)
    check("FP", metrics["fp"], 1)
    check("FN", metrics["fn"], 1)
    check("准确率", metrics["accuracy"], 4.0 / 6.0)
    check("精确率", metrics["precision"], 2.0 / 3.0)
    check("召回率", metrics["recall"], 2.0 / 3.0)


def exercise_5_model_selection() -> None:
    print("\n=== 练习 5：模型选择 ===")
    validation_rmse = np.array([0.82, 0.71, 0.76])
    test_rmse = np.array([0.68, 0.73, 0.65])

    # TODO：只根据验证集选择候选模型索引。
    selected_index = ...
    # TODO：选择完成后，读取该模型的测试 RMSE。
    final_test_rmse = ...

    check("根据验证集选择模型", selected_index, 1)
    check("冻结选择后的测试结果", final_test_rmse, 0.73)


def exercise_6_diagnosis() -> None:
    print("\n=== 练习 6：欠拟合与过拟合判断 ===")
    cases = {
        "a": (0.9, 0.95),
        "b": (0.1, 0.8),
        "c": (0.15, 0.18),
    }

    # TODO：填写 a、b、c 对应的主要现象：欠拟合、过拟合、较好泛化。
    diagnoses = ...
    check("学习曲线诊断", diagnoses, ("欠拟合", "过拟合", "较好泛化"))


def main() -> None:
    print("第 18 天：机器学习基本流程练习")
    exercise_1_split()
    exercise_2_standardization()
    exercise_3_baseline()
    exercise_4_classification_metrics()
    exercise_5_model_selection()
    exercise_6_diagnosis()


if __name__ == "__main__":
    main()

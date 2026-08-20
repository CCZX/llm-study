"""第 12 天：数据清洗、分组统计与可视化练习。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = Path(__file__).parent / "data"
EVALUATIONS_PATH = DATA_DIR / "第12天-模型评测.csv"
MODEL_INFO_PATH = DATA_DIR / "第12天-模型信息.csv"


def check(name: str, actual: Any, expected: Any) -> None:
    """自动检查标量、Series 和 DataFrame。"""
    if actual is Ellipsis:
        print(f"[待完成] {name}")
        return

    try:
        if isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, expected, check_dtype=True)
        elif isinstance(expected, pd.Series):
            pd.testing.assert_series_equal(actual, expected, check_dtype=True)
        elif actual != expected:
            raise AssertionError
    except (AssertionError, AttributeError, TypeError, ValueError):
        print(f"[未通过] {name}")
        print(f"  你的结果：\n{actual}")
        print(f"  预期结果：\n{expected}")
    else:
        print(f"[通过] {name}")


def safe_call(name: str, function: Callable[..., Any], *args: Any) -> Any:
    try:
        return function(*args)
    except Exception as error:  # 练习脚本需要继续执行
        print(f"[未通过] {name} 执行时发生异常：{type(error).__name__}: {error}")
        return Ellipsis


def clean_evaluations(raw: pd.DataFrame) -> pd.DataFrame:
    """删除完整重复，并按模型填充 score 与 gpu_memory_gb。"""
    # TODO：替换下面的 ...；返回新 DataFrame，不修改 raw。
    return ...


def summarize_by_model(cleaned: pd.DataFrame) -> pd.DataFrame:
    """按模型统计运行数、平均分和平均延迟。"""
    # TODO：使用命名聚合，按 mean_score 降序，并重置为连续索引。
    return ...


def merge_model_info(cleaned: pd.DataFrame, model_info: pd.DataFrame) -> pd.DataFrame:
    """把模型信息左连接到每条评测记录。"""
    # TODO：使用 model 关联，增加 indicator，并验证 many_to_one。
    return ...


def make_score_pivot(cleaned: pd.DataFrame) -> pd.DataFrame:
    """生成任务 × 模型的平均分透视表。"""
    # TODO：index 为 task，columns 为 model，values 为 score。
    return ...


def create_plots(cleaned: pd.DataFrame) -> tuple[plt.Figure, Any]:
    """创建得分直方图、延迟散点图和按模型分组的箱线图。"""
    # TODO：返回 (fig, axes)，三个子图都必须设置标题、横轴和纵轴名称。
    # 可参考讲义，但请自己重新实现。
    return ...


def expected_clean(raw: pd.DataFrame) -> pd.DataFrame:
    """仅用于自动检查的参考结果。"""
    result = raw.drop_duplicates().copy()
    score_fill = result.groupby("model")["score"].transform("mean")
    memory_fill = result.groupby("model")["gpu_memory_gb"].transform("median")
    result["score"] = result["score"].fillna(score_fill)
    result["gpu_memory_gb"] = result["gpu_memory_gb"].fillna(memory_fill)
    return result.reset_index(drop=True)


def expected_summary(cleaned: pd.DataFrame) -> pd.DataFrame:
    return (
        cleaned.groupby("model", as_index=False)
        .agg(
            runs=("run_id", "count"),
            mean_score=("score", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
        )
        .sort_values("mean_score", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    print("第 12 天：数据清洗、分组统计与可视化练习")
    raw = pd.read_csv(EVALUATIONS_PATH)
    model_info = pd.read_csv(MODEL_INFO_PATH)
    reference_cleaned = expected_clean(raw)

    print("\n=== 练习 1：清洗 ===")
    cleaned = safe_call("清洗评测数据", clean_evaluations, raw)
    check("清洗评测数据", cleaned, reference_cleaned)
    check("清洗函数不修改原数据", raw.shape, (10, 6))

    print("\n=== 练习 2：分组统计 ===")
    summary = safe_call("按模型汇总", summarize_by_model, reference_cleaned)
    check("按模型汇总", summary, expected_summary(reference_cleaned))

    print("\n=== 练习 3：合并 ===")
    merged = safe_call("合并模型信息", merge_model_info, reference_cleaned, model_info)
    expected_merged = reference_cleaned.merge(
        model_info,
        on="model",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    check("合并模型信息", merged, expected_merged)

    print("\n=== 练习 4：透视表 ===")
    pivot = safe_call("得分透视表", make_score_pivot, reference_cleaned)
    expected_pivot = reference_cleaned.pivot_table(
        index="task",
        columns="model",
        values="score",
        aggfunc="mean",
    )
    check("得分透视表", pivot, expected_pivot)

    print("\n=== 练习 5：相关系数 ===")
    # TODO：选择 score、latency_ms、gpu_memory_gb 三列并计算相关矩阵。
    correlation = ...
    expected_correlation = reference_cleaned[
        ["score", "latency_ms", "gpu_memory_gb"]
    ].corr()
    check("相关系数矩阵", correlation, expected_correlation)

    print("\n=== 练习 6：可视化 ===")
    plot_result = safe_call("创建三个图表", create_plots, reference_cleaned)
    if plot_result is Ellipsis:
        check("创建三个图表", Ellipsis, "包含三个子图的 Figure")
    else:
        fig, axes = plot_result
        axes = list(axes)
        check("创建三个子图", len(axes), 3)
        labels_complete = all(
            ax.get_title() and ax.get_xlabel() and ax.get_ylabel() for ax in axes
        )
        check("图表标题和坐标完整", bool(labels_complete), True)
        plt.close(fig)


if __name__ == "__main__":
    main()

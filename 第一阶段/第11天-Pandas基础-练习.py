"""第 11 天：Pandas 基础练习。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).parent / "data" / "第11天-模型实验.csv"


def check(name: str, actual: Any, expected: Any) -> None:
    """支持标量、Series 和 DataFrame 的自动检查。"""
    if actual is Ellipsis:
        print(f"[待完成] {name}")
        return

    try:
        if isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(
                actual.reset_index(drop=True),
                expected.reset_index(drop=True),
                check_dtype=True,
            )
        elif isinstance(expected, pd.Series):
            pd.testing.assert_series_equal(actual, expected, check_dtype=True)
        elif isinstance(expected, np.ndarray):
            if not isinstance(actual, np.ndarray) or not np.allclose(actual, expected):
                raise AssertionError
        elif actual != expected:
            raise AssertionError
    except (AssertionError, AttributeError, TypeError, ValueError):
        print(f"[未通过] {name}")
        print(f"  你的结果：\n{actual}")
        print(f"  预期结果：\n{expected}")
    else:
        print(f"[通过] {name}")


def main() -> None:
    print("第 11 天：Pandas 基础练习")
    print(f"数据文件：{DATA_PATH}")

    # TODO：使用 pd.read_csv 读取 DATA_PATH。
    df = ...

    if df is Ellipsis:
        for name in [
            "读取 CSV",
            "选择多列",
            "多条件过滤",
            "多列排序",
            "创建计算列",
            "转换类别类型",
            "统计缺失值",
            "统计重复行",
        ]:
            print(f"[待完成] {name}")
        return

    check("读取 CSV", df.shape, (9, 7))

    print("\n=== 练习 1：选择列 ===")
    # TODO：选择 model、accuracy 和 latency_ms 三列，保持为 DataFrame。
    selected_columns = ...
    check(
        "选择多列",
        selected_columns,
        df[["model", "accuracy", "latency_ms"]],
    )

    print("\n=== 练习 2：多条件过滤 ===")
    # TODO：选择 accuracy >= 0.85 且 latency_ms < 100 的全部行。
    efficient_models = ...
    expected_filtered = df.loc[
        (df["accuracy"] >= 0.85) & (df["latency_ms"] < 100)
    ]
    check("多条件过滤", efficient_models, expected_filtered)

    print("\n=== 练习 3：排序 ===")
    # TODO：先按 accuracy 降序，再按 latency_ms 升序，缺失值放最后。
    ranked = ...
    expected_ranked = df.sort_values(
        ["accuracy", "latency_ms"],
        ascending=[False, True],
        na_position="last",
    )
    check("多列排序", ranked, expected_ranked)

    print("\n=== 练习 4：创建新列 ===")
    # TODO：使用 assign 创建 accuracy_pct，值为 accuracy * 100。
    with_percentage = ...
    expected_percentage = df.assign(accuracy_pct=df["accuracy"] * 100.0)
    check("创建计算列", with_percentage, expected_percentage)

    print("\n=== 练习 5：修改数据类型 ===")
    # TODO：复制 df，并把 model 列转换为 category 类型。
    typed = ...
    if typed is Ellipsis:
        check("转换类别类型", Ellipsis, "category")
    else:
        check("转换类别类型", str(typed["model"].dtype), "category")

    print("\n=== 练习 6：缺失值 ===")
    # TODO：统计 df 每一列的缺失值数量。
    missing_count = ...
    check("统计缺失值", missing_count, df.isna().sum())

    print("\n=== 练习 7：重复数据 ===")
    # TODO：统计完整重复行的数量。
    duplicate_count = ...
    check("统计重复行", duplicate_count, 1)


if __name__ == "__main__":
    main()

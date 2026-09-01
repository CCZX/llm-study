"""第 22 天：Tensor、dtype、shape 与 device 练习。

使用方法：
1. 安装并导入 PyTorch。
2. 将每道题中的 ``...`` 替换为自己的实现。
3. 运行本文件，程序会输出 [通过]、[未通过] 或 [待完成]。

运行命令：
    python 第二阶段/第22天-Tensor与设备-练习.py
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch


def check(name: str, actual: Any, expected: Any, atol: float = 1e-7) -> None:
    """打印一道题的检查结果。"""
    if actual is Ellipsis:
        print(f"[待完成] {name}")
        return

    try:
        if isinstance(expected, torch.Tensor):
            passed = (
                isinstance(actual, torch.Tensor)
                and actual.shape == expected.shape
                and actual.dtype == expected.dtype
                and torch.allclose(actual.cpu(), expected.cpu(), atol=atol)
            )
        elif isinstance(expected, np.ndarray):
            passed = isinstance(actual, np.ndarray) and np.allclose(actual, expected, atol=atol)
        elif isinstance(expected, float):
            passed = bool(np.isclose(actual, expected, atol=atol))
        else:
            passed = actual == expected
    except (AttributeError, RuntimeError, TypeError, ValueError):
        passed = False

    print(f"[{'通过' if passed else '未通过'}] {name}")
    if not passed:
        print(f"  你的结果：{actual}")
        print(f"  预期结果：{expected}")


def safe_call(name: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """执行函数并把异常转换成友好的检查输出。"""
    try:
        return function(*args, **kwargs)
    except Exception as error:
        print(f"[未通过] {name} 执行异常：{type(error).__name__}: {error}")
        return Ellipsis


def create_feature_tensor() -> torch.Tensor:
    """返回值为 [[1, 2, 3], [4, 5, 6]] 的 float32 Tensor。"""
    # TODO：使用 torch.tensor，并显式指定 dtype。
    return ...


def select_tensor_parts(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回第一个块、所有块的第 2 行、前两个特征。"""
    # TODO：
    # first_block 的形状应为 (3, 4)
    # second_rows 的形状应为 (2, 4)
    # first_two_features 的形状应为 (2, 3, 2)
    return ...


def reshape_and_permute(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """展平前两维，并把 sequence 维移到最前。"""
    # TODO：
    # flattened: (batch * sequence, hidden)
    # sequence_first: (sequence, batch, hidden)
    return ...


def center_rows(x: torch.Tensor) -> torch.Tensor:
    """使用广播，让每一行减去自己的均值。"""
    # TODO：使用 mean(dim=..., keepdim=True)。
    return ...


def batch_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """对最后一维做线性变换，保留 batch 和 sequence 维。"""
    # TODO：使用矩阵乘法与广播。
    return ...


def numpy_shared_memory_demo() -> tuple[np.ndarray, torch.Tensor]:
    """创建共享内存的 NumPy 数组与 Tensor，并通过 Tensor 修改第一个元素。"""
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    # TODO：不要复制 array；使用共享内存方式转成 Tensor。
    tensor = ...
    if tensor is not Ellipsis:
        # TODO：通过 tensor 把第一个元素改为 99。
        ...
    return array, tensor


def choose_device() -> torch.device:
    """按 CUDA、MPS、CPU 的优先级返回可用设备。"""
    # TODO：使用 torch.cuda.is_available() 和 torch.backends.mps.is_available()。
    return ...


def exercise_1_creation_and_attributes() -> None:
    print("\n=== 练习 1：创建与属性 ===")
    x = safe_call("创建特征 Tensor", create_feature_tensor)
    if x is Ellipsis:
        check("创建特征 Tensor", Ellipsis, torch.zeros(2, 3))
        return

    check(
        "Tensor 内容与 dtype",
        x,
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32),
    )
    check("Tensor shape", tuple(x.shape), (2, 3))
    check("Tensor ndim", x.ndim, 2)
    check("Tensor 元素数", x.numel(), 6)


def exercise_2_indexing() -> None:
    print("\n=== 练习 2：索引与切片 ===")
    x = torch.arange(24).reshape(2, 3, 4)
    result = safe_call("索引与切片", select_tensor_parts, x)
    if result is Ellipsis:
        check("索引与切片", Ellipsis, True)
        return

    first_block, second_rows, first_two_features = result
    check("第一个块", first_block, x[0])
    check("所有块的第 2 行", second_rows, x[:, 1, :])
    check("前两个特征", first_two_features, x[:, :, :2])


def exercise_3_shape_transformations() -> None:
    print("\n=== 练习 3：形状变换 ===")
    x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    result = safe_call("形状变换", reshape_and_permute, x)
    if result is Ellipsis:
        check("形状变换", Ellipsis, True)
        return

    flattened, sequence_first = result
    check("展平前两维", flattened, x.reshape(6, 4))
    check("sequence-first", sequence_first, x.permute(1, 0, 2))


def exercise_4_broadcasting() -> None:
    print("\n=== 练习 4：广播与聚合 ===")
    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    centered = safe_call("逐行中心化", center_rows, x)
    check(
        "逐行中心化",
        centered,
        torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]]),
    )
    if centered is not Ellipsis:
        check("中心化后每行均值为零", centered.mean(dim=1), torch.zeros(2))


def exercise_5_batch_linear() -> None:
    print("\n=== 练习 5：批量线性变换 ===")
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    weight = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 10
    bias = torch.arange(5, dtype=torch.float32)
    output = safe_call("批量线性变换", batch_linear, hidden, weight, bias)
    check("批量线性变换", output, hidden @ weight + bias)
    if output is not Ellipsis:
        check("输出形状", tuple(output.shape), (2, 3, 5))


def exercise_6_numpy_interop() -> None:
    print("\n=== 练习 6：NumPy 共享内存 ===")
    result = safe_call("NumPy 共享内存", numpy_shared_memory_demo)
    if result is Ellipsis:
        check("NumPy 共享内存", Ellipsis, True)
        return

    array, tensor = result
    check("Tensor 内容", tensor, torch.tensor([99.0, 2.0, 3.0]))
    check("修改 Tensor 同时修改 NumPy", array, np.array([99.0, 2.0, 3.0]))


def exercise_7_device() -> None:
    print("\n=== 练习 7：设备无关代码 ===")
    device = safe_call("选择设备", choose_device)
    if device is Ellipsis:
        check("选择设备", Ellipsis, torch.device("cpu"))
        return

    expected_type = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    check("设备优先级", device.type, expected_type)

    x = torch.randn(2, 3).to(device)
    y = torch.ones_like(x)
    check("新 Tensor 继承 device", y.device.type, device.type)
    check("设备上的计算结果", (x + y).device.type, device.type)


def main() -> None:
    print("第 22 天：Tensor 与设备练习")
    print("请把每道题中的 ... 替换为自己的 PyTorch 实现。")
    exercise_1_creation_and_attributes()
    exercise_2_indexing()
    exercise_3_shape_transformations()
    exercise_4_broadcasting()
    exercise_5_batch_linear()
    exercise_6_numpy_interop()
    exercise_7_device()


if __name__ == "__main__":
    main()

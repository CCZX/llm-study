"""第 24 天：nn.Module 与模型构建练习。"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import torch
from torch import nn


def check(name: str, actual: Any, expected: Any, atol: float = 1e-7) -> None:
    if actual is Ellipsis:
        print(f"[待完成] {name}")
        return

    try:
        if isinstance(expected, torch.Tensor):
            passed = (
                isinstance(actual, torch.Tensor)
                and actual.shape == expected.shape
                and torch.allclose(actual.detach().cpu(), expected.cpu(), atol=atol)
            )
        elif isinstance(expected, float):
            passed = abs(float(actual) - expected) <= atol
        else:
            passed = actual == expected
    except (AttributeError, RuntimeError, TypeError, ValueError):
        passed = False

    print(f"[{'通过' if passed else '未通过'}] {name}")
    if not passed:
        print(f"  你的结果：{actual}")
        print(f"  预期结果：{expected}")


def safe_call(name: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except Exception as error:
        print(f"[未通过] {name} 执行异常：{type(error).__name__}: {error}")
        return Ellipsis


class LinearModel(nn.Module):
    """单个 Linear 层。"""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        # TODO：创建 nn.Linear。
        self.linear = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO：调用 self.linear。
        return ...


class MLP(nn.Module):
    """Linear -> ReLU -> Linear，返回 logits。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        # TODO：分别创建 linear1、activation 和 linear2。
        self.linear1 = ...
        self.activation = ...
        self.linear2 = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO：完成前向传播，不要做 Softmax。
        return ...


class StackedLinear(nn.Module):
    """使用 ModuleList 保存若干个同维 Linear 层。"""

    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        # TODO：创建包含 depth 个 nn.Linear(width, width) 的 ModuleList。
        self.layers = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO：依次执行 Linear 和 ReLU。
        return ...


def count_trainable_parameters(model: nn.Module) -> int:
    """返回 requires_grad=True 的参数元素总数。"""
    # TODO：遍历 model.parameters()。
    return ...


def initialize_linear_layers(model: nn.Module) -> None:
    """对所有 Linear 使用 Kaiming normal 初始化，并把偏置清零。"""
    # TODO：可定义内部函数，再调用 model.apply(...)。
    ...


def exercise_1_linear_module() -> None:
    print("\n=== 练习 1：Linear Module ===")
    model = safe_call("创建 LinearModel", LinearModel, 2, 3)
    if model is Ellipsis:
        check("创建 LinearModel", Ellipsis, True)
        return

    if not isinstance(model.linear, nn.Linear):
        check("linear 是 nn.Linear", False, True)
        return

    with torch.no_grad():
        model.linear.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
        )
        model.linear.bias.copy_(torch.tensor([0.5, -0.5, 1.0]))

    x = torch.tensor([[2.0, 3.0], [-1.0, 4.0]])
    output = safe_call("Linear 前向", model, x)
    expected = x @ model.linear.weight.T + model.linear.bias
    check("Linear 输出", output, expected)
    if output is not Ellipsis:
        check("Linear 输出形状", tuple(output.shape), (2, 3))


def exercise_2_mlp() -> None:
    print("\n=== 练习 2：两层 MLP ===")
    model = safe_call("创建 MLP", MLP, 4, 8, 3)
    if model is Ellipsis:
        check("创建 MLP", Ellipsis, True)
        return

    check("linear1 类型", isinstance(model.linear1, nn.Linear), True)
    check("activation 类型", isinstance(model.activation, nn.ReLU), True)
    check("linear2 类型", isinstance(model.linear2, nn.Linear), True)

    for batch_size in [1, 5, 11]:
        x = torch.randn(batch_size, 4)
        output = safe_call(f"batch={batch_size} 前向", model, x)
        if output is not Ellipsis:
            check(f"batch={batch_size} 输出形状", tuple(output.shape), (batch_size, 3))

    sequence_output = safe_call("三维输入前向", model, torch.randn(2, 7, 4))
    if sequence_output is not Ellipsis:
        check("三维输入只变换最后一维", tuple(sequence_output.shape), (2, 7, 3))


def exercise_3_parameter_count() -> None:
    print("\n=== 练习 3：参数统计 ===")
    model = MLP(4, 8, 3)
    if not isinstance(model.linear1, nn.Linear) or not isinstance(model.linear2, nn.Linear):
        check("MLP 参数统计", Ellipsis, 67)
        return

    count = safe_call("统计可训练参数", count_trainable_parameters, model)
    # 4*8+8 + 8*3+3 = 67
    check("MLP 可训练参数数量", count, 67)
    names = {name for name, _ in model.named_parameters()}
    check(
        "参数注册名称",
        names,
        {"linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"},
    )


def exercise_4_module_list() -> None:
    print("\n=== 练习 4：ModuleList ===")
    model = safe_call("创建 StackedLinear", StackedLinear, 4, 3)
    if model is Ellipsis:
        check("创建 StackedLinear", Ellipsis, True)
        return

    check("layers 使用 ModuleList", isinstance(model.layers, nn.ModuleList), True)
    if not isinstance(model.layers, nn.ModuleList):
        return
    check("层数", len(model.layers), 3)
    output = safe_call("堆叠层前向", model, torch.randn(5, 4))
    if output is not Ellipsis:
        check("堆叠层输出形状", tuple(output.shape), (5, 4))
    # 每层 4*4+4=20，共 60。
    check("ModuleList 参数均被注册", count_trainable_parameters(model), 60)


def exercise_5_initialization() -> None:
    print("\n=== 练习 5：参数初始化 ===")
    torch.manual_seed(42)
    model = MLP(4, 8, 3)
    if not isinstance(model.linear1, nn.Linear) or not isinstance(model.linear2, nn.Linear):
        check("初始化 Linear 层", Ellipsis, True)
        return

    safe_call("初始化 Linear 层", initialize_linear_layers, model)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            check(f"{name} 偏置为零", module.bias, torch.zeros_like(module.bias))
            check(f"{name} 权重有限", bool(torch.isfinite(module.weight).all()), True)
            check(f"{name} 权重非全零", bool(torch.any(module.weight != 0)), True)


def exercise_6_state_dict() -> None:
    print("\n=== 练习 6：state_dict ===")
    torch.manual_seed(7)
    model = MLP(4, 8, 3)
    if not isinstance(model.linear1, nn.Linear) or not isinstance(model.linear2, nn.Linear):
        check("state_dict 恢复", Ellipsis, True)
        return

    x = torch.randn(5, 4)
    expected = model(x).detach()
    snapshot = deepcopy(model.state_dict())

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    changed = model(x).detach()
    check("参数修改后输出变化", bool(not torch.allclose(changed, expected)), True)

    model.load_state_dict(snapshot)
    restored = model(x).detach()
    check("加载 state_dict 后恢复输出", restored, expected)


def main() -> None:
    print("第 24 天：nn.Module 与模型构建练习")
    print("请把每道题中的 ... 替换为自己的 PyTorch 实现。")
    exercise_1_linear_module()
    exercise_2_mlp()
    exercise_3_parameter_count()
    exercise_4_module_list()
    exercise_5_initialization()
    exercise_6_state_dict()


if __name__ == "__main__":
    main()

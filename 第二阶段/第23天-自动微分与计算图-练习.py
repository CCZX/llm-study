"""第 23 天：计算图与自动微分练习。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


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


def scalar_derivative(x_value: float) -> torch.Tensor:
    """计算 y=x^2+2x+1 对 x 的梯度。"""
    x = torch.tensor(x_value, dtype=torch.float64, requires_grad=True)
    # TODO：构造 y，调用 backward，并返回 x.grad。
    return ...


def vector_sum_gradient(x: torch.Tensor) -> torch.Tensor:
    """计算 sum(x^2) 对 x 的梯度。"""
    x = x.clone().detach().requires_grad_(True)
    # TODO：先把向量输出聚合成标量，再反向。
    return ...


def linear_mse_gradients(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 MSE、权重梯度和偏置梯度。"""
    weight = weight.clone().detach().requires_grad_(True)
    bias = bias.clone().detach().requires_grad_(True)
    # TODO：前向、均方误差、backward。
    return ...


def accumulated_gradient() -> torch.Tensor:
    """先对 x^2 反向，再对 3x 反向，返回累积梯度。"""
    x = torch.tensor(2.0, requires_grad=True)
    # TODO：分别执行两次前向和 backward，不要清零。
    return ...


def detached_and_no_grad() -> tuple[bool, bool, bool]:
    """检查 detach 和 no_grad 是否停止记录计算图。"""
    x = torch.tensor(2.0, requires_grad=True)
    y = x**2
    # TODO：创建 detached = y.detach()。
    detached = ...
    # TODO：在 torch.no_grad() 中计算 z = x**3。
    z = ...
    if detached is Ellipsis or z is Ellipsis:
        return ...
    return detached.requires_grad, z.requires_grad, detached.grad_fn is None


def manual_sgd_step(
    initial_value: float,
    learning_rate: float,
) -> tuple[torch.Tensor, bool]:
    """对 loss=(w-3)^2 做一次安全的 SGD 更新。"""
    weight = torch.tensor(initial_value, dtype=torch.float64, requires_grad=True)
    # TODO：计算 loss 并反向。
    # TODO：在 no_grad 中执行 weight -= learning_rate * weight.grad。
    # TODO：把 weight.grad 设为 None。
    return ...


def exercise_1_scalar_gradient() -> None:
    print("\n=== 练习 1：标量梯度 ===")
    gradient = safe_call("标量梯度", scalar_derivative, 3.0)
    check("x=3 时的梯度", gradient, torch.tensor(8.0, dtype=torch.float64))


def exercise_2_vector_gradient() -> None:
    print("\n=== 练习 2：非标量输出 ===")
    x = torch.tensor([1.0, 2.0, 3.0])
    gradient = safe_call("向量平方和梯度", vector_sum_gradient, x)
    check("sum(x^2) 的梯度", gradient, torch.tensor([2.0, 4.0, 6.0]))


def exercise_3_linear_mse() -> None:
    print("\n=== 练习 3：线性回归自动微分 ===")
    x = torch.tensor(
        [[1.0, 2.0], [2.0, -1.0], [0.5, 3.0]],
        dtype=torch.float64,
    )
    y = torch.tensor([2.0, -1.0, 3.0], dtype=torch.float64)
    weight = torch.tensor([0.2, -0.3], dtype=torch.float64)
    bias = torch.tensor(0.1, dtype=torch.float64)

    result = safe_call("线性回归自动微分", linear_mse_gradients, x, y, weight, bias)
    if result is Ellipsis:
        check("线性回归自动微分", Ellipsis, True)
        return

    loss, grad_weight, grad_bias = result
    predictions = x @ weight + bias
    errors = predictions - y
    expected_loss = (errors**2).mean()
    expected_weight = (2.0 / len(x)) * (x.T @ errors)
    expected_bias = 2.0 * errors.mean()
    check("MSE", loss, expected_loss)
    check("权重梯度", grad_weight, expected_weight)
    check("偏置梯度", grad_bias, expected_bias)


def exercise_4_accumulation() -> None:
    print("\n=== 练习 4：梯度累积 ===")
    gradient = safe_call("梯度累积", accumulated_gradient)
    check("两次反向后的累积梯度", gradient, torch.tensor(7.0))


def exercise_5_stop_tracking() -> None:
    print("\n=== 练习 5：停止记录梯度 ===")
    result = safe_call("detach 与 no_grad", detached_and_no_grad)
    check("detach 与 no_grad", result, (False, False, True))


def exercise_6_manual_sgd() -> None:
    print("\n=== 练习 6：手写 SGD 更新 ===")
    result = safe_call("手写 SGD", manual_sgd_step, 0.0, 0.1)
    if result is Ellipsis:
        check("手写 SGD", Ellipsis, True)
        return

    weight, grad_is_none = result
    # 初始梯度 2*(0-3)=-6，更新为 0-0.1*(-6)=0.6。
    check("参数更新结果", weight.detach(), torch.tensor(0.6, dtype=torch.float64))
    check("更新后清理梯度", grad_is_none, True)
    check("参数仍是叶子 Tensor", weight.is_leaf, True)
    check("参数仍需要梯度", weight.requires_grad, True)


def main() -> None:
    print("第 23 天：自动微分与计算图练习")
    print("请把每道题中的 ... 替换为自己的 PyTorch 实现。")
    exercise_1_scalar_gradient()
    exercise_2_vector_gradient()
    exercise_3_linear_mse()
    exercise_4_accumulation()
    exercise_5_stop_tracking()
    exercise_6_manual_sgd()


if __name__ == "__main__":
    main()

"""第 25 天：损失函数、优化器与训练循环练习。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn


Batch = tuple[torch.Tensor, torch.Tensor]


def check(name: str, actual: Any, expected: Any, atol: float = 1e-6) -> None:
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


def regression_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """检查 shape 后返回平均 MSE。"""
    # TODO：先确保 predictions.shape == targets.shape，再使用 nn.functional.mse_loss。
    return ...


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """直接使用 logits 和类别编号计算交叉熵。"""
    # TODO：不要做 Softmax。
    return ...


def train_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """执行一个训练 step，返回 Python float loss。"""
    # TODO：
    # 1. model.train()
    # 2. zero_grad(set_to_none=True)
    # 3. forward、loss、backward、step
    return ...


def weighted_epoch_loss(
    batch_mean_losses: list[float],
    batch_sizes: list[int],
) -> float:
    """把若干 batch 平均损失按样本数加权。"""
    # TODO：检查列表非空、长度一致、总样本数大于 0。
    return ...


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[Batch],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """训练一轮并返回按样本数加权的平均损失。"""
    # TODO：遍历 batches，调用与 train_step 等价的逻辑并正确累计。
    return ...


def fit_linear_regression(
    x: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 0.05,
    epochs: int = 200,
) -> tuple[nn.Linear, torch.Tensor]:
    """训练 y≈wx+b，返回模型和每轮更新前损失。"""
    # TODO：
    # 固定 torch.manual_seed(42)，创建 nn.Linear(1, 1)。
    # 使用 MSELoss 与 SGD。
    # history 应为形状 (epochs,) 的 Tensor。
    return ...


def exercise_1_losses() -> None:
    print("\n=== 练习 1：损失函数 ===")
    predictions = torch.tensor([2.0, 4.0, 5.0])
    targets = torch.tensor([1.0, 5.0, 5.0])
    mse = safe_call("回归 MSE", regression_loss, predictions, targets)
    check("回归 MSE", mse, torch.tensor(2.0 / 3.0))

    logits = torch.tensor([[3.0, 1.0, -1.0], [0.0, 2.0, 1.0]])
    labels = torch.tensor([0, 2], dtype=torch.long)
    cross_entropy = safe_call("多分类交叉熵", classification_loss, logits, labels)
    expected = nn.functional.cross_entropy(logits, labels)
    check("直接使用 logits 计算交叉熵", cross_entropy, expected)


def exercise_2_train_step() -> None:
    print("\n=== 练习 2：单个训练 step ===")
    torch.manual_seed(1)
    model = nn.Linear(2, 1)
    inputs = torch.tensor([[1.0, 2.0], [2.0, -1.0]])
    targets = torch.tensor([[3.0], [0.0]])
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    loss = safe_call(
        "单个训练 step",
        train_step,
        model,
        inputs,
        targets,
        criterion,
        optimizer,
    )
    check("train_step 返回 float", isinstance(loss, float), True)
    changed = any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
    check("训练 step 更新参数", changed, True)
    check(
        "反向后至少一个参数有梯度",
        any(parameter.grad is not None for parameter in model.parameters()),
        True,
    )


def exercise_3_weighted_average() -> None:
    print("\n=== 练习 3：整轮平均损失 ===")
    losses = [1.0, 3.0]
    sizes = [4, 1]
    actual = safe_call("加权平均损失", weighted_epoch_loss, losses, sizes)
    check("按样本数加权", actual, 1.4)
    check("不是简单 batch 平均", actual != 2.0 if actual is not Ellipsis else Ellipsis, True)


def exercise_4_train_epoch() -> None:
    print("\n=== 练习 4：训练一个 epoch ===")
    torch.manual_seed(3)
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    batches = [
        (torch.tensor([[-2.0], [-1.0], [0.0]]), torch.tensor([[-5.0], [-2.0], [1.0]])),
        (torch.tensor([[1.0]]), torch.tensor([[4.0]])),
    ]
    loss = safe_call("训练一轮", train_one_epoch, model, batches, criterion, optimizer)
    check("epoch loss 是有限 float", isinstance(loss, float), True)
    if loss is not Ellipsis:
        check("epoch loss 有限", bool(torch.isfinite(torch.tensor(loss))), True)


def exercise_5_fit_linear_regression() -> None:
    print("\n=== 练习 5：拟合线性回归 ===")
    x = torch.linspace(-2, 2, 200).unsqueeze(1)
    y = 3.0 * x - 1.0
    result = safe_call(
        "拟合线性回归",
        fit_linear_regression,
        x,
        y,
        learning_rate=0.05,
        epochs=200,
    )
    if result is Ellipsis:
        check("拟合线性回归", Ellipsis, True)
        return

    model, history = result
    check("history 形状", tuple(history.shape), (200,))
    check("损失显著下降", bool(history[-1] < history[0] * 1e-3), True)
    check("恢复权重", model.weight.detach(), torch.tensor([[3.0]]), atol=1e-2)
    check("恢复偏置", model.bias.detach(), torch.tensor([-1.0]), atol=1e-2)


def main() -> None:
    print("第 25 天：损失、优化器与训练循环练习")
    print("请把每道题中的 ... 替换为自己的 PyTorch 实现。")
    exercise_1_losses()
    exercise_2_train_step()
    exercise_3_weighted_average()
    exercise_4_train_epoch()
    exercise_5_fit_linear_regression()


if __name__ == "__main__":
    main()

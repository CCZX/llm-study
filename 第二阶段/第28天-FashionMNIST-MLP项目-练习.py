"""第 28 天：FashionMNIST MLP 项目骨架练习。

本脚本使用合成数据验证模型、训练、评估和检查点骨架，不下载数据集。
完成所有 TODO 后，再按照同名讲义创建正式 FashionMNIST 项目。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
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


class FashionMLP(nn.Module):
    """Flatten -> Linear(784, 128) -> ReLU -> Dropout -> Linear(128, 10)。"""

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        # TODO：使用 nn.Sequential 创建 self.network。
        self.network = ...

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # TODO：返回 10 类 logits，不要做 Softmax。
        return ...


def split_train_valid_indices(
    sample_count: int,
    valid_size: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回互不重叠、覆盖全部样本的训练和验证索引。"""
    # TODO：检查 0 < valid_size < sample_count，并使用固定种子 randperm。
    return ...


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[Batch],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """训练一轮，返回按样本数加权的 loss 和 accuracy。"""
    # TODO：model.train()，移动 batch，清梯度、前向、反向、更新和累计。
    return ...


def evaluate(
    model: nn.Module,
    batches: Iterable[Batch],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """在 eval + inference_mode 下返回加权 loss 和 accuracy。"""
    # TODO：不要更新参数；空数据应 raise ValueError。
    return ...


def build_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """构建可用于断点续训的内存检查点。"""
    # TODO：深拷贝 model_state、optimizer_state、epoch、best_val_loss、config。
    return ...


def overfit_tiny_batch(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    steps: int = 200,
    learning_rate: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """重复训练同一小 batch，返回损失历史和最终准确率。"""
    # TODO：使用 CrossEntropyLoss 和 SGD；每步记录更新前 loss。
    return ...


def exercise_1_model() -> None:
    print("\n=== 练习 1：FashionMLP ===")
    model = safe_call("创建 FashionMLP", FashionMLP, 0.2)
    if model is Ellipsis or not isinstance(model.network, nn.Sequential):
        check("创建 FashionMLP", Ellipsis, True)
        return

    for batch_size in [1, 8, 31]:
        images = torch.randn(batch_size, 1, 28, 28)
        logits = safe_call(f"batch={batch_size} 前向", model, images)
        if logits is not Ellipsis:
            check(f"batch={batch_size} 输出形状", tuple(logits.shape), (batch_size, 10))
            check(f"batch={batch_size} logits 有限", bool(torch.isfinite(logits).all()), True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    check("模型参数量", parameter_count, 101_770)


def exercise_2_split() -> None:
    print("\n=== 练习 2：训练/验证划分 ===")
    result = safe_call("训练/验证划分", split_train_valid_indices, 100, 20, 7)
    if result is Ellipsis:
        check("训练/验证划分", Ellipsis, True)
        return

    train, valid = result
    check("训练集大小", len(train), 80)
    check("验证集大小", len(valid), 20)
    check("索引无重叠", set(train.tolist()).isdisjoint(valid.tolist()), True)
    check("覆盖全部样本", len(torch.unique(torch.cat([train, valid]))), 100)
    repeated = split_train_valid_indices(100, 20, 7)
    check("固定种子可复现", torch.equal(train, repeated[0]), True)


def exercise_3_training_engine() -> None:
    print("\n=== 练习 3：训练引擎 ===")
    torch.manual_seed(5)
    model = nn.Linear(2, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [
        (
            torch.tensor([[2.0, 0.0], [0.0, 2.0], [3.0, 0.0]]),
            torch.tensor([0, 1, 0]),
        ),
        (
            torch.tensor([[0.0, 3.0]]),
            torch.tensor([1]),
        ),
    ]
    before = deepcopy(model.state_dict())
    metrics = safe_call(
        "训练一个 epoch",
        train_one_epoch,
        model,
        batches,
        criterion,
        optimizer,
        torch.device("cpu"),
    )
    if metrics is Ellipsis:
        check("训练一个 epoch", Ellipsis, True)
        return

    check("训练指标字段", set(metrics), {"loss", "accuracy"})
    check("训练 loss 有限", bool(torch.isfinite(torch.tensor(metrics["loss"]))), True)
    check("训练 accuracy 范围", 0.0 <= metrics["accuracy"] <= 1.0, True)
    changed = any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
    check("训练更新参数", changed, True)


def exercise_4_evaluation() -> None:
    print("\n=== 练习 4：评估引擎 ===")
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
        model.bias.zero_()
    criterion = nn.CrossEntropyLoss()
    batches = [
        (
            torch.tensor([[2.0, 0.0], [0.0, 2.0], [3.0, 0.0]]),
            torch.tensor([0, 1, 0]),
        ),
        (
            torch.tensor([[0.0, 3.0]]),
            torch.tensor([1]),
        ),
    ]
    before = deepcopy(model.state_dict())
    metrics = safe_call(
        "评估",
        evaluate,
        model,
        batches,
        criterion,
        torch.device("cpu"),
    )
    if metrics is Ellipsis:
        check("评估", Ellipsis, True)
        return

    check("评估准确率", metrics["accuracy"], 1.0)
    check("评估模式", model.training, False)
    unchanged = all(
        torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
    check("评估不更新参数", unchanged, True)
    check("评估不留下参数梯度", all(p.grad is None for p in model.parameters()), True)


def exercise_5_tiny_batch_overfit() -> None:
    print("\n=== 练习 5：小 batch 过拟合 ===")
    torch.manual_seed(12)
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    features = torch.tensor([
        [-2.0, -1.0],
        [-1.0, -2.0],
        [1.0, 2.0],
        [2.0, 1.0],
    ])
    labels = torch.tensor([0, 0, 1, 1])
    result = safe_call(
        "小 batch 过拟合",
        overfit_tiny_batch,
        model,
        features,
        labels,
        200,
        0.1,
    )
    if result is Ellipsis:
        check("小 batch 过拟合", Ellipsis, True)
        return

    history, accuracy = result
    check("损失历史形状", tuple(history.shape), (200,))
    check("损失显著下降", bool(history[-1] < history[0] * 0.1), True)
    check("小 batch 最终准确率", accuracy, 1.0)


def exercise_6_checkpoint() -> None:
    print("\n=== 练习 6：项目检查点 ===")
    torch.manual_seed(13)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = {"hidden_dim": 128, "dropout": 0.2, "learning_rate": 1e-3}
    checkpoint = safe_call(
        "构建项目检查点",
        build_checkpoint,
        model,
        optimizer,
        3,
        0.42,
        config,
    )
    if checkpoint is Ellipsis:
        check("构建项目检查点", Ellipsis, True)
        return

    check(
        "检查点字段",
        set(checkpoint),
        {"model_state", "optimizer_state", "epoch", "best_val_loss", "config"},
    )
    original = deepcopy(checkpoint["model_state"])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(5.0)
    check(
        "内存检查点不随模型变化",
        all(torch.equal(original[name], checkpoint["model_state"][name]) for name in original),
        True,
    )

    restored = nn.Linear(2, 2)
    restored.load_state_dict(checkpoint["model_state"])
    fixed_input = torch.randn(4, 2)
    expected_model = nn.Linear(2, 2)
    expected_model.load_state_dict(original)
    check("恢复后固定输入输出一致", restored(fixed_input), expected_model(fixed_input))


def main() -> None:
    print("第 28 天：FashionMNIST MLP 项目骨架练习")
    print("本脚本不下载数据；请把每道题中的 ... 替换为自己的实现。")
    exercise_1_model()
    exercise_2_split()
    exercise_3_training_engine()
    exercise_4_evaluation()
    exercise_5_tiny_batch_overfit()
    exercise_6_checkpoint()


if __name__ == "__main__":
    main()

"""第 27 天：评估、检查点与复现练习。"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn


Batch = tuple[torch.Tensor, torch.Tensor]


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


class DropoutClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy、PyTorch CPU 和可用 CUDA 的随机种子。"""
    # TODO：依次设置 random、numpy、torch；CUDA 可用时设置全部 CUDA 设备。
    ...


def evaluate(
    model: nn.Module,
    batches: Iterable[Batch],
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """在 eval + inference_mode 下返回加权 loss 与 accuracy。"""
    # TODO：遍历所有 batch；空数据应 raise ValueError。
    return ...


def should_replace_best(
    metric: float,
    best_metric: float,
    mode: str,
) -> bool:
    """mode='min' 时越小越好，mode='max' 时越大越好。"""
    # TODO：非法 mode 应 raise ValueError。
    return ...


def build_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_metric: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """构建不会随后续训练变化的内存检查点。"""
    # TODO：使用 deepcopy 保存 model_state、optimizer_state、epoch、最佳指标和配置。
    return ...


def restore_checkpoint(
    checkpoint: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float, dict[str, Any]]:
    """恢复状态，返回下一轮编号、最佳指标和配置。"""
    # TODO：加载两个 state_dict；下一轮为 checkpoint epoch + 1。
    return ...


def exercise_1_modes() -> None:
    print("\n=== 练习 1：训练与评估模式 ===")
    seed_everything(42)
    model = DropoutClassifier()
    x = torch.ones(32, 4)

    model.train()
    train_a = model(x)
    train_b = model(x)
    check("训练模式 Dropout 产生随机输出", bool(not torch.equal(train_a, train_b)), True)

    model.eval()
    with torch.inference_mode():
        eval_a = model(x)
        eval_b = model(x)
    check("评估模式输出稳定", eval_a, eval_b)
    check("推理输出不需要梯度", eval_a.requires_grad, False)


def exercise_2_evaluate() -> None:
    print("\n=== 练习 2：评估循环 ===")
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
        model.bias.zero_()

    batches = [
        (
            torch.tensor([[3.0, 1.0], [1.0, 4.0], [5.0, 0.0]]),
            torch.tensor([0, 1, 0]),
        ),
        (
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([0]),
        ),
    ]
    criterion = nn.CrossEntropyLoss()
    result = safe_call(
        "评估循环",
        evaluate,
        model,
        batches,
        criterion,
        torch.device("cpu"),
    )
    if result is Ellipsis:
        check("评估循环", Ellipsis, True)
        return

    all_x = torch.cat([batch[0] for batch in batches])
    all_y = torch.cat([batch[1] for batch in batches])
    expected_loss = criterion(model(all_x), all_y).item()
    expected_accuracy = (model(all_x).argmax(dim=1) == all_y).float().mean().item()
    check("按样本加权验证损失", result["loss"], expected_loss)
    check("验证准确率", result["accuracy"], expected_accuracy)
    check("evaluate 切换到 eval 模式", model.training, False)
    check(
        "评估没有写入梯度",
        all(parameter.grad is None for parameter in model.parameters()),
        True,
    )


def exercise_3_best_metric() -> None:
    print("\n=== 练习 3：最佳指标判断 ===")
    check("loss 下降时替换", should_replace_best(0.4, 0.5, "min"), True)
    check("loss 上升时不替换", should_replace_best(0.6, 0.5, "min"), False)
    check("accuracy 上升时替换", should_replace_best(0.9, 0.8, "max"), True)
    check("accuracy 下降时不替换", should_replace_best(0.7, 0.8, "max"), False)


def exercise_4_reproducibility() -> None:
    print("\n=== 练习 4：随机种子 ===")
    safe_call("设置随机种子 A", seed_everything, 7)
    first = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )
    safe_call("设置随机种子 B", seed_everything, 7)
    second = (
        random.random(),
        float(np.random.random()),
        torch.rand(3),
    )
    check("Python 随机数可复现", first[0], second[0])
    check("NumPy 随机数可复现", first[1], second[1])
    check("PyTorch 随机数可复现", first[2], second[2])


def exercise_5_checkpoint() -> None:
    print("\n=== 练习 5：检查点与断点续训 ===")
    seed_everything(11)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    x = torch.tensor([[1.0, 2.0], [2.0, -1.0]])
    y = torch.tensor([1, 0])

    loss = nn.functional.cross_entropy(model(x), y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    config = {"input_dim": 2, "output_dim": 2, "learning_rate": 0.1}
    checkpoint = safe_call(
        "构建检查点",
        build_checkpoint,
        model,
        optimizer,
        4,
        0.75,
        config,
    )
    if checkpoint is Ellipsis:
        check("构建检查点", Ellipsis, True)
        return

    expected_keys = {
        "model_state",
        "optimizer_state",
        "epoch",
        "best_val_metric",
        "config",
    }
    check("检查点字段", set(checkpoint), expected_keys)

    saved_state = deepcopy(model.state_dict())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)

    restored_model = nn.Linear(2, 2)
    restored_optimizer = torch.optim.SGD(
        restored_model.parameters(),
        lr=0.1,
        momentum=0.9,
    )
    result = safe_call(
        "恢复检查点",
        restore_checkpoint,
        checkpoint,
        restored_model,
        restored_optimizer,
    )
    if result is Ellipsis:
        check("恢复检查点", Ellipsis, True)
        return

    next_epoch, best_metric, restored_config = result
    check("下一轮编号", next_epoch, 5)
    check("恢复最佳指标", best_metric, 0.75)
    check("恢复配置", restored_config, config)
    for name, value in restored_model.state_dict().items():
        check(f"恢复参数 {name}", value, saved_state[name])
    check(
        "恢复优化器动量状态",
        len(restored_optimizer.state_dict()["state"]) > 0,
        True,
    )


def main() -> None:
    print("第 27 天：评估、检查点与复现练习")
    print("请把每道题中的 ... 替换为自己的 PyTorch 实现。")
    exercise_1_modes()
    exercise_2_evaluate()
    exercise_3_best_metric()
    exercise_4_reproducibility()
    exercise_5_checkpoint()


if __name__ == "__main__":
    main()

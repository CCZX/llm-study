"""第 26 天：Dataset 与 DataLoader 练习。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset


def check(name: str, actual: Any, expected: Any) -> None:
    if actual is Ellipsis:
        print(f"[待完成] {name}")
        return

    try:
        if isinstance(expected, torch.Tensor):
            passed = (
                isinstance(actual, torch.Tensor)
                and actual.shape == expected.shape
                and torch.equal(actual.cpu(), expected.cpu())
            )
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


class FeatureDataset(Dataset):
    """保存二维特征和一维分类标签。"""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        # TODO：数量不一致时 raise ValueError。
        # TODO：特征保存为 float32，标签保存为 long。
        self.features = ...
        self.labels = ...

    def __len__(self) -> int:
        # TODO：返回样本数量。
        return ...

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO：返回一个特征 Tensor 与一个标签 Tensor。
        return ...


def split_indices(
    sample_count: int,
    train_size: int,
    valid_size: int,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """随机划分索引，剩余样本全部进入测试集。"""
    # TODO：校验数量，并使用带固定种子的 torch.randperm。
    return ...


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader:
    """创建 num_workers=0、drop_last=False 的可复现 DataLoader。"""
    # TODO：创建独立 torch.Generator，设置种子并传给 DataLoader。
    return ...


def count_seen_samples(loader: DataLoader) -> int:
    """遍历 DataLoader，返回实际见到的样本总数。"""
    # TODO：不能直接返回 len(loader.dataset)，要真正遍历 batch。
    return ...


def exercise_1_custom_dataset() -> None:
    print("\n=== 练习 1：自定义 Dataset ===")
    features = torch.arange(20).reshape(5, 4)
    labels = torch.tensor([0.0, 1.0, 2.0, 1.0, 0.0])
    dataset = safe_call("创建自定义 Dataset", FeatureDataset, features, labels)
    if dataset is Ellipsis:
        check("创建自定义 Dataset", Ellipsis, True)
        return

    length = safe_call("Dataset 长度", len, dataset)
    check("Dataset 长度", length, 5)
    sample = safe_call("读取单样本", dataset.__getitem__, 2)
    if sample is not Ellipsis:
        sample_x, sample_y = sample
        check("单样本特征", sample_x, torch.tensor([8.0, 9.0, 10.0, 11.0]))
        check("单样本标签", sample_y, torch.tensor(2))
        check("特征 dtype", sample_x.dtype, torch.float32)
        check("标签 dtype", sample_y.dtype, torch.long)


def exercise_2_split() -> None:
    print("\n=== 练习 2：数据划分 ===")
    result = safe_call("划分索引", split_indices, 20, 12, 4, 7)
    if result is Ellipsis:
        check("划分索引", Ellipsis, True)
        return

    train, valid, test = result
    check("训练集大小", len(train), 12)
    check("验证集大小", len(valid), 4)
    check("测试集大小", len(test), 4)
    all_indices = torch.cat([train, valid, test])
    check("覆盖全部样本", len(torch.unique(all_indices)), 20)
    repeated = split_indices(20, 12, 4, 7)
    check("固定种子可复现", torch.equal(train, repeated[0]), True)


def exercise_3_loader_batches() -> None:
    print("\n=== 练习 3：DataLoader batch ===")
    dataset = FeatureDataset(
        torch.arange(40).reshape(10, 4),
        torch.arange(10) % 3,
    )
    if dataset.features is Ellipsis or dataset.labels is Ellipsis:
        check("DataLoader batch", Ellipsis, True)
        return

    loader = safe_call("创建顺序 DataLoader", make_loader, dataset, 4, False, 42)
    if loader is Ellipsis:
        check("创建顺序 DataLoader", Ellipsis, True)
        return

    batches = list(loader)
    check("batch 数量", len(batches), 3)
    check("第一个 batch 特征形状", tuple(batches[0][0].shape), (4, 4))
    check("第一个 batch 标签形状", tuple(batches[0][1].shape), (4,))
    check("最后一个 batch 大小", len(batches[-1][1]), 2)
    check("drop_last 为 False", loader.drop_last, False)
    check("num_workers 为 0", loader.num_workers, 0)


def exercise_4_reproducible_shuffle() -> None:
    print("\n=== 练习 4：可复现 shuffle ===")
    dataset = FeatureDataset(
        torch.arange(40).reshape(10, 4),
        torch.arange(10),
    )
    if dataset.features is Ellipsis or dataset.labels is Ellipsis:
        check("可复现 shuffle", Ellipsis, True)
        return

    loader_a = make_loader(dataset, batch_size=3, shuffle=True, seed=9)
    loader_b = make_loader(dataset, batch_size=3, shuffle=True, seed=9)
    if loader_a is Ellipsis or loader_b is Ellipsis:
        check("可复现 shuffle", Ellipsis, True)
        return

    order_a = torch.cat([labels for _, labels in loader_a])
    order_b = torch.cat([labels for _, labels in loader_b])
    check("相同种子的第一次顺序一致", order_a, order_b)
    check("shuffle 后仍覆盖所有标签", torch.sort(order_a).values, torch.arange(10))


def exercise_5_subsets_and_coverage() -> None:
    print("\n=== 练习 5：Subset 与样本覆盖 ===")
    dataset = FeatureDataset(
        torch.arange(80).reshape(20, 4),
        torch.arange(20) % 3,
    )
    if dataset.features is Ellipsis or dataset.labels is Ellipsis:
        check("Subset 与覆盖", Ellipsis, True)
        return

    train_indices, valid_indices, test_indices = split_indices(20, 12, 4, seed=5)
    if train_indices is Ellipsis:
        check("Subset 与覆盖", Ellipsis, True)
        return

    train_set = Subset(dataset, train_indices.tolist())
    valid_set = Subset(dataset, valid_indices.tolist())
    test_set = Subset(dataset, test_indices.tolist())

    train_loader = make_loader(train_set, 5, True, seed=5)
    valid_loader = make_loader(valid_set, 3, False, seed=5)
    test_loader = make_loader(test_set, 3, False, seed=5)

    check("训练样本全部看到", count_seen_samples(train_loader), 12)
    check("验证样本全部看到", count_seen_samples(valid_loader), 4)
    check("测试样本全部看到", count_seen_samples(test_loader), 4)

    train_ids = set(train_indices.tolist())
    valid_ids = set(valid_indices.tolist())
    test_ids = set(test_indices.tolist())
    check("训练/验证无重叠", train_ids.isdisjoint(valid_ids), True)
    check("训练/测试无重叠", train_ids.isdisjoint(test_ids), True)
    check("验证/测试无重叠", valid_ids.isdisjoint(test_ids), True)


def main() -> None:
    print("第 26 天：Dataset 与 DataLoader 练习")
    print("请把每道题中的 ... 替换为自己的 PyTorch 实现。")
    exercise_1_custom_dataset()
    exercise_2_split()
    exercise_3_loader_batches()
    exercise_4_reproducible_shuffle()
    exercise_5_subsets_and_coverage()


if __name__ == "__main__":
    main()

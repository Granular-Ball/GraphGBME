"""Deterministic epoch-wise shuffling of the labeled training nodes."""

import math
from collections.abc import Iterator

import torch


class ShuffledSeedSampler:
    def __init__(
        self, train_idx: torch.Tensor, seed_batch_size: int, random_seed: int
    ) -> None:
        if seed_batch_size <= 0:
            raise ValueError("seed_batch_size must be positive")
        self.train_idx = torch.unique(train_idx.detach().cpu().long(), sorted=True)
        self.seed_batch_size = seed_batch_size
        self.random_seed = random_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(self.train_idx.numel() / self.seed_batch_size)

    def __iter__(self) -> Iterator[torch.Tensor]:
        generator = torch.Generator(device="cpu").manual_seed(self.random_seed + self.epoch)
        order = torch.randperm(self.train_idx.numel(), generator=generator)
        shuffled = self.train_idx.index_select(0, order)
        for start in range(0, shuffled.numel(), self.seed_batch_size):
            yield shuffled[start : start + self.seed_batch_size]

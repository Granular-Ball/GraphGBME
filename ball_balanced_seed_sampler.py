"""Class-aware, granular-ball-balanced sampling of training seed nodes only."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch


class _CyclicNodePool:
    """Uniform shuffled traversal with reshuffling only after exhaustion."""

    def __init__(self, nodes: torch.Tensor, generator: torch.Generator) -> None:
        self.nodes = torch.unique(nodes.long().cpu(), sorted=True)
        self.node_set = set(self.nodes.tolist())
        self.generator = generator
        self.order = self.nodes.clone()
        self.position = 0
        self.reshuffle()

    def reshuffle(self) -> None:
        if self.nodes.numel():
            permutation = torch.randperm(self.nodes.numel(), generator=self.generator)
            self.order = self.nodes[permutation]
        self.position = 0

    def next(self) -> int:
        if self.nodes.numel() == 0:
            raise RuntimeError("cannot draw from an empty node pool")
        if self.position == self.order.numel():
            self.reshuffle()
        node = int(self.order[self.position])
        self.position += 1
        return node

    def draw(self, count: int, excluded: set[int]) -> list[int]:
        if count <= 0:
            return []
        if self.nodes.numel() == 0:
            raise RuntimeError("cannot draw from an empty node pool")

        result: list[int] = []
        available_count = sum(node not in excluded for node in self.node_set)
        while len(result) < count:
            if available_count == 0:
                repeated = self._draw_raw(count - len(result))
                result.extend(repeated)
                excluded.update(repeated)
                break

            if self.position == self.order.numel():
                self.reshuffle()
            needed = count - len(result)

            # Fast common path: no member of this pool has yet been used in the
            # batch, so the next contiguous portion of the current permutation
            # is accepted verbatim. If this reaches the cycle boundary, the next
            # loop iteration uses ordered filtering after reshuffle, exactly as
            # the legacy per-node implementation did.
            if available_count == self.nodes.numel():
                take = min(needed, self.order.numel() - self.position)
                accepted = self.order[self.position : self.position + take].tolist()
                self.position += take
                result.extend(accepted)
                excluded.update(accepted)
                available_count -= take
                continue

            remaining = self.order[self.position :].tolist()
            consumed = 0
            accepted: list[int] = []
            for node in remaining:
                consumed += 1
                if node not in excluded:
                    accepted.append(node)
                    if len(accepted) == needed:
                        break
            self.position += consumed
            result.extend(accepted)
            excluded.update(accepted)
            available_count -= len(accepted)
        return result

    def _draw_raw(self, count: int) -> list[int]:
        """Return the next ``count`` cyclic nodes using contiguous slices."""
        chunks: list[torch.Tensor] = []
        remaining_count = count
        while remaining_count:
            if self.position == self.order.numel():
                self.reshuffle()
            take = min(remaining_count, self.order.numel() - self.position)
            chunks.append(self.order[self.position : self.position + take])
            self.position += take
            remaining_count -= take
        if not chunks:
            return []
        return torch.cat(chunks).tolist()


class BallBalancedSeedSampler:
    """Yield class-aware seed batches without modifying neighbor sampling.

    Minority status is inferred exclusively from ``y[train_idx]``. Multi-node
    granular balls receive equal round-robin access; their size never determines
    ball selection probability. Singleton balls are merged into one uniform pool.
    """

    def __init__(
        self,
        train_idx: torch.Tensor,
        y: torch.Tensor,
        ball_members: Sequence[torch.Tensor | Sequence[int]],
        seed_batch_size: int,
        minority_ratio: float = 0.25,
        random_seed: int = 42,
    ) -> None:
        if seed_batch_size <= 0:
            raise ValueError("seed_batch_size must be positive")
        if not 0.0 <= minority_ratio <= 1.0:
            raise ValueError("minority_ratio must be in [0, 1]")

        self.train_idx = torch.unique(train_idx.long().cpu(), sorted=True)
        if self.train_idx.numel() == 0:
            raise ValueError("train_idx must not be empty")
        y = y.long().cpu()
        if int(self.train_idx.min()) < 0 or int(self.train_idx.max()) >= y.numel():
            raise IndexError("train_idx contains an invalid node index")

        train_labels = y.index_select(0, self.train_idx)
        classes, counts = torch.unique(train_labels, sorted=True, return_counts=True)
        self.minority_class = int(classes[counts.argmin()])
        minority_mask = train_labels == self.minority_class
        self.minority_nodes = self.train_idx[minority_mask]
        self.majority_nodes = self.train_idx[~minority_mask]
        self.seed_batch_size = seed_batch_size
        self.minority_ratio = minority_ratio
        self.random_seed = random_seed
        self.num_batches = math.ceil(self.train_idx.numel() / seed_batch_size)

        minority_set = set(self.minority_nodes.tolist())
        assigned: set[int] = set()
        multi_balls: list[torch.Tensor] = []
        singleton_nodes: list[int] = []
        for members in ball_members:
            member_tensor = torch.unique(
                torch.as_tensor(members, dtype=torch.long).flatten(), sorted=True
            )
            valid = [
                int(node)
                for node in member_tensor.tolist()
                if int(node) in minority_set and int(node) not in assigned
            ]
            if len(valid) >= 2:
                multi_balls.append(torch.tensor(valid, dtype=torch.long))
            elif len(valid) == 1:
                singleton_nodes.append(valid[0])
            assigned.update(valid)

        # Robustly retain any training-minority node omitted from ball_members.
        singleton_nodes.extend(sorted(minority_set.difference(assigned)))
        self.multi_node_balls = multi_balls
        self.singleton_pool = torch.tensor(singleton_nodes, dtype=torch.long)

        self._generator = torch.Generator(device="cpu")
        self._epoch = -1
        self._ball_pools: list[_CyclicNodePool] = []
        self._ball_order: list[int] = []
        self._ball_position = 0
        self._singleton_sampler: _CyclicNodePool
        self._majority_sampler: _CyclicNodePool
        self.set_epoch(0)

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        """Reset and deterministically shuffle all sampling state for an epoch."""
        self._epoch = int(epoch)
        self._generator.manual_seed(self.random_seed + self._epoch)
        self._ball_pools = [
            _CyclicNodePool(members, self._generator)
            for members in self.multi_node_balls
        ]
        if self._ball_pools:
            order = torch.randperm(len(self._ball_pools), generator=self._generator)
            self._ball_order = order.tolist()
        else:
            self._ball_order = []
        self._ball_position = 0
        self._singleton_sampler = _CyclicNodePool(
            self.singleton_pool, self._generator
        )
        self._majority_sampler = _CyclicNodePool(
            self.majority_nodes, self._generator
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        for _ in range(self.num_batches):
            yield self._next_batch()

    def _next_multi_node(self) -> int:
        if not self._ball_order:
            raise RuntimeError("cannot draw from an empty multi-node ball pool")
        ball_index = self._ball_order[self._ball_position]
        self._ball_position = (self._ball_position + 1) % len(self._ball_order)
        return self._ball_pools[ball_index].next()

    def _draw_multi(self, count: int, excluded: set[int]) -> list[int]:
        result: list[int] = []
        for _ in range(count):
            # Never skip to a larger ball merely because a small ball has begun
            # a new member cycle: equal ball visitation takes precedence. Each
            # ball itself is without replacement until all its members are used.
            node = self._next_multi_node()
            result.append(node)
            excluded.add(node)
        return result

    def _class_quotas(self) -> tuple[int, int]:
        desired_minority = math.floor(self.seed_batch_size * self.minority_ratio)
        minority = min(desired_minority, self.minority_nodes.numel())
        majority = min(
            self.seed_batch_size - minority, self.majority_nodes.numel()
        )

        missing = self.seed_batch_size - minority - majority
        if missing:
            minority_capacity = max(0, self.minority_nodes.numel() - minority)
            addition = min(missing, minority_capacity)
            minority += addition
            missing -= addition
        if missing:
            majority_capacity = max(0, self.majority_nodes.numel() - majority)
            addition = min(missing, majority_capacity)
            majority += addition
            missing -= addition
        if missing:
            # Total unique training nodes are fewer than one batch. Repeats are
            # unavoidable; use any non-empty class while preserving ratio when possible.
            if self.minority_nodes.numel() and not self.majority_nodes.numel():
                minority += missing
            elif self.majority_nodes.numel() and not self.minority_nodes.numel():
                majority += missing
            else:
                repeated_minority = math.floor(missing * self.minority_ratio)
                minority += repeated_minority
                majority += missing - repeated_minority
        return minority, majority

    def _minority_pool_quotas(self, minority_count: int) -> tuple[int, int]:
        multi_size = sum(ball.numel() for ball in self.multi_node_balls)
        singleton_size = self.singleton_pool.numel()
        total = multi_size + singleton_size
        if minority_count == 0 or total == 0:
            return 0, 0
        # Largest-remainder allocation for the two pools; unlike always taking
        # floor, this does not erase a smaller pool's proportional share.
        multi_exact = minority_count * multi_size / total
        multi = math.floor(multi_exact + 0.5)
        singleton = minority_count - multi

        # Prefer unique nodes within the batch and transfer shortages.
        if multi > multi_size:
            singleton += multi - multi_size
            multi = multi_size
        if singleton > singleton_size:
            multi += singleton - singleton_size
            singleton = singleton_size
        return multi, singleton

    def _next_batch(self) -> torch.Tensor:
        minority_count, majority_count = self._class_quotas()
        multi_count, singleton_count = self._minority_pool_quotas(minority_count)
        excluded: set[int] = set()
        seeds = self._draw_multi(multi_count, excluded)
        seeds.extend(self._singleton_sampler.draw(singleton_count, excluded))
        seeds.extend(self._majority_sampler.draw(majority_count, excluded))

        if len(seeds) != self.seed_batch_size:
            raise RuntimeError(
                f"sampler produced {len(seeds)} seeds, expected {self.seed_batch_size}"
            )
        permutation = torch.randperm(len(seeds), generator=self._generator)
        return torch.tensor(seeds, dtype=torch.long)[permutation]


__all__ = ["BallBalancedSeedSampler"]

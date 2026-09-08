from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import torch


class TwoHopMinorityHypergraph:
    """Implicit distance-two minority graph stored as endpoint groups.

    Every hyperedge represents either an original minority-to-minority edge or
    all minority endpoints sharing one intermediate node. Expanding a group as
    a clique would produce the ordinary two-hop adjacency, but keeping the group
    implicit avoids quadratic memory growth around high-degree intermediates.
    """

    def __init__(
        self,
        num_nodes: int,
        edge_offsets: torch.Tensor,
        endpoints: torch.Tensor,
        node_offsets: torch.Tensor,
        node_edges: torch.Tensor,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.edge_offsets = edge_offsets.long().cpu()
        self.endpoints = endpoints.long().cpu()
        self.node_offsets = node_offsets.long().cpu()
        self.node_edges = node_edges.long().cpu()

    def __len__(self) -> int:
        return self.num_nodes

    def _incident_edges(self, node: int) -> torch.Tensor:
        start, end = int(self.node_offsets[node]), int(self.node_offsets[node + 1])
        return self.node_edges[start:end]

    def _edge_endpoints(self, edge: int) -> torch.Tensor:
        start, end = int(self.edge_offsets[edge]), int(self.edge_offsets[edge + 1])
        return self.endpoints[start:end]

    def neighbors(self, node: int) -> list[int]:
        """Materialize one node's neighbors; intended for tests and diagnostics."""
        result: set[int] = set()
        for edge in self._incident_edges(node).tolist():
            result.update(self._edge_endpoints(edge).tolist())
        result.discard(node)
        return sorted(result)

    def connected_components(self) -> list[list[int]]:
        components: list[list[int]] = []
        visited = bytearray(self.num_nodes)
        edge_seen = bytearray(self.edge_offsets.numel() - 1)
        for start in range(self.num_nodes):
            if visited[start]:
                continue
            visited[start] = 1
            stack = [start]
            component: list[int] = []
            while stack:
                node = stack.pop()
                component.append(node)
                for edge in self._incident_edges(node).tolist():
                    if edge_seen[edge]:
                        continue
                    edge_seen[edge] = 1
                    for neighbor in self._edge_endpoints(edge).tolist():
                        if not visited[neighbor]:
                            visited[neighbor] = 1
                            stack.append(neighbor)
            components.append(sorted(component))
        return components

    def frontier_candidates(
        self,
        frontiers: list[set[int]],
        allowed: set[int],
        assigned: dict[int, int],
        processed_edges: set[int],
    ) -> dict[int, set[int]]:
        """Expand one synchronous BFS layer without materializing clique edges."""
        edge_balls: dict[int, set[int]] = {}
        for ball_id, frontier in enumerate(frontiers):
            for source in sorted(frontier):
                for edge in self._incident_edges(source).tolist():
                    if edge not in processed_edges:
                        edge_balls.setdefault(edge, set()).add(ball_id)

        candidates: dict[int, set[int]] = {}
        for edge, ball_ids in edge_balls.items():
            processed_edges.add(edge)
            for node in self._edge_endpoints(edge).tolist():
                if node in allowed and node not in assigned:
                    candidates.setdefault(node, set()).update(ball_ids)
        return candidates


def build_two_hop_minority_hypergraph(
    edge_index: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    num_nodes: int,
    minority_train_idx: torch.Tensor | Sequence[int],
) -> TwoHopMinorityHypergraph:
    """Build the exact distance-<=2 minority graph without expanding cliques."""
    src, dst = edge_index if isinstance(edge_index, tuple) else edge_index
    src = src.detach().cpu().long().flatten()
    dst = dst.detach().cpu().long().flatten()
    minority = torch.as_tensor(minority_train_idx, dtype=torch.long).cpu().flatten()
    if src.numel() != dst.numel():
        raise ValueError("edge endpoints have inconsistent lengths")
    if minority.unique().numel() != minority.numel():
        raise ValueError("minority_train_idx must not contain duplicates")
    if minority.numel() and (int(minority.min()) < 0 or int(minority.max()) >= num_nodes):
        raise IndexError("minority_train_idx contains an invalid node id")

    minority_count = minority.numel()
    if minority_count == 0:
        offsets = torch.zeros(1, dtype=torch.long)
        return TwoHopMinorityHypergraph(0, offsets, torch.empty(0, dtype=torch.long), offsets, torch.empty(0, dtype=torch.long))

    local_id = torch.full((num_nodes,), -1, dtype=torch.long)
    local_id[minority] = torch.arange(minority_count)
    src_local, dst_local = local_id[src], local_id[dst]

    # (intermediate node, local minority endpoint) incidence pairs.
    selected_src, selected_dst = src_local >= 0, dst_local >= 0
    intermediates = torch.cat((dst[selected_src], src[selected_dst]))
    endpoint_ids = torch.cat((src_local[selected_src], dst_local[selected_dst]))
    incidence_keys = intermediates * minority_count + endpoint_ids
    incidence_keys = torch.unique(incidence_keys, sorted=True)
    sorted_intermediates = torch.div(incidence_keys, minority_count, rounding_mode="floor")
    sorted_endpoints = incidence_keys.remainder(minority_count)
    _, group_counts = torch.unique_consecutive(sorted_intermediates, return_counts=True)
    group_keep = group_counts >= 2
    kept_group_counts = group_counts[group_keep]
    kept_group_endpoints = sorted_endpoints[
        torch.repeat_interleave(group_keep, group_counts)
    ]

    # Direct minority edges are two-endpoint hyperedges. They are required even
    # when neither endpoint group above contains both nodes.
    direct = (src_local >= 0) & (dst_local >= 0) & (src_local != dst_local)
    direct_u = torch.minimum(src_local[direct], dst_local[direct])
    direct_v = torch.maximum(src_local[direct], dst_local[direct])
    direct_keys = torch.unique(direct_u * minority_count + direct_v, sorted=True)
    if direct_keys.numel():
        direct_parts = torch.stack(
            (
                torch.div(direct_keys, minority_count, rounding_mode="floor"),
                direct_keys.remainder(minority_count),
            ),
            dim=1,
        )
        edge_counts = torch.cat(
            (kept_group_counts, torch.full((direct_parts.shape[0],), 2, dtype=torch.long))
        )
        endpoints = torch.cat((kept_group_endpoints, direct_parts.flatten()))
    else:
        edge_counts = kept_group_counts
        endpoints = kept_group_endpoints
    edge_offsets = torch.cat((torch.zeros(1, dtype=torch.long), edge_counts.cumsum(0)))

    if endpoints.numel():
        edge_ids = torch.repeat_interleave(torch.arange(edge_counts.numel()), edge_counts)
        order = torch.argsort(endpoints)
        sorted_nodes, node_edges = endpoints[order], edge_ids[order]
        node_counts = torch.bincount(sorted_nodes, minlength=minority_count)
    else:
        node_edges = torch.empty(0, dtype=torch.long)
        node_counts = torch.zeros(minority_count, dtype=torch.long)
    node_offsets = torch.cat((torch.zeros(1, dtype=torch.long), node_counts.cumsum(0)))
    return TwoHopMinorityHypergraph(
        minority_count, edge_offsets, endpoints, node_offsets, node_edges
    )


@dataclass(frozen=True)
class BallGeometry:
    center: torch.Tensor
    radius: float
    radius90: float
    quality: float


def compute_ball_geometry(
    features: torch.Tensor,
    members: torch.Tensor | Sequence[int],
) -> BallGeometry:
    """Compute the radius/quality definition used by the original generator."""
    indices = torch.as_tensor(members, dtype=torch.long, device=features.device).flatten()
    if indices.numel() == 0:
        raise ValueError("a granular ball cannot be empty")
    member_features = features.index_select(0, indices)
    center = member_features.mean(dim=0)
    distances = torch.linalg.vector_norm(member_features - center, dim=1)
    radius = distances.mean() / math.sqrt(member_features.shape[1])
    radius90 = torch.quantile(distances, 0.9)
    quality = 1.0 / (1.0 + radius)
    return BallGeometry(
        center=center,
        radius=float(radius.item()),
        radius90=float(radius90.item()),
        quality=float(quality.item()),
    )


def build_two_hop_minority_adjacency(
    edge_index: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    num_nodes: int,
    minority_train_idx: torch.Tensor | Sequence[int],
) -> list[list[int]]:
    """Build an undirected minority graph for original-graph distance <= 2."""
    if isinstance(edge_index, tuple):
        if len(edge_index) != 2:
            raise ValueError("edge_index tuple must contain (src, dst)")
        src, dst = edge_index
    else:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        src, dst = edge_index[0], edge_index[1]
    src = src.detach().cpu().long().flatten()
    dst = dst.detach().cpu().long().flatten()
    if src.numel() != dst.numel():
        raise ValueError("edge endpoints have inconsistent lengths")

    minority = torch.as_tensor(minority_train_idx, dtype=torch.long).cpu().flatten()
    if minority.unique().numel() != minority.numel():
        raise ValueError("minority_train_idx must not contain duplicates")
    if minority.numel() and (int(minority.min()) < 0 or int(minority.max()) >= num_nodes):
        raise IndexError("minority_train_idx contains an invalid node id")
    adjacency = [set() for _ in range(minority.numel())]
    if minority.numel() == 0:
        return []

    local_id = torch.full((num_nodes,), -1, dtype=torch.long)
    local_id[minority] = torch.arange(minority.numel())
    intermediate_chunks: list[torch.Tensor] = []
    endpoint_chunks: list[torch.Tensor] = []
    for start in range(0, src.numel(), 1_000_000):
        chunk_src = src[start : start + 1_000_000]
        chunk_dst = dst[start : start + 1_000_000]
        src_local, dst_local = local_id[chunk_src], local_id[chunk_dst]

        direct = (src_local >= 0) & (dst_local >= 0) & (src_local != dst_local)
        for u, v in zip(src_local[direct].tolist(), dst_local[direct].tolist()):
            adjacency[u].add(v)
            adjacency[v].add(u)

        src_selected, dst_selected = src_local >= 0, dst_local >= 0
        intermediate_chunks.extend([chunk_dst[src_selected], chunk_src[dst_selected]])
        endpoint_chunks.extend([src_local[src_selected], dst_local[dst_selected]])

    if intermediate_chunks:
        intermediates = torch.cat(intermediate_chunks)
        endpoints = torch.cat(endpoint_chunks)
        order = torch.argsort(intermediates)
        intermediates, endpoints = intermediates[order], endpoints[order]
        starts = torch.cat([
            torch.tensor([0]),
            torch.nonzero(intermediates[1:] != intermediates[:-1]).flatten() + 1,
        ])
        ends = torch.cat([starts[1:], torch.tensor([len(intermediates)])])
        for start, end in zip(starts.tolist(), ends.tolist()):
            local_endpoints = sorted(set(endpoints[start:end].tolist()))
            for u, v in combinations(local_endpoints, 2):
                adjacency[u].add(v)
                adjacency[v].add(u)
    return [sorted(neighbors) for neighbors in adjacency]


@dataclass
class GranularBall:
    member_ids: torch.Tensor
    center: torch.Tensor | None
    quality: float | None
    depth: int = 0


class GranularBallBuilder:
    """Build graph-constrained granular balls with fixed-center BFS assignment."""

    def __init__(
        self,
        quality_threshold: float = 0.9,
        min_split_size: int = 4,
        random_seed: int = 42,
    ) -> None:
        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError("quality_threshold 必须在 [0, 1] 内")
        if min_split_size < 2:
            raise ValueError("min_split_size 必须至少为 2")
        self.quality_threshold = quality_threshold
        self.min_split_size = min_split_size
        self.random_seed = random_seed

    @staticmethod
    def initial_centers(
        features: torch.Tensor,
        k: int | None = None,
        random_seed: int = 42,
    ) -> list[int]:
        """Select data-point centers with deterministic k-medoids++ seeding."""
        n = features.shape[0]
        if n == 0:
            return []
        k = min(n, max(1, k if k is not None else int(math.sqrt(n) + 0.5)))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(random_seed)
        first = int(torch.randint(n, (1,), generator=generator).item())
        selected = [first]
        min_distance = torch.cdist(features, features[first : first + 1]).squeeze(1)
        min_distance[first] = 0
        while len(selected) < k:
            weights = min_distance.square().cpu()
            weights[selected] = 0
            if float(weights.sum()) <= 0.0:
                nxt = next(index for index in range(n) if index not in selected)
            else:
                nxt = int(torch.multinomial(weights, 1, generator=generator).item())
            selected.append(nxt)
            distance = torch.cdist(features, features[nxt : nxt + 1]).squeeze(1)
            min_distance = torch.minimum(min_distance, distance)
            min_distance[selected] = 0
        return selected

    @staticmethod
    def _quality(features: torch.Tensor, members: list[int]) -> tuple[float, torch.Tensor]:
        geometry = compute_ball_geometry(features, members)
        return geometry.quality, geometry.center

    def _grow(
        self,
        features: torch.Tensor,
        adjacency: list[list[int]] | TwoHopMinorityHypergraph,
        allowed: Iterable[int],
        seeds: list[int],
    ) -> list[list[int]]:
        allowed_set = set(int(i) for i in allowed)
        if not seeds:
            return []
        assignment = {seed: ball_id for ball_id, seed in enumerate(seeds)}
        members = [[seed] for seed in seeds]
        fixed_centers = features[torch.tensor(seeds, dtype=torch.long)]
        frontiers = [{seed} for seed in seeds]
        processed_edges: set[int] = set()

        while len(assignment) < len(allowed_set):
            if isinstance(adjacency, TwoHopMinorityHypergraph):
                candidates = adjacency.frontier_candidates(
                    frontiers, allowed_set, assignment, processed_edges
                )
            else:
                candidates: dict[int, set[int]] = {}
                for ball_id, frontier in enumerate(frontiers):
                    for source in sorted(frontier):
                        for node in adjacency[source]:
                            if node in allowed_set and node not in assignment:
                                candidates.setdefault(node, set()).add(ball_id)
            if not candidates:
                break

            layer_assignment: dict[int, int] = {}
            for node in sorted(candidates):
                feasible = sorted(candidates[node])
                layer_assignment[node] = min(
                    feasible,
                    key=lambda ball_id: (
                        float(torch.linalg.vector_norm(features[node] - fixed_centers[ball_id])),
                        ball_id,
                    ),
                )
            next_frontiers = [set() for _ in seeds]
            for node, ball_id in layer_assignment.items():
                assignment[node] = ball_id
                members[ball_id].append(node)
                next_frontiers[ball_id].add(node)
            frontiers = next_frontiers

        if len(assignment) != len(allowed_set):
            raise RuntimeError("同步多源 BFS 未能覆盖当前连通分量")
        return members

    @staticmethod
    def _connected_components(
        adjacency: list[list[int]] | TwoHopMinorityHypergraph,
    ) -> list[list[int]]:
        if isinstance(adjacency, TwoHopMinorityHypergraph):
            return adjacency.connected_components()
        components: list[list[int]] = []
        visited = [False] * len(adjacency)
        for start in range(len(adjacency)):
            if visited[start]:
                continue
            visited[start] = True
            stack = [start]
            component: list[int] = []
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbor in adjacency[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(sorted(component))
        return components

    @staticmethod
    def _split_seeds(features: torch.Tensor, members: list[int]) -> list[int]:
        member_tensor = torch.tensor(members, dtype=torch.long)
        local = features[member_tensor]
        distances = torch.cdist(local, local)
        flat = int(distances.argmax().item())
        n = len(members)
        return [members[flat // n], members[flat % n]]

    def _make_ball(
        self,
        features: torch.Tensor,
        members: Sequence[int],
        depth: int,
    ) -> GranularBall:
        quality, center = self._quality(features, list(members))
        # An initial singleton keeps its geometric compactness (quality 1),
        # while a singleton produced by recursive splitting is marked as zero.
        # Both remain ineligible as prototypes because the model requires at
        # least two members.
        if len(members) == 1 and depth > 0:
            quality = 0.0
        return GranularBall(
            member_ids=torch.tensor(sorted(members), dtype=torch.long),
            center=center,
            quality=quality,
            depth=depth,
        )

    def _split_recursively(
        self,
        features: torch.Tensor,
        adjacency: list[list[int]] | TwoHopMinorityHypergraph,
        parent: GranularBall,
    ) -> list[GranularBall]:
        """Recursively accept a binary split when child quality sum improves."""
        members = parent.member_ids.tolist()
        if parent.quality is None or parent.quality >= self.quality_threshold:
            return [parent]
        if len(members) < self.min_split_size:
            return [parent]
        groups = self._grow(
            features, adjacency, members, self._split_seeds(features, members)
        )
        groups = [group for group in groups if group]
        if len(groups) != 2:
            return [parent]
        children = [
            self._make_ball(features, group, parent.depth + 1)
            for group in groups
        ]
        child_quality_sum = sum(float(child.quality) for child in children)
        if child_quality_sum <= float(parent.quality):
            return [parent]
        leaves: list[GranularBall] = []
        for child in children:
            leaves.extend(self._split_recursively(features, adjacency, child))
        return leaves

    def build(
        self,
        features: torch.Tensor,
        adjacency: list[list[int]] | TwoHopMinorityHypergraph,
        k: int | None = None,
    ) -> list[GranularBall]:
        if features.ndim != 2:
            raise ValueError("features 必须是二维张量")
        if len(adjacency) != features.shape[0]:
            raise ValueError("adjacency 长度必须等于节点数")
        balls: list[GranularBall] = []
        for component_id, component in enumerate(self._connected_components(adjacency)):
            component_tensor = torch.tensor(component, dtype=torch.long)
            local_seeds = self.initial_centers(
                features[component_tensor],
                min(len(component), max(1, k if k is not None else int(math.sqrt(len(component)) + 0.5))),
                random_seed=self.random_seed + component_id,
            )
            seeds = [component[index] for index in local_seeds]
            groups = self._grow(features, adjacency, component, seeds)
            balls.extend(self._make_ball(features, group, 0) for group in groups)
        leaves: list[GranularBall] = []
        for ball in balls:
            leaves.extend(self._split_recursively(features, adjacency, ball))
        return leaves


__all__ = [
    "BallGeometry", "GranularBall", "GranularBallBuilder", "compute_ball_geometry",
    "TwoHopMinorityHypergraph", "build_two_hop_minority_adjacency",
    "build_two_hop_minority_hypergraph",
]

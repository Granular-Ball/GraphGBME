"""Granular-ball prototype enhancement for an existing GNN representation.

This module learns a fused residual from the local and mesoscopic signals after
ordinary neighborhood aggregation. It does not alter graph edges or neighborhood
message weights.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class GranularBallPrototypeEnhancement(nn.Module):
    """Enhance local GNN representations with reliable granular-ball prototypes.

    Args:
        hidden_dim: Dimension of ``h_local`` and ``h_input``.
        min_ball_size: Minimum number of members in a reliable ball. Singleton
            balls are always excluded, even if this value is set below two.
        quality_threshold: Minimum geometric quality of a reliable ball.
        top_k: Maximum number of feature-similar balls selected for each node.
        eps: Stabilizer in the selected-score normalization denominator.
        use_layer_norm: Apply LayerNorm after adding the prototype residual.
        allow_singletons: Permit one-member prototypes for the minority-node
            control method. Granular-ball enhancement should leave this false.

    ``selected_ball_indices`` returned by :meth:`forward` contains indices into
    the original ``ball_members`` and ``ball_centers`` inputs, rather than
    indices into the size-and-quality-filtered reliable subset.
    """

    def __init__(
        self,
        hidden_dim: int,
        min_ball_size: int = 2,
        quality_threshold: float = 0.0,
        top_k: int = 3,
        eps: float = 1e-8,
        use_layer_norm: bool = False,
        allow_singletons: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if min_ball_size <= 0:
            raise ValueError("min_ball_size must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.hidden_dim = hidden_dim
        # Granular-ball prototypes exclude singletons by default.  The point
        # enhancement control deliberately treats each minority node as one
        # singleton prototype, so it opts in explicitly.
        self.min_ball_size = min_ball_size if allow_singletons else max(2, min_ball_size)
        self.quality_threshold = quality_threshold
        self.top_k = top_k
        self.eps = eps
        self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        # Begin close to the previous residual behavior: projected input features
        # and their ball prototypes are non-negative, so the second identity block
        # initially forwards the prototype while training can learn both halves.
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.weight[:, hidden_dim:].copy_(torch.eye(hidden_dim))
            self.fusion.bias.zero_()
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

    def forward(
        self,
        z: torch.Tensor,
        h_local: torch.Tensor,
        h_input: torch.Tensor,
        ball_members: Sequence[torch.Tensor | Sequence[int]],
        ball_centers: torch.Tensor,
        ball_quality: torch.Tensor,
        query_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compatibility wrapper for callers with unprepared ball metadata.

        Performance-sensitive training uses :meth:`forward_prepared`, with the
        reliable subset and flattened membership prepared once by the model.
        """
        self._validate_shapes(
            z,
            h_local,
            h_input,
            ball_members,
            ball_centers,
            ball_quality,
            query_indices,
        )

        device, dtype = h_local.device, h_local.dtype
        z = z.to(device=device, dtype=dtype)
        h_input = h_input.to(device=device, dtype=dtype)
        if query_indices is not None:
            query_indices = query_indices.to(device=device, dtype=torch.long).flatten()
            z_query = z.index_select(0, query_indices)
        else:
            z_query = z

        reliable_original_indices: list[int] = []
        reliable_members: list[torch.Tensor] = []
        for ball_index, members in enumerate(ball_members):
            indices = torch.as_tensor(members, dtype=torch.long, device=device).flatten()
            if indices.numel() and (
                int(indices.min()) < 0 or int(indices.max()) >= h_input.shape[0]
            ):
                raise IndexError(f"ball_members[{ball_index}] contains an invalid node index")
            if (
                indices.numel() >= self.min_ball_size
                and bool(ball_quality[ball_index] >= self.quality_threshold)
            ):
                reliable_original_indices.append(ball_index)
                reliable_members.append(indices)

        original_indices = torch.tensor(
            reliable_original_indices,
            dtype=torch.long,
            device=device,
        )
        if reliable_members:
            member_counts = torch.tensor(
                [members.numel() for members in reliable_members],
                dtype=torch.long,
                device=device,
            )
            member_ball_ids = torch.repeat_interleave(
                torch.arange(len(reliable_members), device=device),
                member_counts,
            )
            flattened_members = torch.cat(reliable_members)
            member_hidden = h_input.index_select(0, flattened_members)
        else:
            member_counts = torch.empty(0, dtype=torch.long, device=device)
            member_ball_ids = torch.empty(0, dtype=torch.long, device=device)
            member_hidden = h_input.new_empty((0, h_input.shape[1]))

        return self.forward_prepared(
            z_query=z_query,
            h_local=h_local,
            member_hidden=member_hidden,
            member_ball_ids=member_ball_ids,
            member_counts=member_counts,
            reliable_centers=ball_centers.to(device=device, dtype=dtype).index_select(
                0, original_indices
            ),
            reliable_quality=ball_quality.to(device=device, dtype=dtype).index_select(
                0, original_indices
            ),
            reliable_original_indices=original_indices,
        )

    def forward_prepared(
        self,
        *,
        z_query: torch.Tensor,
        h_local: torch.Tensor,
        member_hidden: torch.Tensor,
        member_ball_ids: torch.Tensor,
        member_counts: torch.Tensor,
        reliable_centers: torch.Tensor,
        reliable_quality: torch.Tensor,
        reliable_original_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enhance a batch using prefiltered, flattened reliable-ball metadata."""
        self._validate_prepared_shapes(
            z_query=z_query,
            h_local=h_local,
            member_hidden=member_hidden,
            member_ball_ids=member_ball_ids,
            member_counts=member_counts,
            reliable_centers=reliable_centers,
            reliable_quality=reliable_quality,
            reliable_original_indices=reliable_original_indices,
        )
        if member_counts.numel() == 0:
            return self._empty_result(h_local)

        device, dtype = h_local.device, h_local.dtype
        z_query = z_query.to(device=device, dtype=dtype)
        member_hidden = member_hidden.to(device=device, dtype=dtype)
        member_ball_ids = member_ball_ids.to(device=device, dtype=torch.long)
        member_counts = member_counts.to(device=device, dtype=dtype)
        reliable_centers = reliable_centers.to(device=device, dtype=dtype)
        reliable_quality = reliable_quality.to(device=device, dtype=dtype)
        reliable_original_indices = reliable_original_indices.to(
            device=device,
            dtype=torch.long,
        )

        # One differentiable segmented reduction replaces one index_select/mean
        # kernel launch per reliable ball.
        prototypes = member_hidden.new_zeros(
            (member_counts.numel(), member_hidden.shape[1])
        ).index_add(0, member_ball_ids, member_hidden)
        prototypes = prototypes / member_counts.unsqueeze(1)
        feature_distances = torch.cdist(z_query, reliable_centers, p=2)
        scores = reliable_quality.unsqueeze(0) * torch.exp(-feature_distances)
        selected_count = min(self.top_k, scores.shape[1])
        selected_scores, selected_reliable_indices = torch.topk(
            scores, k=selected_count, dim=1, largest=True, sorted=True
        )
        selected_ball_indices = reliable_original_indices[selected_reliable_indices]
        weights = selected_scores / (selected_scores.sum(dim=1, keepdim=True) + self.eps)

        selected_prototypes = prototypes[selected_reliable_indices]
        mesoscopic = (weights.unsqueeze(-1) * selected_prototypes).sum(dim=1)
        gates = selected_scores.max(dim=1).values.clamp(0.0, 1.0)
        fused_residual = torch.relu(
            self.fusion(torch.cat([h_local, mesoscopic], dim=-1))
        )
        enhanced = self.layer_norm(
            h_local + gates.unsqueeze(-1) * fused_residual
        )

        self._check_finite(
            enhanced=enhanced,
            gates=gates,
            selected_scores=selected_scores,
            weights=weights,
        )
        return enhanced, gates, selected_ball_indices, selected_scores, weights

    def _validate_prepared_shapes(
        self,
        *,
        z_query: torch.Tensor,
        h_local: torch.Tensor,
        member_hidden: torch.Tensor,
        member_ball_ids: torch.Tensor,
        member_counts: torch.Tensor,
        reliable_centers: torch.Tensor,
        reliable_quality: torch.Tensor,
        reliable_original_indices: torch.Tensor,
    ) -> None:
        reliable_count = member_counts.numel()
        if z_query.ndim != 2 or z_query.shape[0] != h_local.shape[0]:
            raise ValueError("z_query must have shape [h_local.shape[0], feature_dim]")
        if h_local.ndim != 2 or h_local.shape[1] != self.hidden_dim:
            raise ValueError("h_local must have shape [N, hidden_dim]")
        if member_hidden.ndim != 2 or member_hidden.shape[1] != self.hidden_dim:
            raise ValueError("member_hidden must have shape [M, hidden_dim]")
        if member_ball_ids.ndim != 1 or member_ball_ids.numel() != member_hidden.shape[0]:
            raise ValueError("member_ball_ids must have shape [M]")
        if member_counts.ndim != 1:
            raise ValueError("member_counts must have shape [K]")
        if reliable_centers.ndim != 2 or reliable_centers.shape[0] != reliable_count:
            raise ValueError("reliable_centers must have shape [K, feature_dim]")
        if reliable_centers.shape[1] != z_query.shape[1]:
            raise ValueError("reliable_centers and z_query feature dimensions differ")
        if reliable_quality.shape != (reliable_count,):
            raise ValueError("reliable_quality must have shape [K]")
        if reliable_original_indices.shape != (reliable_count,):
            raise ValueError("reliable_original_indices must have shape [K]")

    def _empty_result(
        self, h_local: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the exact local representation when no reliable ball exists."""
        node_count = h_local.shape[0]
        empty_indices = torch.empty((node_count, 0), dtype=torch.long, device=h_local.device)
        empty_values = h_local.new_empty((node_count, 0))
        gates = h_local.new_zeros(node_count)
        self._check_finite(enhanced=h_local, gates=gates)
        return h_local, gates, empty_indices, empty_values, empty_values.clone()

    def _validate_shapes(
        self,
        z: torch.Tensor,
        h_local: torch.Tensor,
        h_input: torch.Tensor,
        ball_members: Sequence[torch.Tensor | Sequence[int]],
        ball_centers: torch.Tensor,
        ball_quality: torch.Tensor,
        query_indices: torch.Tensor | None,
    ) -> None:
        if z.ndim != 2:
            raise ValueError("z must have shape [N, feature_dim]")
        if h_local.ndim != 2 or h_local.shape[1] != self.hidden_dim:
            raise ValueError("h_local must have shape [N, hidden_dim]")
        if h_input.ndim != 2 or h_input.shape[1] != self.hidden_dim:
            raise ValueError("h_input must have shape [N, hidden_dim]")
        if z.shape[0] != h_input.shape[0]:
            raise ValueError("z and h_input must contain the same number of nodes")
        if query_indices is None:
            if h_input.shape[0] != h_local.shape[0]:
                raise ValueError(
                    "h_input and h_local must have the same node count when "
                    "query_indices is not provided"
                )
        else:
            if query_indices.ndim != 1 or query_indices.numel() != h_local.shape[0]:
                raise ValueError("query_indices must have shape [h_local.shape[0]]")
            if query_indices.numel() and (
                int(query_indices.min()) < 0 or int(query_indices.max()) >= z.shape[0]
            ):
                raise IndexError("query_indices contains an invalid node index")
        if ball_centers.ndim != 2 or ball_centers.shape[1] != z.shape[1]:
            raise ValueError(
                "ball_centers must have shape [K, z.shape[1]]"
            )
        if len(ball_members) != ball_centers.shape[0]:
            raise ValueError("ball_members must contain K entries")
        if ball_quality.ndim != 1 or ball_quality.shape[0] != ball_centers.shape[0]:
            raise ValueError("ball_quality must have shape [K]")

    @staticmethod
    def _check_finite(**tensors: torch.Tensor) -> None:
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"{name} contains NaN or Inf")


__all__ = ["GranularBallPrototypeEnhancement"]

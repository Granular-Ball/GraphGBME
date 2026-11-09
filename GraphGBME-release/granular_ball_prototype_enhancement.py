"""Match granular-ball prototypes and concatenate them with local GNN features."""

from __future__ import annotations

import torch
from torch import nn


class GranularBallPrototypeEnhancement(nn.Module):
    def __init__(self, hidden_dim: int, top_k: int = 3, eps: float = 1e-8) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.eps = eps

    def forward(
        self,
        *,
        z_query: torch.Tensor,
        h_local: torch.Tensor,
        member_hidden: torch.Tensor,
        member_ball_ids: torch.Tensor,
        member_counts: torch.Tensor,
        ball_centers: torch.Tensor,
        ball_quality: torch.Tensor,
    ) -> torch.Tensor:
        """Use flattened ball membership prepared once by the model."""
        ball_count = member_counts.numel()
        if h_local.ndim != 2 or h_local.shape[1] != self.hidden_dim:
            raise ValueError("h_local must have shape [N, hidden_dim]")
        if z_query.ndim != 2 or z_query.shape[0] != h_local.shape[0]:
            raise ValueError("z_query must have shape [N, feature_dim]")
        if member_hidden.ndim != 2 or member_hidden.shape[1] != self.hidden_dim:
            raise ValueError("member_hidden must have shape [M, hidden_dim]")
        if member_ball_ids.shape != (member_hidden.shape[0],):
            raise ValueError("member_ball_ids must have shape [M]")
        if member_counts.ndim != 1:
            raise ValueError("member_counts must have shape [K]")
        if ball_centers.shape != (ball_count, z_query.shape[1]):
            raise ValueError("ball_centers must have shape [K, feature_dim]")
        if ball_quality.shape != (ball_count,):
            raise ValueError("ball_quality must have shape [K]")
        if ball_count == 0:
            if not torch.isfinite(h_local).all():
                raise FloatingPointError("h_local contains NaN or Inf")
            return torch.cat((h_local, torch.zeros_like(h_local)), dim=-1)

        device, dtype = h_local.device, h_local.dtype
        z_query = z_query.to(device=device, dtype=dtype)
        member_hidden = member_hidden.to(device=device, dtype=dtype)
        member_ball_ids = member_ball_ids.to(device=device, dtype=torch.long)
        member_counts = member_counts.to(device=device, dtype=dtype)
        ball_centers = ball_centers.to(device=device, dtype=dtype)
        ball_quality = ball_quality.to(device=device, dtype=dtype)

        # Aggregate all member embeddings with one differentiable reduction.
        prototypes = member_hidden.new_zeros((ball_count, self.hidden_dim))
        prototypes = prototypes.index_add(0, member_ball_ids, member_hidden)
        prototypes = prototypes / member_counts.unsqueeze(1)

        scores = ball_quality.unsqueeze(0) * torch.exp(-torch.cdist(z_query, ball_centers, p=2))
        selected_scores, selected_indices = torch.topk(
            scores, k=min(self.top_k, ball_count), dim=1, sorted=True
        )
        weights = selected_scores / (selected_scores.sum(dim=1, keepdim=True) + self.eps)
        mesoscopic = (weights.unsqueeze(-1) * prototypes[selected_indices]).sum(dim=1)
        return torch.cat((h_local, mesoscopic), dim=-1)

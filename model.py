from __future__ import annotations

import dgl
import torch
from torch import nn
from dgl.nn import SAGEConv

try:
    from .granular_ball_prototype_enhancement import (
        GranularBallPrototypeEnhancement,
    )
except ImportError:
    from granular_ball_prototype_enhancement import GranularBallPrototypeEnhancement


class BallGraphSAGE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        ball_members: list[torch.Tensor] | None = None,
        ball_centers: torch.Tensor | None = None,
        ball_quality: torch.Tensor | None = None,
        min_ball_size: int = 2,
        quality_threshold: float = 0.0,
        prototype_top_k: int = 3,
        prototype_layer_norm: bool = False,
        allow_singleton_prototypes: bool = False,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim, "mean"), SAGEConv(hidden_dim, 2, "mean")]
        )
        self.dropout = nn.Dropout(dropout)
        self.enhancement = GranularBallPrototypeEnhancement(
            hidden_dim=hidden_dim,
            min_ball_size=min_ball_size,
            quality_threshold=quality_threshold,
            top_k=prototype_top_k,
            use_layer_norm=prototype_layer_norm,
            allow_singletons=allow_singleton_prototypes,
        )

        if ball_members is None:
            ball_members = []
        if ball_centers is None:
            ball_centers = torch.empty((0, in_dim))
        if ball_quality is None:
            ball_quality = torch.empty(0)
        if len(ball_members) != ball_centers.shape[0]:
            raise ValueError("ball_members and ball_centers must contain the same K balls")

        # Reliability depends only on fixed ball metadata, so compute it once on
        # CPU instead of repeating Python loops and GPU scalar synchronizations
        # in every mini-batch.
        quality_cpu = ball_quality.detach().cpu().float().flatten()
        if quality_cpu.shape[0] != len(ball_members):
            raise ValueError("ball_quality must contain one value per ball")
        reliable_original_indices: list[int] = []
        reliable_members: list[torch.Tensor] = []
        minimum_size = (
            min_ball_size if allow_singleton_prototypes else max(2, min_ball_size)
        )
        for index, members in enumerate(ball_members):
            member_ids = torch.as_tensor(members, dtype=torch.long).flatten().cpu()
            if member_ids.numel() and int(member_ids.min()) < 0:
                raise IndexError(f"ball_members[{index}] contains a negative node id")
            if (
                member_ids.numel() >= minimum_size
                and float(quality_cpu[index]) >= quality_threshold
            ):
                reliable_original_indices.append(index)
                reliable_members.append(member_ids)

        reliable_count = len(reliable_members)
        if reliable_members:
            flattened_member_ids = torch.cat(reliable_members)
            member_ball_ids = torch.repeat_interleave(
                torch.arange(reliable_count, dtype=torch.long),
                torch.tensor(
                    [members.numel() for members in reliable_members],
                    dtype=torch.long,
                ),
            )
            unique_member_ids, member_to_unique = torch.unique(
                flattened_member_ids,
                sorted=True,
                return_inverse=True,
            )
            member_counts = torch.bincount(
                member_ball_ids,
                minlength=reliable_count,
            )
        else:
            unique_member_ids = torch.empty(0, dtype=torch.long)
            member_to_unique = torch.empty(0, dtype=torch.long)
            member_ball_ids = torch.empty(0, dtype=torch.long)
            member_counts = torch.empty(0, dtype=torch.long)

        reliable_indices_tensor = torch.tensor(
            reliable_original_indices,
            dtype=torch.long,
        )
        centers_cpu = ball_centers.detach().cpu().float()
        self.register_buffer(
            "reliable_member_ids",
            unique_member_ids,
            persistent=False,
        )
        self.register_buffer(
            "reliable_member_to_unique",
            member_to_unique,
            persistent=False,
        )
        self.register_buffer(
            "reliable_member_ball_ids",
            member_ball_ids,
            persistent=False,
        )
        self.register_buffer(
            "reliable_member_counts",
            member_counts,
            persistent=False,
        )
        self.register_buffer(
            "reliable_ball_centers",
            centers_cpu.index_select(0, reliable_indices_tensor),
            persistent=False,
        )
        self.register_buffer(
            "reliable_ball_quality",
            quality_cpu.index_select(0, reliable_indices_tensor),
            persistent=False,
        )
        self.register_buffer(
            "reliable_original_ball_indices",
            reliable_indices_tensor,
            persistent=False,
        )

    def forward(self, blocks: list[dgl.DGLGraph], all_z: torch.Tensor) -> torch.Tensor:
        source_ids = blocks[0].srcdata[dgl.NID].to(all_z.device)
        query_ids = blocks[0].dstdata[dgl.NID].to(all_z.device)

        # DGL already supplies unique block source IDs, and reliable member IDs
        # were deduplicated during initialization. Concatenating their features
        # avoids an expensive GPU sort/unique in every mini-batch. A node in both
        # groups may be projected twice, but the reliable set is small and this
        # is cheaper than sorting the much larger sampled source set.
        source_count = source_ids.numel()
        projected = torch.relu(
            self.input_projection(
                torch.cat(
                    (
                        all_z.index_select(0, source_ids),
                        all_z.index_select(0, self.reliable_member_ids),
                    )
                )
            )
        )
        source_hidden = projected[:source_count]
        h_local = self.layers[0](blocks[0], source_hidden)

        unique_member_hidden = projected[source_count:]
        member_hidden = unique_member_hidden.index_select(
            0,
            self.reliable_member_to_unique,
        )
        x, _, _, _, _ = self.enhancement.forward_prepared(
            z_query=all_z.index_select(0, query_ids),
            h_local=h_local,
            member_hidden=member_hidden,
            member_ball_ids=self.reliable_member_ball_ids,
            member_counts=self.reliable_member_counts,
            reliable_centers=self.reliable_ball_centers,
            reliable_quality=self.reliable_ball_quality,
            reliable_original_indices=self.reliable_original_ball_indices,
        )
        x = self.dropout(torch.relu(x))
        return self.layers[1](blocks[1], x)

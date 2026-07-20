"""Per-source-column set or sequence aggregation."""

from __future__ import annotations

import torch
from torch import nn

from .cell_encoder import TypedCellTokenizer


class RelationColumnAggregator(nn.Module):
    def __init__(
        self,
        *,
        tokenizer: TypedCellTokenizer,
        token_dim: int,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in {"set", "sequence"}:
            raise ValueError("mode must be set or sequence")
        self.tokenizer = tokenizer
        self.mode = mode
        self.position_projection = nn.Linear(1, token_dim)
        self.task_projection = nn.Linear(token_dim, token_dim)
        self.path_projection = nn.Linear(token_dim, token_dim)
        self.attention = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.Tanh(),
            nn.Linear(token_dim, 1),
        )
        self.sequence = nn.GRU(token_dim, token_dim, batch_first=True)
        self.normalization = nn.LayerNorm(token_dim)

    def forward(
        self,
        *,
        values: torch.Tensor,
        missing: torch.Tensor,
        row_mask: torch.Tensor,
        positions: torch.Tensor,
        type_id: torch.Tensor,
        column_features: torch.Tensor,
        task_context: torch.Tensor,
        path_embedding: torch.Tensor,
    ) -> torch.Tensor:
        type_ids = type_id.reshape(1).expand(values.shape[-1])
        column = column_features.reshape(1, -1).expand(values.shape[-1], -1)
        tokens = self.tokenizer(values, missing, type_ids, column)
        tokens = tokens + self.position_projection(positions.unsqueeze(-1))
        tokens = tokens + self.task_projection(task_context)
        tokens = tokens + self.path_projection(path_embedding)
        if self.mode == "sequence":
            encoded, _state = self.sequence(tokens)
        else:
            encoded = tokens
        scores = self.attention(encoded).squeeze(-1)
        scores = scores.masked_fill(~row_mask, -1e4)
        weights = torch.softmax(scores, dim=-1) * row_mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=-2)
        available = row_mask.any(dim=-1, keepdim=True)
        return self.normalization(pooled) * available.float()

    def forward_batched(
        self,
        *,
        values: torch.Tensor,
        missing: torch.Tensor,
        row_mask: torch.Tensor,
        positions: torch.Tensor,
        type_ids: torch.Tensor,
        column_features: torch.Tensor,
        task_context: torch.Tensor,
        path_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate selected [batch, relation-slot] tensors in parallel."""
        if values.ndim != 4:
            raise ValueError(
                "batched source values must be [batch, rows, slots, samples]"
            )
        batch, rows, slots, samples = values.shape
        expanded_types = type_ids[:, None, :, None].expand(
            batch, rows, slots, samples
        )
        expanded_columns = column_features[:, None, :, None, :].expand(
            batch, rows, slots, samples, column_features.shape[-1]
        )
        tokens = self.tokenizer(
            values,
            missing,
            expanded_types,
            expanded_columns,
        )
        tokens = tokens + self.position_projection(
            positions.unsqueeze(-1)
        )
        tokens = tokens + self.task_projection(task_context)[
            :, None, None, None, :
        ]
        tokens = tokens + self.path_projection(path_embeddings)[
            :, None, :, None, :
        ]
        if self.mode == "sequence":
            dimension = tokens.shape[-1]
            flattened = tokens.reshape(-1, samples, dimension)
            encoded, _state = self.sequence(flattened)
            encoded = encoded.reshape(
                batch, rows, slots, samples, dimension
            )
        else:
            encoded = tokens
        scores = self.attention(encoded).squeeze(-1)
        scores = scores.masked_fill(~row_mask, -1e4)
        weights = torch.softmax(scores, dim=-1) * row_mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=-2)
        available = row_mask.any(dim=-1)
        return self.normalization(pooled) * available.unsqueeze(-1), available


__all__ = ["RelationColumnAggregator"]

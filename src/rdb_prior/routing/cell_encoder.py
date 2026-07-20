"""Typed cell tokens and support-conditioned column-preserving context."""

from __future__ import annotations

import torch
from torch import nn


class TypedCellTokenizer(nn.Module):
    def __init__(self, *, token_dim: int, type_embedding_dim: int) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(6, type_embedding_dim)
        self.value_encoder = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.type_projection = nn.Linear(type_embedding_dim, token_dim)
        self.column_projection = nn.Linear(8, token_dim)
        self.normalization = nn.LayerNorm(token_dim)

    def forward(
        self,
        values: torch.Tensor,
        missing: torch.Tensor,
        type_ids: torch.Tensor,
        column_features: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape != missing.shape:
            raise ValueError("values and missing must have equal shape")
        value_inputs = torch.stack(
            (values.float(), missing.float()), dim=-1
        )
        value_tokens = self.value_encoder(value_inputs)
        type_tokens = self.type_projection(self.type_embedding(type_ids.long()))
        column_tokens = self.column_projection(column_features.float())
        while type_tokens.ndim < value_tokens.ndim:
            type_tokens = type_tokens.unsqueeze(0)
            column_tokens = column_tokens.unsqueeze(0)
        return self.normalization(value_tokens + type_tokens + column_tokens)


class SupportTaskEncoder(nn.Module):
    """Pool across columns only after support labels have been introduced."""

    def __init__(self, token_dim: int) -> None:
        super().__init__()
        self.label_encoder = nn.Sequential(
            nn.Linear(1, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.output = nn.Sequential(
            nn.Linear(2 * token_dim, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )

    def forward(
        self,
        cell_tokens: torch.Tensor,
        labels: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        if cell_tokens.ndim != 3:
            raise ValueError("cell_tokens must have shape [rows, columns, dim]")
        support = cell_tokens[support_mask]
        if not len(support):
            raise ValueError("support set must not be empty")
        label_tokens = self.label_encoder(
            labels[support_mask].float().reshape(-1, 1)
        )
        # Column outputs are retained until labels are present; this pooling is
        # therefore task-conditioned and never sees query labels.
        feature_summary = support.mean(dim=(0, 1))
        label_summary = label_tokens.mean(dim=0)
        return self.output(torch.cat((feature_summary, label_summary), dim=-1))


class ColumnPreservingContextualizer(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.film = nn.Linear(token_dim, 2 * token_dim)
        self.normalization = nn.LayerNorm(token_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        task_context: torch.Tensor,
    ) -> torch.Tensor:
        scale, shift = self.film(task_context).chunk(2, dim=-1)
        conditioned = tokens * (1.0 + scale) + shift
        # The encoder changes context, not shape: every input column retains a
        # corresponding output token.
        return self.normalization(self.encoder(conditioned))


__all__ = [
    "TypedCellTokenizer",
    "SupportTaskEncoder",
    "ColumnPreservingContextualizer",
]

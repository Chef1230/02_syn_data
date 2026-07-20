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
            # Insert broadcast axes before the column/sample axis. This works
            # for both [rows, columns] and [batch, rows, columns] inputs.
            type_tokens = type_tokens.unsqueeze(-3)
            column_tokens = column_tokens.unsqueeze(-3)
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
        *,
        row_mask: torch.Tensor | None = None,
        column_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squeeze = cell_tokens.ndim == 3
        if squeeze:
            cell_tokens = cell_tokens.unsqueeze(0)
            labels = labels.unsqueeze(0)
            support_mask = support_mask.unsqueeze(0)
            row_mask = (
                torch.ones_like(support_mask)
                if row_mask is None
                else row_mask.unsqueeze(0)
            )
            column_mask = (
                torch.ones(
                    1,
                    cell_tokens.shape[2],
                    dtype=torch.bool,
                    device=cell_tokens.device,
                )
                if column_mask is None
                else column_mask.unsqueeze(0)
            )
        elif cell_tokens.ndim == 4:
            if row_mask is None:
                row_mask = torch.ones_like(support_mask)
            if column_mask is None:
                column_mask = torch.ones(
                    cell_tokens.shape[0],
                    cell_tokens.shape[2],
                    dtype=torch.bool,
                    device=cell_tokens.device,
                )
        else:
            raise ValueError(
                "cell_tokens must have shape [rows, columns, dim] or "
                "[batch, rows, columns, dim]"
            )
        valid_support = support_mask.bool() & row_mask.bool()
        if torch.any(valid_support.sum(dim=1) == 0):
            raise ValueError("every task support set must be non-empty")
        cell_mask = valid_support[:, :, None] & column_mask[:, None, :].bool()
        feature_summary = (
            cell_tokens * cell_mask.unsqueeze(-1)
        ).sum(dim=(1, 2)) / cell_mask.sum(dim=(1, 2), keepdim=False).clamp_min(1).unsqueeze(-1)
        label_tokens = self.label_encoder(labels.float().unsqueeze(-1))
        label_summary = (
            label_tokens * valid_support.unsqueeze(-1)
        ).sum(dim=1) / valid_support.sum(dim=1, keepdim=True).clamp_min(1)
        # Column outputs are retained until labels are present; this pooling is
        # therefore task-conditioned and never sees query labels.
        result = self.output(
            torch.cat((feature_summary, label_summary), dim=-1)
        )
        return result.squeeze(0) if squeeze else result


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
        *,
        row_mask: torch.Tensor | None = None,
        column_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squeeze = tokens.ndim == 3
        if squeeze:
            tokens = tokens.unsqueeze(0)
            task_context = task_context.unsqueeze(0)
            row_mask = (
                torch.ones(
                    1, tokens.shape[1], dtype=torch.bool, device=tokens.device
                )
                if row_mask is None
                else row_mask.unsqueeze(0)
            )
            column_mask = (
                torch.ones(
                    1, tokens.shape[2], dtype=torch.bool, device=tokens.device
                )
                if column_mask is None
                else column_mask.unsqueeze(0)
            )
        elif tokens.ndim == 4:
            if row_mask is None:
                row_mask = torch.ones(
                    tokens.shape[:2], dtype=torch.bool, device=tokens.device
                )
            if column_mask is None:
                column_mask = torch.ones(
                    tokens.shape[0],
                    tokens.shape[2],
                    dtype=torch.bool,
                    device=tokens.device,
                )
        else:
            raise ValueError(
                "tokens must have shape [rows, columns, dim] or "
                "[batch, rows, columns, dim]"
            )
        scale, shift = self.film(task_context).chunk(2, dim=-1)
        conditioned = (
            tokens * (1.0 + scale[:, None, None, :])
            + shift[:, None, None, :]
        )
        batch_size, rows, columns, dimension = conditioned.shape
        flattened = conditioned.reshape(batch_size * rows, columns, dimension)
        padding = (~column_mask.bool())[:, None, :].expand(
            batch_size, rows, columns
        ).reshape(batch_size * rows, columns)
        # The encoder changes context, not shape: every input column retains a
        # corresponding output token.
        encoded = self.encoder(flattened, src_key_padding_mask=padding)
        encoded = encoded.reshape(batch_size, rows, columns, dimension)
        valid = row_mask[:, :, None] & column_mask[:, None, :]
        encoded = self.normalization(encoded) * valid.unsqueeze(-1)
        return encoded.squeeze(0) if squeeze else encoded


__all__ = [
    "TypedCellTokenizer",
    "SupportTaskEncoder",
    "ColumnPreservingContextualizer",
]

"""PFN-style support/query transformer consuming relation-column tokens."""

from __future__ import annotations

import torch
from torch import nn


class PFNStyleTransformer(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        heads: int,
        layers: int,
        dropout: float,
        max_classes: int,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.target_token_type = nn.Parameter(torch.zeros(token_dim))
        self.relation_token_type = nn.Parameter(torch.zeros(token_dim))
        feature_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.feature_encoder = nn.TransformerEncoder(
            feature_layer, num_layers=layers
        )
        row_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.row_encoder = nn.TransformerEncoder(row_layer, num_layers=layers)
        self.label_encoder = nn.Sequential(
            nn.Linear(1, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )
        self.split_embedding = nn.Embedding(2, token_dim)
        self.classification_head = nn.Linear(token_dim, max_classes)
        self.regression_head = nn.Linear(token_dim, 1)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        *,
        target_tokens: torch.Tensor,
        relation_tokens: torch.Tensor,
        relation_mask: torch.Tensor,
        labels: torch.Tensor,
        support_mask: torch.Tensor,
        row_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        squeeze = target_tokens.ndim == 3
        if squeeze:
            target_tokens = target_tokens.unsqueeze(0)
            relation_tokens = relation_tokens.unsqueeze(0)
            relation_mask = relation_mask.unsqueeze(0)
            labels = labels.unsqueeze(0)
            support_mask = support_mask.unsqueeze(0)
            row_mask = (
                torch.ones_like(support_mask)
                if row_mask is None
                else row_mask.unsqueeze(0)
            )
            target_mask = (
                torch.ones(
                    1,
                    target_tokens.shape[2],
                    dtype=torch.bool,
                    device=target_tokens.device,
                )
                if target_mask is None
                else target_mask.unsqueeze(0)
            )
        elif target_tokens.ndim == 4:
            if row_mask is None:
                row_mask = torch.ones_like(support_mask)
            if target_mask is None:
                target_mask = torch.ones(
                    target_tokens.shape[0],
                    target_tokens.shape[2],
                    dtype=torch.bool,
                    device=target_tokens.device,
                )
        else:
            raise ValueError(
                "target_tokens must be [rows, columns, dim] or "
                "[batch, rows, columns, dim]"
            )
        batch, rows, target_count, dimension = target_tokens.shape
        target = target_tokens + self.target_token_type
        relation = relation_tokens + self.relation_token_type
        cls = self.cls_token.reshape(1, 1, 1, dimension).expand(
            batch, rows, 1, dimension
        )
        features = torch.cat((cls, target, relation), dim=2)
        cls_mask = torch.zeros(
            (batch, rows, 1),
            dtype=torch.bool,
            device=target.device,
        )
        target_padding = (~target_mask.bool())[:, None, :].expand(
            batch, rows, target_count
        )
        key_padding = torch.cat(
            (cls_mask, target_padding, ~relation_mask.bool()), dim=2
        )
        feature_count = features.shape[2]
        encoded_features = self.feature_encoder(
            features.reshape(batch * rows, feature_count, dimension),
            src_key_padding_mask=key_padding.reshape(batch * rows, feature_count),
        )
        row_tokens = encoded_features[:, 0].reshape(batch, rows, dimension)
        encoded_labels = self.label_encoder(labels.float().unsqueeze(-1))
        visible_labels = support_mask.bool() & row_mask.bool()
        label_tokens = encoded_labels * visible_labels.unsqueeze(-1)
        row_tokens = (
            row_tokens
            + label_tokens
            + self.split_embedding(support_mask.long())
        )
        # Query labels never enter this stream. Rows may interact through X and
        # support labels, matching PFN-style in-context prediction.
        contextual_rows = self.row_encoder(
            row_tokens,
            src_key_padding_mask=~row_mask.bool(),
        )
        contextual_rows = contextual_rows * row_mask.unsqueeze(-1)
        classification = self.classification_head(contextual_rows)
        regression = self.regression_head(contextual_rows).squeeze(-1)
        if squeeze:
            return classification[0], regression[0], contextual_rows[0]
        return classification, regression, contextual_rows


__all__ = ["PFNStyleTransformer"]

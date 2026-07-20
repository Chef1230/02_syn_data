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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = target_tokens.shape[0]
        target = target_tokens + self.target_token_type
        relation = relation_tokens + self.relation_token_type
        cls = self.cls_token.expand(rows, -1, -1)
        features = torch.cat((cls, target, relation), dim=1)
        prefix_mask = torch.zeros(
            (rows, 1 + target.shape[1]),
            dtype=torch.bool,
            device=target.device,
        )
        key_padding = torch.cat((prefix_mask, ~relation_mask), dim=1)
        encoded_features = self.feature_encoder(
            features, src_key_padding_mask=key_padding
        )
        row_tokens = encoded_features[:, 0]
        label_tokens = torch.zeros_like(row_tokens)
        label_tokens[support_mask] = self.label_encoder(
            labels[support_mask].float().reshape(-1, 1)
        )
        row_tokens = row_tokens + label_tokens
        row_tokens = row_tokens + self.split_embedding(support_mask.long())
        # Query labels never enter this stream. Rows may interact through X and
        # support labels, matching PFN-style in-context prediction.
        contextual_rows = self.row_encoder(row_tokens.unsqueeze(0)).squeeze(0)
        return (
            self.classification_head(contextual_rows),
            self.regression_head(contextual_rows).squeeze(-1),
            contextual_rows,
        )


__all__ = ["PFNStyleTransformer"]

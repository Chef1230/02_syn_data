"""Support-conditioned MLP path and source-column selectors."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True, kw_only=True)
class SparseSelection:
    logits: torch.Tensor
    soft_probabilities: torch.Tensor
    gates: torch.Tensor
    hard_mask: torch.Tensor


def straight_through_topk(
    logits: torch.Tensor,
    *,
    k: int,
    valid_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> SparseSelection:
    if logits.ndim < 1:
        raise ValueError("straight_through_topk expects at least one dimension")
    valid = (
        torch.ones_like(logits, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.bool()
    )
    if valid.shape != logits.shape:
        raise ValueError("valid_mask must have the same shape as logits")
    probabilities = torch.sigmoid(logits / temperature) * valid.float()
    hard = torch.zeros_like(valid)
    selected_count = min(k, logits.shape[-1])
    if selected_count:
        selected = torch.topk(
            logits.masked_fill(~valid, torch.finfo(logits.dtype).min),
            k=selected_count,
            dim=-1,
        ).indices
        hard.scatter_(-1, selected, True)
        hard &= valid
    gates = hard.float() + probabilities - probabilities.detach()
    return SparseSelection(
        logits=logits,
        soft_probabilities=probabilities,
        gates=gates,
        hard_mask=hard,
    )


class MLPPathRouter(nn.Module):
    def __init__(
        self,
        *,
        path_feature_dim: int,
        token_dim: int,
        hidden_dim: int,
        top_k: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.path_encoder = nn.Sequential(
            nn.Linear(path_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.scorer = nn.Sequential(
            nn.Linear(2 * token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        path_features: torch.Tensor,
        task_context: torch.Tensor,
        path_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SparseSelection]:
        path_embeddings = self.path_encoder(path_features.float())
        if path_embeddings.ndim == 2:
            context = task_context.expand(len(path_embeddings), -1)
        elif path_embeddings.ndim == 3:
            context = task_context[:, None, :].expand_as(path_embeddings)
        else:
            raise ValueError("path_features must be [paths, dim] or [batch, paths, dim]")
        logits = self.scorer(
            torch.cat((path_embeddings, context), dim=-1)
        ).squeeze(-1)
        selection = straight_through_topk(
            logits,
            k=self.top_k,
            valid_mask=path_mask,
            temperature=self.temperature,
        )
        return path_embeddings, selection


class MLPSourceColumnSelector(nn.Module):
    def __init__(
        self,
        *,
        column_feature_dim: int,
        token_dim: int,
        hidden_dim: int,
        top_c: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.top_c = top_c
        self.temperature = temperature
        self.column_encoder = nn.Sequential(
            nn.Linear(column_feature_dim, token_dim),
            nn.GELU(),
            nn.LayerNorm(token_dim),
        )
        self.scorer = nn.Sequential(
            nn.Linear(3 * token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        column_features: torch.Tensor,
        column_mask: torch.Tensor,
        path_embeddings: torch.Tensor,
        task_context: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        column_embeddings = self.column_encoder(column_features.float())
        if column_embeddings.ndim == 3:
            path_context = path_embeddings[:, None, :].expand_as(column_embeddings)
            task = task_context.reshape(1, 1, -1).expand_as(column_embeddings)
        elif column_embeddings.ndim == 4:
            path_context = path_embeddings[:, :, None, :].expand_as(
                column_embeddings
            )
            task = task_context[:, None, None, :].expand_as(column_embeddings)
        else:
            raise ValueError(
                "column_features must be [paths, columns, dim] or "
                "[batch, paths, columns, dim]"
            )
        logits = self.scorer(
            torch.cat((column_embeddings, path_context, task), dim=-1)
        ).squeeze(-1)
        selection = straight_through_topk(
            logits,
            k=self.top_c,
            valid_mask=column_mask,
            temperature=self.temperature,
        )
        return (
            column_embeddings,
            logits,
            selection.soft_probabilities,
            selection.gates,
            selection.hard_mask,
        )


__all__ = [
    "SparseSelection",
    "straight_through_topk",
    "MLPPathRouter",
    "MLPSourceColumnSelector",
]

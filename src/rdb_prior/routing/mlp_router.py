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
    if logits.ndim != 1:
        raise ValueError("straight_through_topk expects a vector")
    valid = (
        torch.ones_like(logits, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.bool()
    )
    probabilities = torch.sigmoid(logits / temperature) * valid.float()
    hard = torch.zeros_like(valid)
    valid_count = int(valid.sum().item())
    if valid_count:
        selected = torch.topk(
            logits.masked_fill(~valid, torch.finfo(logits.dtype).min),
            k=min(k, valid_count),
        ).indices
        hard[selected] = True
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
    ) -> tuple[torch.Tensor, SparseSelection]:
        path_embeddings = self.path_encoder(path_features.float())
        context = task_context.expand(len(path_embeddings), -1)
        logits = self.scorer(
            torch.cat((path_embeddings, context), dim=-1)
        ).squeeze(-1)
        selection = straight_through_topk(
            logits,
            k=self.top_k,
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
        path_context = path_embeddings[:, None, :].expand_as(column_embeddings)
        task = task_context.reshape(1, 1, -1).expand_as(column_embeddings)
        logits = self.scorer(
            torch.cat((column_embeddings, path_context, task), dim=-1)
        ).squeeze(-1)
        gates = torch.zeros_like(logits)
        probabilities = torch.zeros_like(logits)
        hard_mask = torch.zeros_like(column_mask, dtype=torch.bool)
        for path_index in range(logits.shape[0]):
            selection = straight_through_topk(
                logits[path_index],
                k=self.top_c,
                valid_mask=column_mask[path_index],
                temperature=self.temperature,
            )
            gates[path_index] = selection.gates
            probabilities[path_index] = selection.soft_probabilities
            hard_mask[path_index] = selection.hard_mask
        return column_embeddings, logits, probabilities, gates, hard_mask


__all__ = [
    "SparseSelection",
    "straight_through_topk",
    "MLPPathRouter",
    "MLPSourceColumnSelector",
]

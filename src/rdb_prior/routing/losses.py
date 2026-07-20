"""Five-part objective for supervised sparse routing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from rdb_prior.task.model import PredictionType

from .config import RouterTrainingConfig
from .data import RoutedTaskTensors
from .network import SparseRouterOutput


@dataclass(frozen=True, slots=True, kw_only=True)
class RouterLosses:
    total: torch.Tensor
    query_prediction: torch.Tensor
    route: torch.Tensor
    cost: torch.Tensor
    sparse: torch.Tensor
    diversity: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach().cpu()),
            "query_prediction": float(self.query_prediction.detach().cpu()),
            "route": float(self.route.detach().cpu()),
            "cost": float(self.cost.detach().cpu()),
            "sparse": float(self.sparse.detach().cpu()),
            "diversity": float(self.diversity.detach().cpu()),
        }


def sparse_router_loss(
    output: SparseRouterOutput,
    batch: RoutedTaskTensors,
    config: RouterTrainingConfig,
) -> RouterLosses:
    query = batch.query_mask
    if batch.prediction_type is PredictionType.CLASSIFICATION:
        if batch.num_classes > output.classification_logits.shape[-1]:
            raise ValueError("task has more classes than model.max_classes")
        query_prediction = F.cross_entropy(
            output.classification_logits[query, : batch.num_classes],
            batch.labels[query].long(),
        )
    else:
        query_prediction = F.mse_loss(
            output.regression_prediction[query], batch.labels[query].float()
        )

    route_terms = F.binary_cross_entropy_with_logits(
        output.route_selection.logits,
        batch.route_targets.float(),
        reduction="none",
    )
    route = (route_terms * batch.route_weights).sum() / (
        batch.route_weights.sum().clamp_min(1e-6)
    )

    route_probability = output.route_selection.soft_probabilities
    column_probability = output.column_probabilities
    valid_columns = batch.source_column_mask.float()
    mean_column_fraction = (
        (column_probability * valid_columns).sum(dim=-1)
        / valid_columns.sum(dim=-1).clamp_min(1.0)
    )
    effective_read = route_probability * mean_column_fraction
    scalar_path_cost = batch.path_costs.mean(dim=-1)
    cost = (effective_read * scalar_path_cost).sum() / (
        effective_read.sum().clamp_min(1e-6)
    )

    route_excess = F.relu(
        route_probability.sum() - config.model.top_k_paths
    ) / max(1, config.model.max_candidates)
    column_counts = (column_probability * valid_columns).sum(dim=-1)
    column_excess = F.relu(
        column_counts - config.model.max_source_columns
    ) / valid_columns.sum(dim=-1).clamp_min(1.0)
    sparse = route_excess.square() + (
        column_excess * route_probability
    ).sum() / route_probability.sum().clamp_min(1e-6)

    gates = output.route_selection.gates
    pair_weights = gates[:, None] * gates[None, :]
    off_diagonal = ~torch.eye(
        len(gates), dtype=torch.bool, device=gates.device
    )
    diversity_numerator = (
        pair_weights * batch.path_similarity * off_diagonal.float()
    ).sum()
    diversity_denominator = (
        pair_weights * off_diagonal.float()
    ).sum().clamp_min(1e-6)
    diversity = diversity_numerator / diversity_denominator

    total = (
        query_prediction
        + config.lambda_route * route
        + config.lambda_cost * cost
        + config.lambda_sparse * sparse
        + config.lambda_diversity * diversity
    )
    return RouterLosses(
        total=total,
        query_prediction=query_prediction,
        route=route,
        cost=cost,
        sparse=sparse,
        diversity=diversity,
    )


__all__ = ["RouterLosses", "sparse_router_loss"]

"""End-to-end sparse relational PFN network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .aggregation import RelationColumnAggregator
from .cell_encoder import (
    ColumnPreservingContextualizer,
    SupportTaskEncoder,
    TypedCellTokenizer,
)
from .config import RouterModelConfig
from .data import RoutedTaskTensors
from .mlp_router import (
    MLPPathRouter,
    MLPSourceColumnSelector,
    SparseSelection,
)
from .transformer import PFNStyleTransformer


@dataclass(frozen=True, slots=True, kw_only=True)
class SparseRouterOutput:
    classification_logits: torch.Tensor
    regression_prediction: torch.Tensor
    row_embeddings: torch.Tensor
    route_selection: SparseSelection
    column_logits: torch.Tensor
    column_probabilities: torch.Tensor
    column_gates: torch.Tensor
    column_hard_mask: torch.Tensor
    path_embeddings: torch.Tensor
    relation_token_count: int
    target_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    relation_mask: torch.Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class SparseSelectionState:
    target_tokens: torch.Tensor
    task_context: torch.Tensor
    path_embeddings: torch.Tensor
    route_selection: SparseSelection
    column_logits: torch.Tensor
    column_probabilities: torch.Tensor
    column_gates: torch.Tensor
    column_hard_mask: torch.Tensor


class SparseRelationalPFN(nn.Module):
    def __init__(self, config: RouterModelConfig) -> None:
        super().__init__()
        self.config = config
        self.cell_tokenizer = TypedCellTokenizer(
            token_dim=config.token_dim,
            type_embedding_dim=config.type_embedding_dim,
        )
        self.task_encoder = SupportTaskEncoder(config.token_dim)
        self.target_contextualizer = ColumnPreservingContextualizer(
            token_dim=config.token_dim,
            heads=config.transformer_heads,
            layers=max(1, config.transformer_layers // 2),
            dropout=config.dropout,
        )
        self.path_router = MLPPathRouter(
            path_feature_dim=config.path_feature_dim,
            token_dim=config.token_dim,
            hidden_dim=config.router_hidden_dim,
            top_k=config.top_k_paths,
            temperature=config.temperature,
        )
        self.column_selector = MLPSourceColumnSelector(
            column_feature_dim=config.column_feature_dim,
            token_dim=config.token_dim,
            hidden_dim=config.router_hidden_dim,
            top_c=config.max_source_columns,
            temperature=config.temperature,
        )
        self.relation_aggregator = RelationColumnAggregator(
            tokenizer=self.cell_tokenizer,
            token_dim=config.token_dim,
            mode=config.aggregation,
        )
        self.backend = PFNStyleTransformer(
            token_dim=config.token_dim,
            heads=config.transformer_heads,
            layers=config.transformer_layers,
            dropout=config.dropout,
            max_classes=config.max_classes,
        )

    def select(self, batch: RoutedTaskTensors) -> SparseSelectionState:
        """Select paths/columns without reading any candidate relation cells."""
        target_tokens = self.cell_tokenizer(
            batch.target_values,
            batch.target_missing,
            batch.target_type_ids,
            batch.target_column_features,
        )
        task_context = self.task_encoder(
            target_tokens, batch.labels, batch.support_mask
        )
        target_tokens = self.target_contextualizer(
            target_tokens, task_context
        )
        path_embeddings, route_selection = self.path_router(
            batch.path_features, task_context
        )
        (
            _column_embeddings,
            column_logits,
            column_probabilities,
            column_gates,
            column_hard_mask,
        ) = self.column_selector(
            batch.source_column_features,
            batch.source_column_mask,
            path_embeddings,
            task_context,
        )
        return SparseSelectionState(
            target_tokens=target_tokens,
            task_context=task_context,
            path_embeddings=path_embeddings,
            route_selection=route_selection,
            column_logits=column_logits,
            column_probabilities=column_probabilities,
            column_gates=column_gates,
            column_hard_mask=column_hard_mask,
        )

    def forward(
        self,
        batch: RoutedTaskTensors,
        selection_state: SparseSelectionState | None = None,
    ) -> SparseRouterOutput:
        state = selection_state or self.select(batch)
        target_tokens = state.target_tokens
        task_context = state.task_context
        path_embeddings = state.path_embeddings
        route_selection = state.route_selection
        column_logits = state.column_logits
        column_probabilities = state.column_probabilities
        column_gates = state.column_gates
        column_hard_mask = state.column_hard_mask

        relation_tokens: list[torch.Tensor] = []
        relation_masks: list[torch.Tensor] = []
        selected_paths = torch.nonzero(
            route_selection.hard_mask, as_tuple=False
        ).flatten()
        for path_index_tensor in selected_paths:
            path_index = int(path_index_tensor.item())
            selected_columns = torch.nonzero(
                column_hard_mask[path_index], as_tuple=False
            ).flatten()
            for column_index_tensor in selected_columns:
                column_index = int(column_index_tensor.item())
                row_mask = batch.source_row_mask[
                    :, path_index, column_index, :
                ]
                token = self.relation_aggregator(
                    values=batch.source_values[
                        :, path_index, column_index, :
                    ],
                    missing=batch.source_missing[
                        :, path_index, column_index, :
                    ],
                    row_mask=row_mask,
                    positions=batch.source_positions[:, path_index, :],
                    type_id=batch.source_type_ids[path_index, column_index],
                    column_features=batch.source_column_features[
                        path_index, column_index
                    ],
                    task_context=task_context,
                    path_embedding=path_embeddings[path_index],
                )
                # Straight-through gates keep execution hard-sparse while the
                # query objective can still update both MLP selectors.
                gate = (
                    route_selection.gates[path_index]
                    * column_gates[path_index, column_index]
                )
                relation_tokens.append(token * gate)
                relation_masks.append(row_mask.any(dim=-1))

        rows = batch.target_values.shape[0]
        if relation_tokens:
            relation_tensor = torch.stack(relation_tokens, dim=1)
            relation_mask = torch.stack(relation_masks, dim=1)
        else:
            relation_tensor = target_tokens.new_zeros(
                (rows, 0, self.config.token_dim)
            )
            relation_mask = torch.zeros(
                (rows, 0), dtype=torch.bool, device=target_tokens.device
            )
        classification, regression, row_embeddings = self.backend(
            target_tokens=target_tokens,
            relation_tokens=relation_tensor,
            relation_mask=relation_mask,
            labels=batch.labels,
            support_mask=batch.support_mask,
        )
        return SparseRouterOutput(
            classification_logits=classification,
            regression_prediction=regression,
            row_embeddings=row_embeddings,
            route_selection=route_selection,
            column_logits=column_logits,
            column_probabilities=column_probabilities,
            column_gates=column_gates,
            column_hard_mask=column_hard_mask,
            path_embeddings=path_embeddings,
            relation_token_count=len(relation_tokens),
            target_tokens=target_tokens,
            relation_tokens=relation_tensor,
            relation_mask=relation_mask,
        )


__all__ = [
    "SparseRouterOutput",
    "SparseSelectionState",
    "SparseRelationalPFN",
]

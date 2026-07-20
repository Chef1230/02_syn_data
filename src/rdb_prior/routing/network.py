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
from .data import RoutedTaskBatch, RoutedTaskTensors
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

    def select(
        self, batch: RoutedTaskTensors | RoutedTaskBatch
    ) -> SparseSelectionState:
        """Select paths/columns without reading any candidate relation cells."""
        target_tokens = self.cell_tokenizer(
            batch.target_values,
            batch.target_missing,
            batch.target_type_ids,
            batch.target_column_features,
        )
        task_context = self.task_encoder(
            target_tokens,
            batch.labels,
            batch.support_mask,
            row_mask=getattr(batch, "row_mask", None),
            column_mask=getattr(batch, "target_column_mask", None),
        )
        target_tokens = self.target_contextualizer(
            target_tokens,
            task_context,
            row_mask=getattr(batch, "row_mask", None),
            column_mask=getattr(batch, "target_column_mask", None),
        )
        path_embeddings, route_selection = self.path_router(
            batch.path_features,
            task_context,
            getattr(batch, "path_mask", None),
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
        batch: RoutedTaskTensors | RoutedTaskBatch,
        selection_state: SparseSelectionState | None = None,
    ) -> SparseRouterOutput:
        if isinstance(batch, RoutedTaskBatch):
            return self._forward_batched(batch, selection_state)
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

    def _forward_batched(
        self,
        batch: RoutedTaskBatch,
        selection_state: SparseSelectionState | None,
    ) -> SparseRouterOutput:
        state = selection_state or self.select(batch)
        active = (
            state.route_selection.hard_mask[:, :, None]
            & state.column_hard_mask
            & batch.source_column_mask
        )
        batch_size, rows, paths, columns, samples = batch.source_values.shape
        flattened_active = active.reshape(batch_size, paths * columns)
        flattened_gates = (
            state.route_selection.gates[:, :, None] * state.column_gates
        ).reshape(batch_size, paths * columns)
        max_slots = self.config.top_k_paths * self.config.max_source_columns
        gather_indices = torch.zeros(
            batch_size,
            max_slots,
            dtype=torch.long,
            device=batch.source_values.device,
        )
        slot_mask = torch.zeros(
            batch_size,
            max_slots,
            dtype=torch.bool,
            device=batch.source_values.device,
        )
        for batch_index in range(batch_size):
            indices = torch.nonzero(
                flattened_active[batch_index], as_tuple=False
            ).flatten()[:max_slots]
            count = indices.numel()
            if count:
                gather_indices[batch_index, :count] = indices
                slot_mask[batch_index, :count] = True
        source_indices = gather_indices[:, None, :, None].expand(
            batch_size, rows, max_slots, samples
        )
        selected_values = torch.gather(
            batch.source_values.reshape(
                batch_size, rows, paths * columns, samples
            ),
            2,
            source_indices,
        )
        selected_missing = torch.gather(
            batch.source_missing.reshape(
                batch_size, rows, paths * columns, samples
            ),
            2,
            source_indices,
        )
        selected_row_mask = torch.gather(
            batch.source_row_mask.reshape(
                batch_size, rows, paths * columns, samples
            ),
            2,
            source_indices,
        )
        selected_path_indices = torch.div(
            gather_indices, columns, rounding_mode="floor"
        )
        selected_positions = torch.gather(
            batch.source_positions,
            2,
            selected_path_indices[:, None, :, None].expand(
                batch_size, rows, max_slots, samples
            ),
        )
        selected_type_ids = torch.gather(
            batch.source_type_ids.reshape(batch_size, paths * columns),
            1,
            gather_indices,
        )
        selected_column_features = torch.gather(
            batch.source_column_features.reshape(
                batch_size, paths * columns, -1
            ),
            1,
            gather_indices[:, :, None].expand(
                batch_size,
                max_slots,
                batch.source_column_features.shape[-1],
            ),
        )
        selected_path_embeddings = torch.gather(
            state.path_embeddings,
            1,
            selected_path_indices[:, :, None].expand(
                batch_size, max_slots, state.path_embeddings.shape[-1]
            ),
        )
        relation_tokens, relation_mask = self.relation_aggregator.forward_batched(
            values=selected_values,
            missing=selected_missing,
            row_mask=selected_row_mask,
            positions=selected_positions,
            type_ids=selected_type_ids,
            column_features=selected_column_features,
            task_context=state.task_context,
            path_embeddings=selected_path_embeddings,
        )
        selected_gates = torch.gather(
            flattened_gates, 1, gather_indices
        )
        relation_tokens = relation_tokens * selected_gates[:, None, :, None]
        relation_mask = (
            relation_mask
            & slot_mask[:, None, :]
            & batch.row_mask[:, :, None]
        )
        classification, regression, row_embeddings = self.backend(
            target_tokens=state.target_tokens,
            relation_tokens=relation_tokens,
            relation_mask=relation_mask,
            labels=batch.labels,
            support_mask=batch.support_mask,
            row_mask=batch.row_mask,
            target_mask=batch.target_column_mask,
        )
        return SparseRouterOutput(
            classification_logits=classification,
            regression_prediction=regression,
            row_embeddings=row_embeddings,
            route_selection=state.route_selection,
            column_logits=state.column_logits,
            column_probabilities=state.column_probabilities,
            column_gates=state.column_gates,
            column_hard_mask=state.column_hard_mask,
            path_embeddings=state.path_embeddings,
            relation_token_count=int(slot_mask.sum().detach().cpu()),
            target_tokens=state.target_tokens,
            relation_tokens=relation_tokens,
            relation_mask=relation_mask,
        )


__all__ = [
    "SparseRouterOutput",
    "SparseSelectionState",
    "SparseRelationalPFN",
]

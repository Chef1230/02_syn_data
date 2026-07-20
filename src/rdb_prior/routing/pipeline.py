"""Public orchestration surface for sparse routing."""

from .trainer import RouterTrainingResult, train_sparse_router

__all__ = ["RouterTrainingResult", "train_sparse_router"]

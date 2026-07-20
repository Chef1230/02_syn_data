"""Atomic sparse-router checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .config import RouterModelConfig
from .network import SparseRelationalPFN


def save_router_checkpoint(
    path: Path,
    *,
    model: SparseRelationalPFN,
    epoch: int,
    validation_loss: float,
    metrics: Mapping[str, Any],
    overwrite: bool = True,
) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"router checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "rdb-prior-sparse-router-v1",
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "epoch": epoch,
            "validation_loss": validation_loss,
            "metrics": dict(metrics),
        },
        temporary,
    )
    temporary.replace(path)
    return path


def load_router_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[SparseRelationalPFN, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch.
        payload = torch.load(path, map_location=device)
    if payload.get("format") != "rdb-prior-sparse-router-v1":
        raise ValueError("unsupported sparse router checkpoint")
    model = SparseRelationalPFN(RouterModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model, payload


__all__ = ["save_router_checkpoint", "load_router_checkpoint"]

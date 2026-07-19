"""Atomic storage for RDBPFN-compatible dbinfer_bench datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from .model import RDBPFNDataset


_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class RDBPFNArtifactWriter:
    output_root: Path
    overwrite: bool = False
    compress: bool = True

    def commit(self, dataset: RDBPFNDataset) -> Path:
        if not _ARTIFACT_ID.fullmatch(dataset.dataset_name):
            raise ValueError("dataset_name is not artifact-safe")
        target = self.output_root / dataset.dataset_name
        temporary = self.output_root / f".{dataset.dataset_name}.tmp"
        self.output_root.mkdir(parents=True, exist_ok=True)
        if target.exists() and not self.overwrite:
            raise FileExistsError(
                f"RDBPFN dataset already exists: {target}; use overwrite=True"
            )
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        try:
            data_directory = temporary / "data"
            data_directory.mkdir()
            for table_name, columns in dataset.tables.items():
                self._write_npz(data_directory / f"{table_name}.npz", columns)
            task_directory = temporary / dataset.task_name
            task_directory.mkdir()
            for split_name, columns in dataset.splits.items():
                self._write_npz(task_directory / f"{split_name}.npz", columns)
            (temporary / "metadata.yaml").write_text(
                yaml.safe_dump(
                    dict(dataset.metadata),
                    sort_keys=False,
                    allow_unicode=False,
                ),
                encoding="utf-8",
            )
            if target.exists():
                shutil.rmtree(target)
            temporary.replace(target)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return target

    def write_manifest(
        self,
        *,
        configuration: Mapping[str, Any],
        entries: Iterable[Mapping[str, Any]],
        filename: str = "manifest.json",
    ) -> Path:
        if not _ARTIFACT_ID.fullmatch(filename) or not filename.endswith(".json"):
            raise ValueError("manifest filename is not artifact-safe JSON")
        encoded_entries = [dict(entry) for entry in entries]
        path = self.output_root / filename
        if path.exists() and not self.overwrite:
            raise FileExistsError(
                f"RDBPFN export manifest already exists: {path}; use overwrite=True"
            )
        _write_json(
            path,
            {
                "artifact_type": "rdbpfn_export_manifest",
                "artifact_version": 1,
                "configuration": dict(configuration),
                "dataset_count": len(encoded_entries),
                "entries": encoded_entries,
            },
        )
        return path

    def _write_npz(
        self,
        path: Path,
        columns: Mapping[str, np.ndarray],
    ) -> None:
        with path.open("wb") as handle:
            writer = np.savez_compressed if self.compress else np.savez
            writer(handle, **dict(columns))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = ["RDBPFNArtifactWriter"]

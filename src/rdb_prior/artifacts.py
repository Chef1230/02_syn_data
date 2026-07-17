# src/rdb_prior/artifacts.py
# -*- coding: utf-8 -*-
"""Atomic JSON artifact writing for the schema-only generation stage."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from rdb_prior.compilation.model import PhysicalSchema
from rdb_prior.runtime import RuntimeRecord
from rdb_prior.schema.blueprint import SchemaBlueprint
from rdb_prior.schema.spec import constraint_to_dict
from rdb_prior.schema.validation import ValidationReport


_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def blueprint_to_dict(blueprint: SchemaBlueprint) -> dict[str, Any]:
    return {
        "blueprint_id": blueprint.blueprint_id,
        "nodes": [
            {
                "node_id": node.node_id,
                "role": node.role.value,
                "rank": node.rank,
            }
            for node in blueprint.nodes
        ],
        "edges": [
            {
                "edge_id": edge.edge_id,
                "parent_node_id": edge.parent_node_id,
                "child_node_id": edge.child_node_id,
            }
            for edge in blueprint.edges
        ],
        "constraints": [
            constraint_to_dict(constraint)
            for constraint in blueprint.constraints
        ],
    }


def validation_report_to_dict(
    report: ValidationReport,
) -> dict[str, Any]:
    encoded_issues: list[dict[str, Any]] = []
    for issue in report.issues:
        encoded = {
            "layer": issue.layer.value,
            "level": issue.level.value,
            "code": issue.code,
            "message": issue.message,
            "node_ids": list(issue.node_ids),
            "edge_ids": list(issue.edge_ids),
        }
        if issue.constraint_id is not None:
            encoded["constraint_id"] = issue.constraint_id
        if issue.motif_type is not None:
            encoded["motif_type"] = issue.motif_type
        encoded_issues.append(encoded)

    return {
        "blueprint_id": report.blueprint_id,
        "is_valid": report.is_valid,
        "issues": encoded_issues,
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaArtifactWriter:
    output_root: Path
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path):
            raise TypeError("output_root must be pathlib.Path")
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean")

    @property
    def schema_directory(self) -> Path:
        return self.output_root / "schemas"

    def commit(
        self,
        *,
        sample_id: str,
        runtime: RuntimeRecord,
        blueprint: SchemaBlueprint,
        physical_schema: PhysicalSchema,
        report: ValidationReport,
    ) -> Path:
        if not isinstance(sample_id, str) or not _ARTIFACT_ID.fullmatch(
            sample_id
        ):
            raise ValueError("sample_id is not safe for an artifact filename")
        if not isinstance(runtime, RuntimeRecord):
            raise TypeError("runtime must be RuntimeRecord")
        if not isinstance(blueprint, SchemaBlueprint):
            raise TypeError("blueprint must be SchemaBlueprint")
        if not isinstance(physical_schema, PhysicalSchema):
            raise TypeError("physical_schema must be PhysicalSchema")
        if not isinstance(report, ValidationReport):
            raise TypeError("report must be ValidationReport")
        if not report.is_valid:
            raise ValueError("Cannot commit an invalid schema blueprint")

        output_path = self.schema_directory / f"{sample_id}.json"
        payload = {
            "artifact_type": "physical_schema",
            "sample_id": sample_id,
            "runtime": runtime.to_dict(),
            "blueprint": blueprint_to_dict(blueprint),
            "physical_schema": physical_schema.to_dict(),
            "validation": validation_report_to_dict(report),
        }
        self._write_json(output_path, payload)
        return output_path

    def write_manifest(
        self,
        *,
        configuration: Mapping[str, Any],
        entries: Iterable[Mapping[str, Any]],
    ) -> Path:
        manifest_path = self.output_root / "manifest.json"
        payload = {
            "artifact_type": "physical_schema_manifest",
            "configuration": dict(configuration),
            "entries": [dict(entry) for entry in entries],
        }
        self._write_json(manifest_path, payload)
        return manifest_path

    def _write_json(
        self,
        path: Path,
        payload: Mapping[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not self.overwrite:
            raise FileExistsError(
                f"Artifact already exists: {path}; use overwrite=True"
            )

        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
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
        temporary_path.replace(path)


__all__ = [
    "blueprint_to_dict",
    "validation_report_to_dict",
    "SchemaArtifactWriter",
]

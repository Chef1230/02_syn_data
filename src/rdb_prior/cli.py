# src/rdb_prior/cli.py
# -*- coding: utf-8 -*-
"""Command-line entry points for staged synthetic-data generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rdb_prior.config import (
    SchemaConfigError,
    SchemaConfigOverrides,
    load_schema_pipeline_config,
)
from rdb_prior.pipeline import (
    generate_physical_schemas,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rdb-prior")
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema = subparsers.add_parser(
        "schema",
        help="sample validated SchemaBlueprint and PhysicalSchema artifacts",
    )
    schema.add_argument(
        "--config",
        type=Path,
        default=Path("configs/refactor_v1.yaml"),
        help="YAML/JSON schema-pipeline config",
    )
    schema.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override paths.schema_output_root",
    )
    schema.add_argument(
        "--count",
        type=int,
        default=None,
        help="override generation.num_schemas",
    )
    schema.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override top-level seed",
    )
    schema.add_argument("--start-index", type=int, default=None)
    schema.add_argument("--sample-id-prefix", default=None)
    schema.add_argument("--min-tables", type=int, default=None)
    schema.add_argument("--max-tables", type=int, default=None)
    schema.add_argument("--max-rank", type=int, default=None)
    schema.add_argument("--max-extra-edges", type=int, default=None)
    schema.add_argument(
        "--extra-edge-probability",
        type=float,
        default=None,
    )
    schema.add_argument("--min-feature-columns", type=int, default=None)
    schema.add_argument("--max-feature-columns", type=int, default=None)
    schema.add_argument(
        "--feature-nullable-probability",
        type=float,
        default=None,
    )
    schema.add_argument("--blueprint-id-prefix", default=None)
    schema.add_argument("--schema-id-prefix", default=None)
    schema.add_argument("--progress-every", type=int, default=None)
    schema.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override generation.overwrite",
    )
    schema.add_argument(
        "--validate-config-only",
        action="store_true",
        help="validate and print resolved config without generating files",
    )
    return parser


def _run_schema(args: argparse.Namespace) -> int:
    overrides = SchemaConfigOverrides(
        output_root=args.output_dir,
        num_schemas=args.count,
        base_seed=args.seed,
        start_index=args.start_index,
        sample_id_prefix=args.sample_id_prefix,
        progress_every=args.progress_every,
        overwrite=args.overwrite,
        min_tables=args.min_tables,
        max_tables=args.max_tables,
        max_rank=args.max_rank,
        max_extra_edges=args.max_extra_edges,
        extra_edge_probability=args.extra_edge_probability,
        min_feature_columns=args.min_feature_columns,
        max_feature_columns=args.max_feature_columns,
        feature_nullable_probability=(
            args.feature_nullable_probability
        ),
        blueprint_id_prefix=args.blueprint_id_prefix,
        schema_id_prefix=args.schema_id_prefix,
    )
    config = load_schema_pipeline_config(
        args.config,
        overrides=overrides,
    )

    if args.validate_config_only:
        payload = config.to_dict()
        payload["output_root"] = str(config.output_root)
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    def progress(completed: int, total: int, sample_id: str) -> None:
        print(
            f"[schema] {completed}/{total} completed: {sample_id}",
            flush=True,
        )

    result = generate_physical_schemas(config, progress=progress)
    print(
        json.dumps(
            {
                "generated_count": result.generated_count,
                "output_root": str(result.output_root),
                "manifest": str(result.manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "schema":
        try:
            return _run_schema(args)
        except SchemaConfigError as error:
            parser.error(str(error))
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]

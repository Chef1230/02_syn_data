# src/rdb_prior/cli.py
# -*- coding: utf-8 -*-
"""Command-line entry points for staged synthetic-data generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rdb_prior.config import (
    InstanceConfigOverrides,
    RDBPFNExportConfigOverrides,
    SchemaConfigError,
    SchemaConfigOverrides,
    TaskConfigOverrides,
    load_instance_pipeline_config,
    load_rdbpfn_export_config,
    load_schema_pipeline_config,
    load_task_pipeline_config,
)
from rdb_prior.export.pipeline import export_rdbpfn_tasks
from rdb_prior.observability import (
    ProgressReporter,
    close_logging,
    configure_logging,
)
from rdb_prior.pipeline import (
    generate_database_instances,
    generate_physical_schemas,
    generate_tasks,
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
    schema.add_argument(
        "--schema-dot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="write one Graphviz DOT file per generated schema",
    )
    schema.add_argument(
        "--schema-graph-format",
        choices=("none", "png", "svg", "pdf"),
        default=None,
        help="optionally render DOT files with Graphviz",
    )
    schema.add_argument(
        "--graphviz-command",
        default=None,
        help="Graphviz dot executable name or path",
    )
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
    _add_observability_arguments(schema)

    instance = subparsers.add_parser(
        "instance",
        help="materialize database instances from a physical schema manifest",
    )
    instance.add_argument(
        "--config",
        type=Path,
        default=Path("configs/refactor_v1.yaml"),
    )
    instance.add_argument("--schema-manifest", type=Path, default=None)
    instance.add_argument("--output-dir", type=Path, default=None)
    instance.add_argument("--count", type=int, default=None)
    instance.add_argument("--start-index", type=int, default=None)
    instance.add_argument("--shard-id", type=int, default=None)
    instance.add_argument("--num-shards", type=int, default=None)
    instance.add_argument("--progress-every", type=int, default=None)
    instance.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    instance.add_argument("--validate-config-only", action="store_true")
    _add_observability_arguments(instance)

    task = subparsers.add_parser(
        "task",
        help="derive supervised relational tasks from an instance manifest",
    )
    task.add_argument(
        "--config",
        type=Path,
        default=Path("configs/refactor_v1.yaml"),
    )
    task.add_argument("--instance-manifest", type=Path, default=None)
    task.add_argument("--output-dir", type=Path, default=None)
    task.add_argument(
        "--database-count",
        "--count",
        dest="database_count",
        type=int,
        default=None,
    )
    task.add_argument("--tasks-per-database", type=int, default=None)
    task.add_argument("--start-index", type=int, default=None)
    task.add_argument("--shard-id", type=int, default=None)
    task.add_argument("--num-shards", type=int, default=None)
    task.add_argument("--progress-every", type=int, default=None)
    task.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    task.add_argument("--validate-config-only", action="store_true")
    _add_observability_arguments(task)

    export = subparsers.add_parser(
        "rdbpfn-export",
        help="export task artifacts as RDBPFN dbinfer_bench datasets",
    )
    export.add_argument(
        "--config",
        type=Path,
        default=Path("configs/refactor_v1.yaml"),
    )
    export.add_argument("--task-manifest", type=Path, default=None)
    export.add_argument("--output-dir", type=Path, default=None)
    export.add_argument("--count", dest="task_count", type=int, default=None)
    export.add_argument("--start-index", type=int, default=None)
    export.add_argument("--shard-id", type=int, default=None)
    export.add_argument("--num-shards", type=int, default=None)
    export.add_argument("--validation-fraction", type=float, default=None)
    export.add_argument("--min-validation-rows", type=int, default=None)
    export.add_argument(
        "--compress",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    export.add_argument("--progress-every", type=int, default=None)
    export.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    export.add_argument("--validate-config-only", action="store_true")
    _add_observability_arguments(export)
    return parser


def _add_observability_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="write timestamped logs to this file",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="show a terminal progress bar; default is auto-detect",
    )
    parser.add_argument("--progress-width", type=int, default=28)


def _run_schema(args: argparse.Namespace) -> int:
    logger = configure_logging(
        level=args.log_level,
        log_file=args.log_file,
    ).getChild("schema")
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
        write_schema_dot=args.schema_dot,
        schema_graph_format=args.schema_graph_format,
        graphviz_command=args.graphviz_command,
    )
    config = load_schema_pipeline_config(
        args.config,
        overrides=overrides,
    )

    if args.validate_config_only:
        logger.info("schema configuration validated: %s", args.config)
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
        close_logging()
        return 0

    reporter = ProgressReporter(
        stage="schema",
        total=config.num_schemas,
        logger=logger,
        log_every=config.progress_every,
        enabled=args.progress,
        width=args.progress_width,
    )
    try:
        result = generate_physical_schemas(
            config,
            progress=lambda completed, total, sample_id: reporter.update(
                completed,
                total,
                sample_id,
            ),
        )
    except Exception:
        logger.exception("schema pipeline failed")
        raise
    finally:
        reporter.close()
        close_logging()
    print(
        json.dumps(
            {
                "generated_count": result.generated_count,
                "dot_count": len(result.dot_paths),
                "image_count": len(result.image_paths),
                "output_root": str(result.output_root),
                "manifest": str(result.manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_instance(args: argparse.Namespace) -> int:
    logger = configure_logging(
        level=args.log_level,
        log_file=args.log_file,
    ).getChild("instance")
    config = load_instance_pipeline_config(
        args.config,
        overrides=InstanceConfigOverrides(
            schema_manifest=args.schema_manifest,
            output_root=args.output_dir,
            count=args.count,
            start_index=args.start_index,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            progress_every=args.progress_every,
            overwrite=args.overwrite,
        ),
    )
    if args.validate_config_only:
        logger.info("instance configuration validated: %s", args.config)
        payload = config.to_dict()
        payload["output_root"] = str(config.output_root)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        close_logging()
        return 0

    reporter = ProgressReporter(
        stage="instance",
        logger=logger,
        log_every=config.progress_every,
        enabled=args.progress,
        width=args.progress_width,
    )
    try:
        result = generate_database_instances(
            config,
            progress=lambda completed, total, sample_id: reporter.update(
                completed,
                total,
                sample_id,
            ),
        )
    except Exception:
        logger.exception("instance pipeline failed")
        raise
    finally:
        reporter.close()
        close_logging()
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


def _run_task(args: argparse.Namespace) -> int:
    logger = configure_logging(
        level=args.log_level,
        log_file=args.log_file,
    ).getChild("task")
    config = load_task_pipeline_config(
        args.config,
        overrides=TaskConfigOverrides(
            instance_manifest=args.instance_manifest,
            output_root=args.output_dir,
            database_count=args.database_count,
            tasks_per_database=args.tasks_per_database,
            start_index=args.start_index,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            progress_every=args.progress_every,
            overwrite=args.overwrite,
        ),
    )
    if args.validate_config_only:
        logger.info("task configuration validated: %s", args.config)
        payload = config.to_dict()
        payload["output_root"] = str(config.output_root)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        close_logging()
        return 0

    reporter = ProgressReporter(
        stage="task",
        logger=logger,
        log_every=config.progress_every,
        enabled=args.progress,
        width=args.progress_width,
    )

    def progress(
        completed: int,
        total: int,
        sample_id: str,
        task_count: int,
    ) -> None:
        reporter.update(
            completed,
            total,
            sample_id,
            detail=f"tasks={task_count}",
        )

    try:
        result = generate_tasks(config, progress=progress)
    except Exception:
        logger.exception("task pipeline failed")
        raise
    finally:
        reporter.close()
        close_logging()
    print(
        json.dumps(
            {
                "database_count": result.database_count,
                "task_count": result.task_count,
                "output_root": str(result.output_root),
                "manifest": str(result.manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _run_rdbpfn_export(args: argparse.Namespace) -> int:
    logger = configure_logging(
        level=args.log_level,
        log_file=args.log_file,
    ).getChild("rdbpfn-export")
    config = load_rdbpfn_export_config(
        args.config,
        overrides=RDBPFNExportConfigOverrides(
            task_manifest=args.task_manifest,
            output_root=args.output_dir,
            task_count=args.task_count,
            start_index=args.start_index,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            validation_fraction=args.validation_fraction,
            min_validation_rows=args.min_validation_rows,
            compress=args.compress,
            progress_every=args.progress_every,
            overwrite=args.overwrite,
        ),
    )
    if args.validate_config_only:
        logger.info("RDBPFN export configuration validated: %s", args.config)
        payload = config.to_dict()
        payload["output_root"] = str(config.output_root)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        close_logging()
        return 0

    reporter = ProgressReporter(
        stage="rdbpfn-export",
        logger=logger,
        log_every=config.progress_every,
        enabled=args.progress,
        width=args.progress_width,
    )
    try:
        result = export_rdbpfn_tasks(
            config,
            progress=lambda completed, total, dataset_name: reporter.update(
                completed,
                total,
                dataset_name,
            ),
        )
    except Exception:
        logger.exception("RDBPFN export pipeline failed")
        raise
    finally:
        reporter.close()
        close_logging()
    print(
        json.dumps(
            {
                "dataset_count": result.dataset_count,
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
    if args.command == "instance":
        try:
            return _run_instance(args)
        except SchemaConfigError as error:
            parser.error(str(error))
    if args.command == "task":
        try:
            return _run_task(args)
        except SchemaConfigError as error:
            parser.error(str(error))
    if args.command == "rdbpfn-export":
        try:
            return _run_rdbpfn_export(args)
        except SchemaConfigError as error:
            parser.error(str(error))
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]

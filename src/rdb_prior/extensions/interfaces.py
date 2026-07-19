"""Stable extension boundaries for the staged generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rdb_prior.compilation.model import CompilationResult
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.blueprint import SchemaBlueprint


@runtime_checkable
class DomainPrototypeProvider(Protocol):
    def sample(self, runtime: RuntimeContext) -> object | None: ...


@runtime_checkable
class BlueprintProvider(Protocol):
    def sample(
        self,
        sample_id: str | int,
        runtime: RuntimeContext,
        domain: object | None,
    ) -> SchemaBlueprint: ...


@runtime_checkable
class ProcessGrammarProvider(Protocol):
    def instantiate(
        self,
        domain: object | None,
        blueprint: SchemaBlueprint,
        runtime: RuntimeContext,
    ) -> tuple[object, ...]: ...


@runtime_checkable
class TaskPlanProvider(Protocol):
    def plan(
        self,
        domain: object | None,
        blueprint: SchemaBlueprint,
        processes: tuple[object, ...],
        runtime: RuntimeContext,
    ) -> object | None: ...


@runtime_checkable
class DatabaseDesignSampler(Protocol):
    def sample(
        self,
        blueprint: SchemaBlueprint,
        task_plan: object | None,
        runtime: RuntimeContext,
    ) -> object | None: ...


@runtime_checkable
class SchemaCompilerExtension(Protocol):
    def compile(
        self,
        blueprint: SchemaBlueprint,
        design: object | None,
        sample_id: str | int,
        runtime: RuntimeContext,
    ) -> CompilationResult: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ExtensionBundle:
    domain: DomainPrototypeProvider
    blueprint: BlueprintProvider
    process: ProcessGrammarProvider
    task: TaskPlanProvider
    design: DatabaseDesignSampler
    compiler: SchemaCompilerExtension

    def __post_init__(self) -> None:
        checks = (
            ("domain", self.domain, DomainPrototypeProvider),
            ("blueprint", self.blueprint, BlueprintProvider),
            ("process", self.process, ProcessGrammarProvider),
            ("task", self.task, TaskPlanProvider),
            ("design", self.design, DatabaseDesignSampler),
            ("compiler", self.compiler, SchemaCompilerExtension),
        )
        for name, value, protocol in checks:
            if not isinstance(value, protocol):
                raise TypeError(f"{name} does not implement {protocol.__name__}")


__all__ = [
    "DomainPrototypeProvider",
    "BlueprintProvider",
    "ProcessGrammarProvider",
    "TaskPlanProvider",
    "DatabaseDesignSampler",
    "SchemaCompilerExtension",
    "ExtensionBundle",
]

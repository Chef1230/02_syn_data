"""Role-aware binding of physical schemas into executable InstancePlans."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

from rdb_prior.compilation.model import PhysicalForeignKey, PhysicalSchema
from rdb_prior.instance.plan import (
    FeatureSCMFamily,
    InstancePlan,
    PopulationPlan,
    RelationMechanismPlan,
    RootCauseFamily,
    TableMechanismPlan,
    TemporalFamily,
)
from rdb_prior.instance.scm_prior import (
    sample_scm_meta_parameters,
    sample_table_scm_parameters,
)
from rdb_prior.runtime import RuntimeContext
from rdb_prior.schema.spec import Optionality, TableRole


_DEFAULT_SCM_WEIGHTS = (
    (FeatureSCMFamily.EXOGENOUS, 0.30),
    (FeatureSCMFamily.LINEAR, 0.40),
    (FeatureSCMFamily.CAM, 0.20),
    (FeatureSCMFamily.MLP, 0.10),
)

_DEFAULT_ROOT_CAUSE_WEIGHTS = (
    (RootCauseFamily.STANDARD_NORMAL, 0.30),
    (RootCauseFamily.LINEAR, 0.20),
    (RootCauseFamily.NONLINEAR, 0.20),
    (RootCauseFamily.LOGNORMAL, 0.15),
    (RootCauseFamily.GAUSSIAN_MIXTURE, 0.15),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class InstancePlannerConfig:
    entity_rows_min: int = 128
    entity_rows_max: int = 512
    lookup_rows_min: int = 4
    lookup_rows_max: int = 32
    event_fanout_min: float = 0.75
    event_fanout_max: float = 4.0
    bridge_rows_factor_min: float = 0.5
    bridge_rows_factor_max: float = 2.0
    detail_fanout_min: float = 0.75
    detail_fanout_max: float = 4.0
    entity_child_factor_min: float = 0.5
    entity_child_factor_max: float = 1.5
    max_rows_per_table: int = 2048
    latent_dimension: int = 4
    optional_rate_min: float = 0.05
    optional_rate_max: float = 0.25
    affinity_strength: float = 1.0
    degree_strength: float = 0.8
    feature_missing_rate_min: float = 0.02
    feature_missing_rate_max: float = 0.15
    feature_noise_scale_min: float = 0.0001
    feature_noise_scale_max: float = 0.3
    scm_signal_scale_min: float = 0.01
    scm_signal_scale_max: float = 10.0
    scm_meta_relative_std_min: float = 0.01
    scm_meta_relative_std_max: float = 1.0
    scm_activation_scale_min: float = 0.1
    scm_activation_scale_max: float = 100.0
    scm_output_scale_log_std: float = 1.4
    scm_long_tail_probability: float = 0.5
    scm_long_tail_alpha_min: float = 1.1
    scm_long_tail_alpha_max: float = 2.0
    categorical_cardinality_min: int = 3
    categorical_cardinality_max: int = 12
    time_scale_seconds_min: float = 300.0
    time_scale_seconds_max: float = 86_400.0
    scm_weights: tuple[tuple[FeatureSCMFamily, float], ...] = (
        _DEFAULT_SCM_WEIGHTS
    )
    # Root-cause latent distribution prior — each table draws one family
    # that controls the shape of its exogenous latent variables.
    root_cause_weights: tuple[tuple[RootCauseFamily, float], ...] = (
        _DEFAULT_ROOT_CAUSE_WEIGHTS
    )
    # MLP structural prior ranges — sampled once per database, each MLP
    # table then draws its own depth / hidden factor / dropout realisation.
    mlp_depth_min: int = 1
    mlp_depth_max: int = 4
    mlp_hidden_factor_min: float = 0.5
    mlp_hidden_factor_max: float = 4.0
    mlp_dropout_probability: float = 0.6
    mlp_dropout_rate_min: float = 0.05
    mlp_dropout_rate_max: float = 0.4

    def __post_init__(self) -> None:
        _integer_range(
            "entity rows",
            self.entity_rows_min,
            self.entity_rows_max,
        )
        _integer_range(
            "lookup rows",
            self.lookup_rows_min,
            self.lookup_rows_max,
        )
        for name, low, high in (
            ("event fanout", self.event_fanout_min, self.event_fanout_max),
            (
                "bridge rows factor",
                self.bridge_rows_factor_min,
                self.bridge_rows_factor_max,
            ),
            ("detail fanout", self.detail_fanout_min, self.detail_fanout_max),
            (
                "entity child factor",
                self.entity_child_factor_min,
                self.entity_child_factor_max,
            ),
        ):
            _positive_range(name, low, high)
        for name, value in (
            ("max_rows_per_table", self.max_rows_per_table),
            ("latent_dimension", self.latent_dimension),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_rows_per_table < self.entity_rows_min:
            raise ValueError("max_rows_per_table is below entity_rows_min")
        if not 0 <= self.optional_rate_min <= self.optional_rate_max < 1:
            raise ValueError("optional rates must satisfy 0 <= min <= max < 1")
        if self.affinity_strength <= 0 or self.degree_strength < 0:
            raise ValueError("affinity must be positive and degree non-negative")
        if not (
            0
            <= self.feature_missing_rate_min
            <= self.feature_missing_rate_max
            < 1
        ):
            raise ValueError("feature missing rates must satisfy 0 <= min <= max < 1")
        _positive_range(
            "feature noise scale",
            self.feature_noise_scale_min,
            self.feature_noise_scale_max,
        )
        _positive_range(
            "SCM signal scale",
            self.scm_signal_scale_min,
            self.scm_signal_scale_max,
        )
        _positive_range(
            "SCM meta relative std",
            self.scm_meta_relative_std_min,
            self.scm_meta_relative_std_max,
        )
        _positive_range(
            "SCM activation scale",
            self.scm_activation_scale_min,
            self.scm_activation_scale_max,
        )
        if self.scm_output_scale_log_std <= 0:
            raise ValueError("scm_output_scale_log_std must be positive")
        if not 0 <= self.scm_long_tail_probability <= 1:
            raise ValueError("scm_long_tail_probability must be in [0, 1]")
        _positive_range(
            "SCM long-tail alpha",
            self.scm_long_tail_alpha_min,
            self.scm_long_tail_alpha_max,
        )
        if self.scm_long_tail_alpha_min <= 1:
            raise ValueError("scm_long_tail_alpha_min must be greater than 1")
        _integer_range("MLP depth", self.mlp_depth_min, self.mlp_depth_max)
        if self.mlp_depth_min < 1:
            raise ValueError("mlp_depth_min must be at least 1")
        _positive_range(
            "MLP hidden factor",
            self.mlp_hidden_factor_min,
            self.mlp_hidden_factor_max,
        )
        if not 0 <= self.mlp_dropout_probability <= 1:
            raise ValueError("mlp_dropout_probability must be in [0, 1]")
        if not 0 <= self.mlp_dropout_rate_min <= self.mlp_dropout_rate_max < 1:
            raise ValueError(
                "mlp_dropout_rate must satisfy 0 <= min <= max < 1"
            )
        _integer_range(
            "categorical cardinality",
            self.categorical_cardinality_min,
            self.categorical_cardinality_max,
        )
        _positive_range(
            "time scale seconds",
            self.time_scale_seconds_min,
            self.time_scale_seconds_max,
        )
        if not isinstance(self.scm_weights, tuple) or not self.scm_weights:
            raise ValueError("scm_weights must be a non-empty tuple")
        families = tuple(family for family, _weight in self.scm_weights)
        if len(set(families)) != len(families):
            raise ValueError("scm_weights families must be unique")
        for family, weight in self.scm_weights:
            if family not in {
                FeatureSCMFamily.EXOGENOUS,
                FeatureSCMFamily.LINEAR,
                FeatureSCMFamily.CAM,
                FeatureSCMFamily.MLP,
            }:
                raise ValueError(
                    "scm_weights supports exogenous, linear, cam and mlp"
                )
            if weight <= 0:
                raise ValueError("scm weights must be positive")
        if not isinstance(self.root_cause_weights, tuple) or not self.root_cause_weights:
            raise ValueError("root_cause_weights must be a non-empty tuple")
        rc_families = tuple(family for family, _w in self.root_cause_weights)
        if len(set(rc_families)) != len(rc_families):
            raise ValueError("root_cause_weights families must be unique")
        for family, weight in self.root_cause_weights:
            if not isinstance(family, RootCauseFamily):
                raise TypeError("root_cause_weights keys must be RootCauseFamily")
            if weight <= 0:
                raise ValueError("root_cause weights must be positive")


class InstancePlanner:
    def __init__(self, config: InstancePlannerConfig | None = None) -> None:
        self.config = config or InstancePlannerConfig()

    def plan(
        self,
        *,
        sample_id: str,
        schema: PhysicalSchema,
        runtime: RuntimeContext,
    ) -> InstancePlan:
        if not isinstance(schema, PhysicalSchema):
            raise TypeError("schema must be PhysicalSchema")
        if not isinstance(runtime, RuntimeContext):
            raise TypeError("runtime must be RuntimeContext")

        order = tuple(
            table.table_id
            for table in sorted(
                schema.tables,
                key=lambda table: (table.rank, table.table_id),
            )
        )
        incoming = {
            table.table_id: tuple(
                fk
                for fk in schema.foreign_keys
                if fk.child_table_id == table.table_id
            )
            for table in schema.tables
        }
        meta = sample_scm_meta_parameters(
            runtime.numpy_rng("instance", "scm-meta"),
            signal_mean_min=self.config.scm_signal_scale_min,
            signal_mean_max=self.config.scm_signal_scale_max,
            noise_mean_min=self.config.feature_noise_scale_min,
            noise_mean_max=self.config.feature_noise_scale_max,
            relative_std_min=self.config.scm_meta_relative_std_min,
            relative_std_max=self.config.scm_meta_relative_std_max,
            activation_scale_min=self.config.scm_activation_scale_min,
            activation_scale_max=self.config.scm_activation_scale_max,
            output_log_std=self.config.scm_output_scale_log_std,
            long_tail_probability=self.config.scm_long_tail_probability,
            long_tail_alpha_min=self.config.scm_long_tail_alpha_min,
            long_tail_alpha_max=self.config.scm_long_tail_alpha_max,
            mlp_depth_min=self.config.mlp_depth_min,
            mlp_depth_max=self.config.mlp_depth_max,
            mlp_hidden_factor_min=self.config.mlp_hidden_factor_min,
            mlp_hidden_factor_max=self.config.mlp_hidden_factor_max,
            mlp_dropout_probability=self.config.mlp_dropout_probability,
            mlp_dropout_rate_min=self.config.mlp_dropout_rate_min,
            mlp_dropout_rate_max=self.config.mlp_dropout_rate_max,
        )
        table_plans: dict[str, TableMechanismPlan] = {}
        for table_id in order:
            table = schema.table(table_id)
            row_count, strategy, multiplier = self._row_count(
                table.role,
                incoming[table_id],
                table_plans,
                runtime,
                table_id,
            )
            temporal = self._temporal_family(
                table.role,
                incoming[table_id],
                schema,
            )
            family = self._feature_family(table.role, runtime, table_id)
            root_cause = self._root_cause_family(table.role, runtime, table_id)
            parameter_rng = runtime.numpy_rng(
                "instance", "table-parameters", table_id
            )
            scm_parameters = sample_table_scm_parameters(
                parameter_rng,
                meta,
                activation_scale_min=self.config.scm_activation_scale_min,
                activation_scale_max=self.config.scm_activation_scale_max,
            )
            table_plans[table_id] = TableMechanismPlan(
                table_id=table_id,
                role=table.role,
                population=PopulationPlan(
                    strategy=strategy,
                    row_count=row_count,
                    parameters=(("multiplier", multiplier),),
                ),
                latent_dimension=self.config.latent_dimension,
                feature_family=family,
                root_cause_family=root_cause,
                temporal_family=temporal,
                latent_seed=runtime.seed("instance", "latent", table_id),
                feature_seed=runtime.seed("instance", "feature", table_id),
                temporal_seed=runtime.seed("instance", "time", table_id),
                parameters=(
                    (
                        "categorical_cardinality",
                        float(
                            parameter_rng.integers(
                                self.config.categorical_cardinality_min,
                                self.config.categorical_cardinality_max + 1,
                            )
                        ),
                    ),
                    (
                        "missing_rate",
                        float(
                            parameter_rng.uniform(
                                self.config.feature_missing_rate_min,
                                self.config.feature_missing_rate_max,
                            )
                        ),
                    ),
                    (
                        "time_scale_seconds",
                        float(
                            parameter_rng.uniform(
                                self.config.time_scale_seconds_min,
                                self.config.time_scale_seconds_max,
                            )
                        ),
                    ),
                )
                + scm_parameters,
            )

        relations = self._relation_plans(schema, runtime)
        return InstancePlan(
            plan_id=f"instance_plan_{sample_id}",
            sample_id=sample_id,
            schema_id=schema.schema_id,
            blueprint_id=schema.blueprint_id,
            global_seed=runtime.seed("instance", "global-latent"),
            generation_order=order,
            tables=tuple(table_plans[table_id] for table_id in order),
            relations=relations,
            parameters=meta.parameters,
        )

    def _row_count(
        self,
        role: TableRole,
        incoming: tuple[PhysicalForeignKey, ...],
        existing: dict[str, TableMechanismPlan],
        runtime: RuntimeContext,
        table_id: str,
    ) -> tuple[int, str, float]:
        rng = runtime.numpy_rng("instance", "population", table_id)
        parent_counts = [
            existing[fk.parent_table_id].population.row_count
            for fk in incoming
            if fk.parent_table_id in existing
            and fk.relation_strategy != "lookup_assignment"
        ]
        if role is TableRole.LOOKUP:
            count = int(
                rng.integers(
                    self.config.lookup_rows_min,
                    self.config.lookup_rows_max + 1,
                )
            )
            return count, "fixed_exogenous", 1.0
        if not parent_counts:
            if role is not TableRole.ENTITY:
                raise ValueError(f"root table {table_id} must be Entity/Lookup")
            value = exp(
                rng.uniform(
                    log(self.config.entity_rows_min),
                    log(self.config.entity_rows_max + 1),
                )
            )
            return int(value), "root_entity", 1.0

        if role is TableRole.EVENT:
            low, high = self.config.event_fanout_min, self.config.event_fanout_max
            strategy = "parent_conditioned_event"
            basis = max(parent_counts)
        elif role is TableRole.DETAIL:
            low, high = self.config.detail_fanout_min, self.config.detail_fanout_max
            strategy = "parent_conditioned_detail"
            basis = max(parent_counts)
        elif role is TableRole.BRIDGE:
            low = self.config.bridge_rows_factor_min
            high = self.config.bridge_rows_factor_max
            strategy = "joint_bridge_population"
            basis = int(sqrt(parent_counts[0] * parent_counts[-1]))
        else:
            low = self.config.entity_child_factor_min
            high = self.config.entity_child_factor_max
            strategy = "entity_hierarchy_population"
            basis = max(parent_counts)

        multiplier = float(rng.uniform(low, high))
        count = max(1, min(self.config.max_rows_per_table, round(basis * multiplier)))
        return count, strategy, multiplier

    def _feature_family(
        self,
        role: TableRole,
        runtime: RuntimeContext,
        table_id: str,
    ) -> FeatureSCMFamily:
        if role is TableRole.LOOKUP:
            return FeatureSCMFamily.EXOGENOUS
        rng = runtime.python_rng("instance", "scm-family", table_id)
        families, weights = zip(*self.config.scm_weights)
        return rng.choices(families, weights=weights, k=1)[0]

    def _root_cause_family(
        self,
        role: TableRole,
        runtime: RuntimeContext,
        table_id: str,
    ) -> RootCauseFamily:
        """Sample the exogenous latent distribution family for one table."""
        if role is TableRole.LOOKUP:
            return RootCauseFamily.STANDARD_NORMAL
        rng = runtime.python_rng("instance", "root-cause", table_id)
        families, weights = zip(*self.config.root_cause_weights)
        return rng.choices(families, weights=weights, k=1)[0]

    @staticmethod
    def _temporal_family(
        role: TableRole,
        incoming: tuple[PhysicalForeignKey, ...],
        schema: PhysicalSchema,
    ) -> TemporalFamily:
        if role is not TableRole.EVENT:
            return TemporalFamily.NONE
        if any(
            schema.table(fk.parent_table_id).role is TableRole.EVENT
            for fk in incoming
        ):
            return TemporalFamily.TIME_LAGGED
        return TemporalFamily.PARENT_BURST

    def _relation_plans(
        self,
        schema: PhysicalSchema,
        runtime: RuntimeContext,
    ) -> tuple[RelationMechanismPlan, ...]:
        results: list[RelationMechanismPlan] = []
        for table in sorted(schema.tables, key=lambda item: item.table_id):
            incoming = tuple(
                fk
                for fk in schema.foreign_keys
                if fk.child_table_id == table.table_id
            )
            structural = tuple(
                fk for fk in incoming if fk.relation_strategy != "lookup_assignment"
            )
            auxiliary = tuple(
                fk for fk in incoming if fk.relation_strategy == "lookup_assignment"
            )
            if table.role is TableRole.BRIDGE and len(structural) >= 2:
                results.append(
                    self._relation_group(
                        group_id=f"RG_{table.table_id}_bridge",
                        fks=structural,
                        family="affinity_bridge",
                        runtime=runtime,
                        allow_lookup_transition=False,
                    )
                )
            else:
                auxiliary = incoming

            for fk in auxiliary:
                results.append(
                    self._relation_group(
                        group_id=f"RG_{fk.foreign_key_id}",
                        fks=(fk,),
                        family=fk.relation_strategy,
                        runtime=runtime,
                        allow_lookup_transition=(table.role is TableRole.EVENT),
                    )
                )
        return tuple(results)

    def _relation_group(
        self,
        *,
        group_id: str,
        fks: tuple[PhysicalForeignKey, ...],
        family: str,
        runtime: RuntimeContext,
        allow_lookup_transition: bool,
    ) -> RelationMechanismPlan:
        rng = runtime.numpy_rng("instance", "relation-plan", group_id)
        if family == "lookup_assignment":
            choices = ["lookup_cpt", "lookup_softmax"]
            if allow_lookup_transition:
                choices.append("lookup_transition")
            family = choices[int(rng.integers(0, len(choices)))]
        optional_rates = tuple(
            0.0
            if fk.optionality is Optionality.REQUIRED
            else float(
                rng.uniform(
                    self.config.optional_rate_min,
                    self.config.optional_rate_max,
                )
            )
            for fk in fks
        )
        return RelationMechanismPlan(
            relation_group_id=group_id,
            foreign_key_ids=tuple(fk.foreign_key_id for fk in fks),
            parent_table_ids=tuple(fk.parent_table_id for fk in fks),
            child_table_id=fks[0].child_table_id,
            family=family,
            optional_rates=optional_rates,
            seed=runtime.seed("instance", "relation", group_id),
            parameters=(
                ("affinity_strength", self.config.affinity_strength),
                ("degree_strength", self.config.degree_strength),
            ),
        )


def _integer_range(name: str, low: int, high: int) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (low, high)):
        raise TypeError(f"{name} bounds must be integers")
    if low < 1 or high < low:
        raise ValueError(f"invalid {name} bounds")


def _positive_range(name: str, low: float, high: float) -> None:
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (low, high)):
        raise TypeError(f"{name} bounds must be numeric")
    if low <= 0 or high < low:
        raise ValueError(f"invalid {name} bounds")


__all__ = ["InstancePlannerConfig", "InstancePlanner"]

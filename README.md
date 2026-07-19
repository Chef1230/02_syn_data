# rdb-prior staged generation pipeline

Stage 01 samples a validated anonymous `SchemaBlueprint`, preserves private
motif provenance, and compiles a randomized `PhysicalSchema`. Stage 02 binds
that schema to a reproducible `InstancePlan` and materializes validated table
rows, keys, features, and event times. Stage 03 derives supervised tasks from
the frozen database without changing its schema or rows.

## Run

The schema command reads `configs/refactor_v1.yaml`. The reference config is
derived from the schema-related priors in `syn_data/configs/default.yaml` and
contains only options implemented by the current schema stage.

```bash
bash scripts/v1/01_schema.sh
```

Validate and print the fully resolved configuration without writing files:

```bash
VALIDATE_CONFIG_ONLY=1 bash scripts/v1/01_schema.sh
```

Generate a small isolated test batch:

```bash
NUM_SCHEMAS=5 \
OUTPUT_DIR=outputs/schema_smoke \
OVERWRITE=1 \
bash scripts/v1/01_schema.sh
```

Run the intended 20k schema feasibility batch:

```bash
NUM_SCHEMAS=20000 \
BASE_SEED=42 \
OUTPUT_DIR=outputs/schema_v1_20k \
OVERWRITE=0 \
bash scripts/v1/01_schema.sh
```

`01_schema.sh` accepts these run-level environment overrides:

- `CONFIG_PATH`
- `RDB_PRIOR_CONFIG` (legacy alias, lower precedence than `CONFIG_PATH`)
- `OUTPUT_DIR`
- `NUM_SCHEMAS`
- `BASE_SEED`
- `START_INDEX`
- `SAMPLE_ID_PREFIX`
- `PROGRESS_EVERY`
- `SCHEMA_DOT`
- `SCHEMA_GRAPH_FORMAT` (`none`, `png`, `svg`, or `pdf`)
- `GRAPHVIZ_COMMAND`
- `OVERWRITE`
- `VALIDATE_CONFIG_ONLY`
- `PYTHON_BIN`

Unset variables preserve the YAML value. A non-option first argument is also
treated as the config path. Additional CLI arguments may be appended to the
script and take precedence over environment-derived options.

The Python entry point can also be called directly from an installed package:

```bash
rdb-prior schema --config configs/refactor_v1.yaml --count 5
```

Stage 01 writes DOT without an extra Python dependency. To also render one PNG
per schema, install Graphviz so `dot` is on `PATH`, then run:

```bash
SCHEMA_GRAPH_FORMAT=png bash scripts/v1/01_schema.sh
```

An existing DOT artifact can also be rendered manually:

```bash
dot -Tpng outputs/schema_v1_sample/schema_graphs/sample_000000.dot \
  -o outputs/schema_v1_sample/schema_graphs/sample_000000.png
```

Generate instances for the schema manifest configured in the YAML:

```bash
bash scripts/v1/02_instance.sh
```

The most common stage-02 overrides are:

```bash
SCHEMA_MANIFEST=outputs/schema_v1_20k/manifest.json \
INSTANCE_OUTPUT_DIR=outputs/instance_v1_20k \
NUM_INSTANCES=20000 \
OVERWRITE=0 \
bash scripts/v1/02_instance.sh
```

`START_INDEX`, `SHARD_ID`, `NUM_SHARDS`, `PROGRESS_EVERY`, `CONFIG_PATH`,
`PYTHON_BIN`, and `VALIDATE_CONFIG_ONLY` are also supported. The direct entry
point is `rdb-prior instance`.

Generate tasks for the configured instance manifest:

```bash
bash scripts/v1/03_task.sh
```

Control the exact requested task count per selected database:

```bash
INSTANCE_MANIFEST=outputs/instance_v1_20k/manifest.json \
TASK_OUTPUT_DIR=outputs/task_v1_20k \
NUM_DATABASES=20000 \
TASKS_PER_DATABASE=2 \
OVERWRITE=0 \
bash scripts/v1/03_task.sh
```

`03_task.sh` also supports `START_INDEX`, `SHARD_ID`, `NUM_SHARDS`,
`PROGRESS_EVERY`, `CONFIG_PATH`, `PYTHON_BIN`, and
`VALIDATE_CONFIG_ONLY`. Run all three configured stages with:

```bash
bash scripts/v1/generate_v1.sh
```

## Logging and progress

All three commands write timestamped logs to stderr and keep the final JSON
summary on stdout. Interactive terminals show a progress bar automatically;
redirected jobs emit one progress log every `progress_every` completed items.

```bash
rdb-prior task \
  --config configs/refactor_v1.yaml \
  --log-level DEBUG \
  --log-file logs/task_v1.log \
  --progress
```

The shell scripts expose the same controls:

```bash
LOG_LEVEL=INFO \
LOG_FILE=logs/generate_v1.log \
PROGRESS_BAR=1 \
PROGRESS_WIDTH=32 \
bash scripts/v1/generate_v1.sh
```

Use `PROGRESS_BAR=0` or `--no-progress` for batch logs. Valid log levels are
`DEBUG`, `INFO`, `WARNING`, and `ERROR`. DEBUG records each committed artifact;
INFO records stage start, periodic progress, and completion.

## Configuration

The YAML loader rejects unknown fields, wrong scalar types, unsupported roles,
unknown motif names, overlapping feature-column ranges and inconsistent
value/weight lists.

- `seed`: deterministic root seed.
- `paths`: schema artifact output root.
- `generation`: batch size, index range, artifact ID prefix, progress,
  overwrite behavior and project version.
- `schema_graph`: DOT creation, optional Graphviz rendering format and whether
  diagnostic column/role metadata is included in the graph.
- `schema`: table-count range/distribution, rank bound, extra-edge sampling and
  Blueprint ID prefix. Motif occurrence bounds and background attachment
  probability prevent every table from being forced by a structural motif.
- `motifs.weights`: enabled structural motifs and positive sampling weights.
- `physical_design`: PhysicalSchema ID prefix, feature-column defaults,
  table-count rules, optional role overrides, nullability probability and
  primary-key name candidates.
- `instance`: role-conditioned population bounds, shared latent dimension, FK
  affinity, optionality rates, SCM mixture, missingness, noise, categorical
  cardinality, and event-time scales.
- `instance_generation`: schema selection, deterministic sharding, progress,
  overwrite behavior and project version.
- `task`: tasks per database, mechanism weights, support/query sizing,
  classification quality thresholds, and future cutoff/horizon ranges.
- `task_generation`: instance selection, deterministic sharding, progress,
  overwrite behavior and project version.

Feature-column precedence is `role override -> table-count rule -> default`.
The only valid role override keys are `entity`, `event`, `lookup`, `bridge`
and `detail`.

Connectivity, FK DAG, self-loop prohibition, parallel-FK prohibition, rank
ordering, role-edge legality, Entity/Event presence and Bridge parent counts
are hard validity contracts. They are deliberately not configurable switches.
Task/Process/Temporal relations are also not schema motif configuration; they
belong to later Task/Process stages.

## Output

```text
OUTPUT_DIR/
  manifest.json
  schemas/
    sample_000000.json
    sample_000001.json
    ...
  schema_graphs/
    sample_000000.dot
    sample_000001.dot
    ...
```

Each V2 schema artifact contains the runtime reproduction record, completed
logical blueprint, anonymous motif occurrences and bindings, physical schema,
logical-to-physical compilation trace, and validation report. The artifact can
be loaded with `rdb_prior.artifacts.load_schema_artifact` by a later stage.

Motif provenance and role metadata are generator-private. Physical table and
feature names remain anonymous; exporters must not expose generator metadata as
model features.
The manifest records graph paths under each entry's `graph_artifacts` field.
Rendered `png`, `svg`, or `pdf` files appear beside their DOT source when
`schema_graph.render_format` is enabled.

Stage 01 stops at `CompilationResult(PhysicalSchema, CompilationTrace)`.
The stage-02 output is:

```text
INSTANCE_OUTPUT_DIR/
  manifest.json
  instances/sample_000000/
    artifact.json
    instance_plan.json
    runtime.json
    validation.json
    tables/*.npz
```

Entity tables use Linear, CAM, or MLP SCMs. Lookup tables are exogenous and
their incoming selections use CPT, latent softmax, or state-transition
mechanisms. Event tables use parent-burst or time-lagged temporal mechanisms.
Bridge parents are sampled
jointly with affinity, while Detail tables use parent-conditioned populations
and the same Linear/CAM/MLP family. One shared latent registry drives FK choice
and feature contexts; FK rows are therefore not sampled uniformly.

The instance artifact can be loaded with
`rdb_prior.artifacts.load_instance_artifact`.

The stage-03 output is:

```text
TASK_OUTPUT_DIR/
  manifest.json
  tasks/sample_000000/task_sample_000000_000/
    artifact.json
    task_plan.json
    task_data.npz
    runtime.json
    validation.json
```

V1 implements `relation_attribute` and `future_event_existence`. Relation
attribute tasks mask the physical target column and split Event rows in time
order. Future-event tasks label Entity rows from one Entity -> Event FK in a
sampled future window; every Event table is hidden after the common cutoff.
Rows downstream of hidden events are also hidden through the FK closure; use
`rdb_prior.task.view.build_task_view` to apply the canonical view. Invalid,
degenerate, leaking, or undersized tasks are rejected. By default the
pipeline requires exactly `tasks_per_database` valid tasks and fails instead
of silently reducing the count. Load artifacts with
`rdb_prior.task.artifacts.load_task_artifact`.

Business-process grammar, domain prototypes, role-aware compiler replacement,
database-design randomization, additional Task DSL mechanisms, and legacy
export remain later stages.

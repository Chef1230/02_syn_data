# rdb-prior schema pipeline

The current runnable stage samples a validated anonymous `SchemaBlueprint`,
compiles randomized physical tables/columns/foreign keys, and writes one JSON
artifact per schema plus a batch manifest.

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

## Configuration

The YAML loader rejects unknown fields, wrong scalar types, unsupported roles,
unknown motif names, overlapping feature-column ranges and inconsistent
value/weight lists.

- `seed`: deterministic root seed.
- `paths`: schema artifact output root.
- `generation`: batch size, index range, artifact ID prefix, progress,
  overwrite behavior and project version.
- `schema`: table-count range/distribution, rank bound, extra-edge sampling and
  Blueprint ID prefix.
- `motifs.weights`: enabled structural motifs and positive sampling weights.
- `physical_design`: PhysicalSchema ID prefix, feature-column defaults,
  table-count rules, optional role overrides, nullability probability and
  primary-key name candidates.

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
```

Each schema artifact contains the runtime reproduction record, completed
logical blueprint, physical schema, and validation report. Motif occurrences
and slot bindings are construction-time values and are not persisted.

This stage stops at `PhysicalSchema`. Row-count planning, concrete database
instances, Task/Process DSL, database-level validation and legacy export are
separate later pipeline stages.

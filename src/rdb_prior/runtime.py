# src/rdb_prior/runtime.py
# -*- coding: utf-8 -*-
"""
Deterministic runtime and persistable runtime records.

Random streams are derived from stable semantic identifiers:

    base_seed
      -> sample_id
          -> stage
              -> object_id
                  -> component

The derived streams are independent of:

- worker count;
- process scheduling;
- sample execution order;
- table traversal order;
- foreign-key traversal order.

RuntimeRecord stores the deterministic runtime identity needed to reproduce a
generated database instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import blake2b
import json
from pathlib import Path
import platform
import random
from typing import Any, Final, Mapping, TypeAlias

import numpy as np


_HASH_PERSON: Final[bytes] = b"rdbprior-seed-v1"
_DEFAULT_SEED_VERSION: Final[str] = "v1"
_DEFAULT_BIT_GENERATOR: Final[str] = "PCG64DXSM"
_RUNTIME_RECORD_VERSION: Final[str] = "v1"

# Seed scopes are part of the reproducibility contract, so they deliberately
# accept only stable logical identifiers.  Callers must normalize enums to
# their explicit value and must not use filesystem paths or runtime objects.
SeedPart: TypeAlias = str | int
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = (
    JsonScalar
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)


# ---------------------------------------------------------------------------
# Stable seed encoding
# ---------------------------------------------------------------------------


def _frame(payload: bytes) -> bytes:
    """Prefix bytes with their length to avoid concatenation ambiguity."""
    return len(payload).to_bytes(
        8,
        byteorder="big",
        signed=False,
    ) + payload


def _encode_seed_part(value: SeedPart) -> bytes:
    """
    Convert a supported seed component to a stable byte representation.

    Never use Python's built-in ``hash`` for seed derivation because it is
    randomized across interpreter processes.
    """
    if isinstance(value, Enum):
        raise TypeError(
            "Enum values are not valid seed-scope identifiers; "
            "pass enum.value explicitly"
        )

    if isinstance(value, bool):
        raise TypeError(
            "Boolean values are not valid seed-scope identifiers"
        )

    if isinstance(value, int):
        return b"I" + _frame(str(value).encode("ascii"))

    if isinstance(value, str):
        if not value.strip():
            raise ValueError(
                "String seed-scope identifiers must not be empty"
            )
        return b"S" + _frame(value.encode("utf-8"))

    raise TypeError(
        "Seed scopes accept only logical identifiers of type str or int; "
        f"got {type(value).__name__!r}: {value!r}"
    )


def derive_seed(
    base_seed: int,
    *scope: SeedPart,
    version: str = _DEFAULT_SEED_VERSION,
) -> int:
    """
    Derive a stable unsigned 64-bit seed.

    The result depends only on:

    - base_seed;
    - seed derivation version;
    - ordered semantic scope.
    """
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")

    if not isinstance(version, str):
        raise TypeError("version must be a string")

    if not version.strip():
        raise ValueError("version must not be empty")

    digest = blake2b(
        digest_size=8,
        person=_HASH_PERSON,
    )

    parts: tuple[SeedPart, ...] = (
        "seed-version",
        version,
        "base-seed",
        base_seed,
        *scope,
    )

    for part in parts:
        digest.update(_frame(_encode_seed_part(part)))

    return int.from_bytes(
        digest.digest(),
        byteorder="big",
        signed=False,
    )


# ---------------------------------------------------------------------------
# Persistable runtime record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeRecord:
    """
    Persistable identity of one deterministic runtime scope.

    This record is intended to be embedded in a database manifest. It stores
    enough information to reconstruct the corresponding RuntimeContext.

    ``derived_seed`` is redundant by design. It allows corruption, incompatible
    seed logic, or accidental scope changes to be detected during loading.
    """

    base_seed: int
    scope: tuple[SeedPart, ...]
    seed_version: str
    derived_seed: int

    bit_generator: str = _DEFAULT_BIT_GENERATOR
    record_version: str = _RUNTIME_RECORD_VERSION

    project_version: str | None = None
    config_digest: str | None = None

    python_version: str = field(
        default_factory=platform.python_version,
    )
    numpy_version: str = field(
        default_factory=lambda: np.__version__,
    )

    metadata: Mapping[str, JsonValue] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        if isinstance(self.base_seed, bool) or not isinstance(
            self.base_seed,
            int,
        ):
            raise TypeError("base_seed must be an integer")

        if not isinstance(self.scope, tuple):
            raise TypeError("scope must be a tuple of logical identifiers")

        if not isinstance(self.seed_version, str):
            raise TypeError("seed_version must be a string")

        if not self.seed_version.strip():
            raise ValueError("seed_version must not be empty")

        if not self.record_version:
            raise ValueError("record_version must not be empty")

        if self.bit_generator != _DEFAULT_BIT_GENERATOR:
            raise ValueError(
                f"Unsupported bit generator: {self.bit_generator!r}"
            )

        for part in self.scope:
            _encode_seed_part(part)

        expected_seed = derive_seed(
            self.base_seed,
            *self.scope,
            version=self.seed_version,
        )

        if self.derived_seed != expected_seed:
            raise ValueError(
                "RuntimeRecord derived_seed does not match "
                "base_seed, scope, and seed_version"
            )

        _validate_json_mapping(self.metadata)

    @classmethod
    def from_context(
        cls,
        context: RuntimeContext,
        *,
        project_version: str | None = None,
        config_digest: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> RuntimeRecord:
        """Create a persistable record from a runtime context."""
        return cls(
            base_seed=context.base_seed,
            scope=context.scope,
            seed_version=context.seed_version,
            derived_seed=context.seed(),
            bit_generator=_DEFAULT_BIT_GENERATOR,
            project_version=project_version,
            config_digest=config_digest,
            metadata=dict(metadata or {}),
        )

    def restore_context(self) -> RuntimeContext:
        """Reconstruct the exact deterministic runtime context."""
        return RuntimeContext(
            base_seed=self.base_seed,
            scope=self.scope,
            seed_version=self.seed_version,
        )

    def verify(self) -> None:
        """
        Verify that the stored seed still matches the current derivation logic.

        Validation also runs automatically in ``__post_init__``.
        """
        expected = derive_seed(
            self.base_seed,
            *self.scope,
            version=self.seed_version,
        )

        if expected != self.derived_seed:
            raise ValueError(
                f"Runtime seed mismatch: "
                f"stored={self.derived_seed}, expected={expected}"
            )

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert the record to a JSON-compatible dictionary."""
        return {
            "record_version": self.record_version,
            "base_seed": self.base_seed,
            "scope": [
                _seed_part_to_json(part)
                for part in self.scope
            ],
            "seed_version": self.seed_version,
            "derived_seed": self.derived_seed,
            "bit_generator": self.bit_generator,
            "project_version": self.project_version,
            "config_digest": self.config_digest,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> RuntimeRecord:
        """Construct and validate a record from decoded JSON."""
        scope_data = data.get("scope")

        if not isinstance(scope_data, list):
            raise ValueError("RuntimeRecord scope must be a list")

        scope = tuple(
            _seed_part_from_json(item)
            for item in scope_data
        )

        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("RuntimeRecord metadata must be an object")

        return cls(
            record_version=str(
                data.get(
                    "record_version",
                    _RUNTIME_RECORD_VERSION,
                )
            ),
            base_seed=int(data["base_seed"]),
            scope=scope,
            seed_version=str(data["seed_version"]),
            derived_seed=int(data["derived_seed"]),
            bit_generator=str(
                data.get(
                    "bit_generator",
                    _DEFAULT_BIT_GENERATOR,
                )
            ),
            project_version=_optional_str(
                data.get("project_version")
            ),
            config_digest=_optional_str(
                data.get("config_digest")
            ),
            python_version=str(
                data.get(
                    "python_version",
                    "unknown",
                )
            ),
            numpy_version=str(
                data.get(
                    "numpy_version",
                    "unknown",
                )
            ),
            metadata=dict(metadata),
        )

    def save_json(
        self,
        path: str | Path,
        *,
        indent: int = 2,
    ) -> None:
        """Atomically persist the record as UTF-8 JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        temporary_path.write_text(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=indent,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(output_path)

    @classmethod
    def load_json(
        cls,
        path: str | Path,
    ) -> RuntimeRecord:
        """Load and validate a runtime record from JSON."""
        input_path = Path(path)

        data = json.loads(
            input_path.read_text(encoding="utf-8")
        )

        if not isinstance(data, dict):
            raise ValueError(
                "RuntimeRecord JSON root must be an object"
            )

        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Immutable runtime context
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """
    Immutable deterministic random namespace.

    Creating child contexts does not consume random state.
    """

    base_seed: int
    scope: tuple[SeedPart, ...] = ()
    seed_version: str = _DEFAULT_SEED_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.base_seed, bool) or not isinstance(
            self.base_seed,
            int,
        ):
            raise TypeError("base_seed must be an integer")

        if not isinstance(self.scope, tuple):
            raise TypeError("scope must be a tuple of logical identifiers")

        if not isinstance(self.seed_version, str):
            raise TypeError("seed_version must be a string")

        if not self.seed_version.strip():
            raise ValueError("seed_version must not be empty")

        for part in self.scope:
            _encode_seed_part(part)

    def child(
        self,
        *scope: SeedPart,
    ) -> RuntimeContext:
        """Create a deterministic child namespace."""
        for part in scope:
            _encode_seed_part(part)

        return RuntimeContext(
            base_seed=self.base_seed,
            scope=self.scope + tuple(scope),
            seed_version=self.seed_version,
        )

    def for_sample(
        self,
        sample_id: str | int,
    ) -> RuntimeContext:
        """Create the canonical runtime namespace of one sample."""
        return self.child("sample", sample_id)

    def seed(
        self,
        *scope: SeedPart,
    ) -> int:
        """Return a stable unsigned 64-bit seed."""
        return derive_seed(
            self.base_seed,
            *self.scope,
            *scope,
            version=self.seed_version,
        )

    def numpy_rng(
        self,
        *scope: SeedPart,
    ) -> np.random.Generator:
        """Create a fresh deterministic NumPy RNG."""
        return np.random.Generator(
            np.random.PCG64DXSM(
                self.seed(*scope)
            )
        )

    def python_rng(
        self,
        *scope: SeedPart,
    ) -> random.Random:
        """Create a fresh deterministic standard-library RNG."""
        return random.Random(
            self.seed(*scope)
        )

    def uint32_seed(
        self,
        *scope: SeedPart,
    ) -> int:
        """Return a seed for APIs limited to unsigned 32-bit values."""
        return self.seed(*scope) & 0xFFFF_FFFF

    def uint63_seed(
        self,
        *scope: SeedPart,
    ) -> int:
        """Return a non-negative signed-64-bit-compatible seed."""
        return self.seed(*scope) & 0x7FFF_FFFF_FFFF_FFFF

    def record(
        self,
        *,
        project_version: str | None = None,
        config_digest: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> RuntimeRecord:
        """Create a persistable record of this runtime namespace."""
        return RuntimeRecord.from_context(
            self,
            project_version=project_version,
            config_digest=config_digest,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Optional configuration digest
# ---------------------------------------------------------------------------


def digest_config(
    config: Mapping[str, JsonValue],
) -> str:
    """
    Produce a stable digest of a JSON-compatible configuration.

    The digest is useful in RuntimeRecord and manifest files. It does not
    participate in seed derivation unless explicitly added to the runtime
    scope.
    """
    _validate_json_mapping(config)

    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return blake2b(
        payload,
        digest_size=16,
        person=b"rdb-config-v1",
    ).hexdigest()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _seed_part_to_json(
    value: SeedPart,
) -> dict[str, JsonValue]:
    """
    Preserve seed-part types during JSON serialization.

    Type preservation matters because integer ``1`` and string ``"1"`` must
    produce different seed paths.
    """
    if isinstance(value, Enum):
        raise TypeError(
            "Enum values are not valid seed-scope identifiers; "
            "pass enum.value explicitly"
        )

    if isinstance(value, bool):
        raise TypeError(
            "Boolean values are not valid seed-scope identifiers"
        )

    if isinstance(value, int):
        return {"type": "int", "value": value}

    if isinstance(value, str):
        if not value.strip():
            raise ValueError(
                "String seed-scope identifiers must not be empty"
            )
        return {"type": "str", "value": value}

    raise TypeError(
        "Seed scopes accept only logical identifiers of type str or int; "
        f"got {type(value).__name__}"
    )


def _seed_part_from_json(
    data: Any,
) -> SeedPart:
    if not isinstance(data, Mapping):
        raise ValueError(
            "Serialized seed part must be an object"
        )

    part_type = data.get("type")
    value = data.get("value")

    if part_type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Invalid serialized int seed part")
        return value

    if part_type == "str":
        if not isinstance(value, str):
            raise ValueError("Invalid serialized str seed part")
        if not value.strip():
            raise ValueError("Serialized str seed part must not be empty")
        return value

    raise ValueError(
        f"Unknown serialized seed part type: {part_type!r}"
    )


def _optional_str(
    value: Any,
) -> str | None:
    if value is None:
        return None
    return str(value)


def _validate_json_mapping(
    value: Mapping[str, Any],
) -> None:
    """Fail early if metadata cannot be serialized deterministically."""
    try:
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Value must be JSON serializable and must not contain NaN "
            "or Infinity"
        ) from error


__all__ = [
    "SeedPart",
    "JsonValue",
    "RuntimeContext",
    "RuntimeRecord",
    "derive_seed",
    "digest_config",
]

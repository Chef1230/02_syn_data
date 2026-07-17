from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.runtime import RuntimeContext, RuntimeRecord, derive_seed


class Stage(str, Enum):
    SCHEMA = "schema"


class RuntimeContextTests(unittest.TestCase):
    def test_same_logical_scope_is_deterministic(self) -> None:
        runtime = RuntimeContext(42).for_sample("db_000123")

        first = runtime.numpy_rng("schema", "motif_003").integers(
            0,
            1_000,
            size=16,
        )
        second = runtime.numpy_rng("schema", "motif_003").integers(
            0,
            1_000,
            size=16,
        )

        self.assertEqual(first.tolist(), second.tolist())

    def test_record_round_trip_preserves_string_and_integer_ids(self) -> None:
        runtime = RuntimeContext(42).child(
            "sample",
            "db_000123",
            "attempt",
            2,
        )
        record = runtime.record(metadata={"sample_id": "db_000123"})

        decoded = RuntimeRecord.from_dict(record.to_dict())

        self.assertEqual(record, decoded)
        self.assertEqual(runtime, decoded.restore_context())

    def test_json_round_trip_preserves_runtime(self) -> None:
        record = RuntimeContext(42).for_sample("db_000123").record()

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "runtime.json"
            record.save_json(path)
            loaded = RuntimeRecord.load_json(path)

        self.assertEqual(record, loaded)

    def test_enum_requires_explicit_logical_value(self) -> None:
        with self.assertRaisesRegex(TypeError, "enum.value"):
            RuntimeContext(42).child(Stage.SCHEMA)

        normalized = RuntimeContext(42).child(Stage.SCHEMA.value)
        self.assertEqual(
            normalized.seed(),
            derive_seed(42, "schema"),
        )

    def test_non_logical_scope_values_are_rejected(self) -> None:
        invalid_values = (
            True,
            False,
            None,
            1.5,
            b"schema",
            Path("schema"),
        )

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    RuntimeContext(42).child(value)  # type: ignore[arg-type]

    def test_empty_string_id_is_rejected(self) -> None:
        for value in ("", "   "):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    RuntimeContext(42).child(value)

    def test_scope_must_be_an_immutable_tuple(self) -> None:
        with self.assertRaisesRegex(TypeError, "scope must be a tuple"):
            RuntimeContext(42, scope=["schema"])  # type: ignore[arg-type]

    def test_legacy_serialized_scope_types_are_rejected(self) -> None:
        record = RuntimeContext(42).child("schema").record().to_dict()
        record["scope"] = [{"type": "path", "value": "schema.json"}]

        with self.assertRaisesRegex(ValueError, "Unknown serialized"):
            RuntimeRecord.from_dict(record)

    def test_record_json_contains_only_logical_scope_ids(self) -> None:
        record = RuntimeContext(42).child("schema", 3).record()
        payload = json.loads(json.dumps(record.to_dict()))

        self.assertEqual(
            [
                {"type": "str", "value": "schema"},
                {"type": "int", "value": 3},
            ],
            payload["scope"],
        )


if __name__ == "__main__":
    unittest.main()

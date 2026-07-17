"""Default extension wiring.

Runtime primitives live in :mod:`rdb_prior.runtime`.  These re-exports keep
older extension imports working without maintaining a second implementation.
"""

from ..runtime import RuntimeContext, RuntimeRecord, derive_seed, digest_config

__all__ = [
    "RuntimeContext",
    "RuntimeRecord",
    "derive_seed",
    "digest_config",
]

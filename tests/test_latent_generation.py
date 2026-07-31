from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.generation.latent import _gaussian_mixture_root_cause


class GaussianMixtureRootCauseTests(unittest.TestCase):
    def test_supports_small_and_boundary_table_sizes(self) -> None:
        for rows in (1, 2, 4, 7, 8, 16):
            with self.subTest(rows=rows):
                values = _gaussian_mixture_root_cause(
                    np.random.default_rng(7),
                    rows,
                    4,
                )

                self.assertEqual((rows, 4), values.shape)
                self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()

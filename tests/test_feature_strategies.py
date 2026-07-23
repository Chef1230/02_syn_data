from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from rdb_prior.generation.feature_strategies import generate_feature_signal
from rdb_prior.instance.plan import FeatureSCMFamily


class FeatureStrategyTests(unittest.TestCase):
    def test_all_existing_scm_families_support_meta_parameters(self) -> None:
        context = np.random.default_rng(7).normal(size=(128, 12))
        for family in FeatureSCMFamily:
            first = generate_feature_signal(
                family,
                context,
                np.random.default_rng(11),
                noise_scale=0.001,
                signal_scale=3.0,
                activation_scale=10.0,
                output_scale=2.0,
                long_tail_enabled=True,
                long_tail_alpha=1.3,
            )
            second = generate_feature_signal(
                family,
                context,
                np.random.default_rng(11),
                noise_scale=0.001,
                signal_scale=3.0,
                activation_scale=10.0,
                output_scale=2.0,
                long_tail_enabled=True,
                long_tail_alpha=1.3,
            )
            self.assertTrue(np.isfinite(first).all())
            np.testing.assert_array_equal(first, second)
            self.assertAlmostEqual(0.0, float(first.mean()), places=8)
            self.assertAlmostEqual(1.0, float(first.std()), places=8)

    def test_invalid_long_tail_alpha_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than 1"):
            generate_feature_signal(
                FeatureSCMFamily.LINEAR,
                np.ones((8, 2)),
                np.random.default_rng(1),
                noise_scale=0.01,
                long_tail_enabled=True,
                long_tail_alpha=1.0,
            )


if __name__ == "__main__":
    unittest.main()

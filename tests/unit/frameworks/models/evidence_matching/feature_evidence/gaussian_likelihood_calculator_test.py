# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for the C1 Gaussian log-likelihood feature evidence calculator.

PLAIN-LANGUAGE OVERVIEW
=======================
These tests check three separate things, and it is worth keeping them apart:

1. **The maths is self-consistent.** Adding up log-likelihoods and then
   normalising really does produce a probability distribution that sums to 1 and
   puts its peak on the true object (Bishop, 2006, *Pattern Recognition and
   Machine Learning*, Sec. 4.3.4).
2. **The documented failure mode is real, not hypothetical.** The honest
   probability calculation genuinely does fall below the -1 value that Monty's
   displacer reserves to mean "no stored node was close enough to compare
   against." This is a test that *asserts a problem exists*, which is unusual but
   deliberate -- it is the evidence for the write-up's central negative finding.
3. **The class actually wired into experiments changes nothing.** The four
   ``TelemetryGaussianCalculator`` tests are the safety gate: if that class ever
   stops returning the shipped answer exactly, every benchmark comparison built
   on it is invalid.

Style follows the neighbouring ``calculator_test.py``: plain
``unittest.TestCase`` plus ``hypothesis`` for property-based checks. There is no
``conftest.py`` and no fixtures anywhere in this test tree.

Works cited
-----------
Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer.
Leadholm, N., Clay, V., Knudstrup, S., Lee, H., & Hawkins, J. (2026). "Thousand
    Brains systems." *Neural Computation, 38*(6), 845-896.
"""

from __future__ import annotations

import unittest

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from scipy.special import logsumexp

from tbp.monty.frameworks.models.evidence_matching.feature_evidence.calculator import (
    DefaultFeatureEvidenceCalculator,
)
from tbp.monty.frameworks.models.evidence_matching.feature_evidence.gaussian_likelihood_calculator import (  # noqa: E501
    FlooredGaussianLogLikelihoodCalculator,
    GaussianLogLikelihoodCalculator,
    TelemetryGaussianCalculator,
)

# The feature layout Monty actually ships with for a single sensor patch: three
# colour numbers (hue, saturation, value), two curvature numbers, and a pose that
# the feature calculator skips because the displacer scores it separately.
ORDER = ["pose_vectors", "hsv", "principal_curvatures_log"]

# Tolerances = how far an observation may drift and still count as a match.
# These are the shipped values from `conf/monty/learning_module/`.
TOL = {
    "hsv": np.array([0.1, 0.2, 0.2]),
    "principal_curvatures_log": np.ones(2),
    "pose_vectors": np.full(3, 0.35),
}

# Weights = how much each feature counts. Hue is weighted 2 and its companions
# 0.5, i.e. Monty cares about which colour something is more than how vivid it is.
WEIGHTS = {
    "hsv": np.array([2.0, 0.5, 0.5]),
    "principal_curvatures_log": np.ones(2),
    "pose_vectors": np.ones(3),
}


def _calc(cls, stored, query, tolerances=TOL, weights=WEIGHTS):
    """Call any calculator the way Monty's scorer does: keyword arguments only.

    Returns:
        The per-node feature evidence produced by ``cls``.
    """
    return cls.calculate(
        channel_feature_array=stored,
        channel_feature_order=ORDER,
        channel_feature_weights=weights,
        channel_query_features=query,
        channel_tolerances=tolerances,
    )


class GaussianLogLikelihoodCalculatorTest(unittest.TestCase):
    def setUp(self) -> None:
        # 64 pretend stored points on an object's surface, with a fixed seed so
        # the numbers are the same on every run.
        rng = np.random.default_rng(0)
        self.stored = np.column_stack(
            [
                rng.uniform(0, 1, 64),
                rng.uniform(0, 1, 64),
                rng.uniform(0, 1, 64),
                rng.normal(0, 1, 64),
                rng.normal(0, 1, 64),
            ]
        )
        self.query = {
            "hsv": np.array([0.5, 0.5, 0.5]),
            "principal_curvatures_log": np.array([0.0, 0.0]),
            "pose_vectors": np.zeros(3),
        }

    def test_shape_matches_the_baseline_calculator(self) -> None:
        """Same input layout in, same output shape out -- the drop-in contract."""
        base = _calc(DefaultFeatureEvidenceCalculator, self.stored, self.query)
        ours = _calc(GaussianLogLikelihoodCalculator, self.stored, self.query)
        self.assertEqual(base.shape, ours.shape)

    def test_peak_is_attained_at_zero_residual(self) -> None:
        """A perfect match must score best, as any sane likelihood must."""
        exact = self.stored[:1].copy()
        query = {
            "hsv": exact[0, :3],
            "principal_curvatures_log": exact[0, 3:],
            "pose_vectors": np.zeros(3),
        }
        best = _calc(GaussianLogLikelihoodCalculator, exact, query)[0]
        others = _calc(GaussianLogLikelihoodCalculator, self.stored, query)
        self.assertAlmostEqual(best, float(np.max(others)), places=9)

    def test_strict_arm_can_fall_below_the_minus_one_sentinel(self) -> None:
        """Asserts the PROBLEM, on purpose.

        Monty's displacer writes -1 into a slot meaning "nothing stored was near
        enough to compare against." An honest log-likelihood is unbounded below,
        so it can score *worse* than -1 -- at which point a confidently refuted
        hypothesis gets rescued back up to the sentinel by a far-away neighbour.
        This test exists so the write-up can cite a passing check rather than an
        argument. If it ever fails, the negative finding needs revisiting.
        """
        ours = _calc(GaussianLogLikelihoodCalculator, self.stored, self.query)
        self.assertTrue(np.any(ours < -1.0))

    def test_floored_arm_respects_the_stock_range_convention(self) -> None:
        """The clamped arm stays in [0, SPAN], as the shipped one stays in [0, 1]."""
        ours = _calc(FlooredGaussianLogLikelihoodCalculator, self.stored, self.query)
        self.assertGreaterEqual(float(np.min(ours)), 0.0)
        self.assertLessEqual(
            float(np.max(ours)), FlooredGaussianLogLikelihoodCalculator.SPAN + 1e-12
        )

    def test_nan_stored_value_propagates_like_the_baseline(self) -> None:
        """A node with a missing stored value must not be silently treated as fine."""
        stored = self.stored.copy()
        stored[3, 1] = np.nan
        base = _calc(DefaultFeatureEvidenceCalculator, stored, self.query)
        ours = _calc(GaussianLogLikelihoodCalculator, stored, self.query)
        self.assertEqual(np.isnan(base[3]), np.isnan(ours[3]))
        self.assertEqual(int(np.isnan(base).sum()), int(np.isnan(ours).sum()))

    def test_circular_hue_uses_the_wrapped_distance(self) -> None:
        """Colour is a wheel, not a line: 0.98 is nearer 0.02 than it is 0.50."""
        stored = np.array([[0.02, 0.5, 0.5, 0.0, 0.0], [0.50, 0.5, 0.5, 0.0, 0.0]])
        query = {
            "hsv": np.array([0.98, 0.5, 0.5]),
            "principal_curvatures_log": np.array([0.0, 0.0]),
            "pose_vectors": np.zeros(3),
        }
        ours = _calc(GaussianLogLikelihoodCalculator, stored, query)
        self.assertGreater(ours[0], ours[1])

    def test_accumulate_then_softmax_is_the_exact_posterior(self) -> None:
        """The headline result: adding up scores and normalising gives a probability.

        Show one candidate 20 noisy views of itself, keep a running total of the
        log-likelihoods, then exponentiate and normalise. The result is a genuine
        probability distribution -- it sums to 1 -- and its peak sits on the
        object the observations actually came from. This is what "Monty is 87%
        sure it is a mug" would mean, and the shipped 0-to-1 score cannot supply
        it (Bishop, 2006, Sec. 4.3.4).
        """
        rng = np.random.default_rng(3)
        cand = self.stored[:5]
        sigma = np.full(5, 0.1)
        tol = {
            "hsv": sigma[:3],
            "principal_curvatures_log": sigma[3:],
            "pose_vectors": np.full(3, 0.35),
        }
        unit = {
            "hsv": np.ones(3),
            "principal_curvatures_log": np.ones(2),
            "pose_vectors": np.ones(3),
        }
        truth = 2
        accumulated = np.zeros(5)
        per_step = []
        for _ in range(20):
            obs = cand[truth] + rng.normal(0, sigma)
            query = {
                "hsv": obs[:3],
                "principal_curvatures_log": obs[3:],
                "pose_vectors": np.zeros(3),
            }
            step = _calc(GaussianLogLikelihoodCalculator, cand, query, tol, unit)
            per_step.append(step)
            accumulated += step
        direct = np.sum(per_step, axis=0)
        np.testing.assert_allclose(accumulated, direct, rtol=0, atol=1e-12)
        posterior = np.exp(accumulated - logsumexp(accumulated))
        self.assertAlmostEqual(float(posterior.sum()), 1.0, places=12)
        self.assertEqual(int(np.argmax(posterior)), truth)

    @given(offset=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False))
    def test_softmax_is_invariant_to_the_peak_shift(self, offset: float) -> None:
        """Adding the same number to every competitor changes nothing.

        Tried across many random offsets by ``hypothesis``. This is the licence
        for the floored arm to slide the whole scale up or down freely: only
        differences between hypotheses carry information, never their absolute
        level.
        """
        ours = _calc(GaussianLogLikelihoodCalculator, self.stored, self.query)
        a = np.exp(ours - logsumexp(ours))
        b = np.exp((ours + offset) - logsumexp(ours + offset))
        np.testing.assert_allclose(a, b, atol=1e-12)


class TelemetryGaussianCalculatorTest(unittest.TestCase):
    """The safety gate for the arm that is actually run: it must change nothing."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.stored = np.column_stack(
            [
                rng.uniform(0, 1, 64),
                rng.uniform(0, 1, 64),
                rng.uniform(0, 1, 64),
                rng.normal(0, 1, 64),
                rng.normal(0, 1, 64),
            ]
        )
        self.query = {
            "hsv": np.array([0.5, 0.5, 0.5]),
            "principal_curvatures_log": np.array([0.0, 0.0]),
            "pose_vectors": np.zeros(3),
        }
        # The record store is class-level, so it survives between tests unless
        # cleared -- exactly the hazard `reset_records` exists to handle.
        TelemetryGaussianCalculator.reset_records()

    def test_returns_the_baseline_bitwise(self) -> None:
        """The whole experimental design rests on this one assertion.

        Not "close to" the shipped answer -- identical, every bit. That is what
        licenses comparing a telemetry run directly against Monty's published
        benchmark numbers.
        """
        base = _calc(DefaultFeatureEvidenceCalculator, self.stored, self.query)
        ours = _calc(TelemetryGaussianCalculator, self.stored, self.query)
        np.testing.assert_array_equal(base, ours)

    def test_records_one_log_likelihood_vector_per_call(self) -> None:
        """The read-out accumulates as a side effect, one entry per scored step."""
        for _ in range(3):
            _calc(TelemetryGaussianCalculator, self.stored, self.query)
        self.assertEqual(len(TelemetryGaussianCalculator.RECORDS), 3)
        self.assertEqual(
            TelemetryGaussianCalculator.RECORDS[0].shape, (self.stored.shape[0],)
        )

    def test_recorded_values_reconstruct_a_normalised_posterior(self) -> None:
        """What the wiretap is for: a calibrated confidence, computed afterwards.

        Monty behaved exactly as shipped throughout; the probability is
        recovered from the recording once the run is over.
        """
        rng = np.random.default_rng(3)
        cand = self.stored[:5]
        sigma = np.full(5, 0.1)
        tol = {
            "hsv": sigma[:3],
            "principal_curvatures_log": sigma[3:],
            "pose_vectors": np.full(3, 0.35),
        }
        unit = {
            "hsv": np.ones(3),
            "principal_curvatures_log": np.ones(2),
            "pose_vectors": np.ones(3),
        }
        truth = 2
        for _ in range(15):
            obs = cand[truth] + rng.normal(0, sigma)
            query = {
                "hsv": obs[:3],
                "principal_curvatures_log": obs[3:],
                "pose_vectors": np.zeros(3),
            }
            _calc(TelemetryGaussianCalculator, cand, query, tol, unit)
        accumulated = np.sum(TelemetryGaussianCalculator.RECORDS, axis=0)
        posterior = np.exp(accumulated - logsumexp(accumulated))
        self.assertAlmostEqual(float(posterior.sum()), 1.0, places=12)
        self.assertEqual(int(np.argmax(posterior)), truth)

    def test_reset_records_clears_between_episodes(self) -> None:
        """Without this call, one episode's telemetry contaminates the next."""
        _calc(TelemetryGaussianCalculator, self.stored, self.query)
        TelemetryGaussianCalculator.reset_records()
        self.assertEqual(TelemetryGaussianCalculator.RECORDS, [])


if __name__ == "__main__":
    unittest.main()

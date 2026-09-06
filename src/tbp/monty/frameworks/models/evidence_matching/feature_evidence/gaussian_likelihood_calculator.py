# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""C1 -- reading Monty's feature-match score as a probability (Gaussian likelihood).

PLAIN-LANGUAGE OVERVIEW (read this first; no maths required)
============================================================

What this file is for
---------------------
Monty recognises objects by touching or looking at them one patch at a time. Each
time it takes a new observation it asks, of every object it has ever learned,
"how well does what I just saw match what I would expect if this were that
object, seen from this angle?" That question is answered by a *feature evidence
calculator*. Monty ships with one (``DefaultFeatureEvidenceCalculator``, next
door in ``calculator.py``). This file adds a second one that answers the same
question in the language of probability instead.

Why anyone would want a second one
----------------------------------
The shipped calculator returns a *score* between 0 and 1. A score is useful for
ranking -- bigger is better -- but it is not a probability, so you cannot say
"Monty is 87% sure this is a mug." The calculator here returns a *log-likelihood*,
which is a quantity you genuinely can turn into a percentage. Doing so gives the
system a calibrated confidence, which the reference publication says can be
extracted "at any time" but deliberately never defines.

The two ways of combining evidence, in everyday terms
-----------------------------------------------------
Suppose you are checking whether a stranger is your friend Alex, using three
clues: hair colour, height, and voice.

* The shipped calculator **averages** the three clues. If hair and height match
  perfectly but the voice is completely wrong, averaging gives roughly 2/3 --
  still a decent score. One bad clue gets diluted by the good ones.
* This calculator **adds up** the three clues on a logarithmic scale. A
  completely wrong voice contributes a large negative number that the two good
  clues cannot cancel out. One bad clue can veto the whole match.

Neither is "right"; they encode different beliefs about how clues relate. The
adding rule is the one probability theory prescribes if the clues are
independent. The averaging rule is what Monty's designers chose, on purpose --
see "The tension" below. Measured on Monty's real stored features, the two rules
rank candidate matches only moderately similarly (Spearman rank correlation
~0.47), so this is a genuine behavioural difference, not a relabelling.

What "tolerance" and "precision" mean here
------------------------------------------
Monty already stores, for every feature, a *tolerance*: how far an observation
may drift from the stored value and still count as a match. A tolerance of 0.1
on hue means "hues within 0.1 of each other are the same colour, as far as I am
concerned."

A tolerance is a statement about width. Probability theory has its own word for
the same idea inverted: *precision*, written as the Greek letter pi, defined as
one divided by the tolerance squared. Wide tolerance = low precision = "I do not
trust this measurement much." Narrow tolerance = high precision = "I trust this
measurement a lot." That one-line translation, tolerance -> precision, is the
hinge the whole thesis turns on, because precision is exactly the quantity that
the neuroscience literature says is disturbed in hallucination and delusion
(Feldman & Friston, 2010).

The tension this file exposes (this is the actual research result)
-----------------------------------------------------------------
Monty's designers made a deliberate architectural commitment: non-pose features
such as colour may *support* a hypothesis but may never *eliminate* one. Only
shape and orientation can rule an object out. They defend this with the human
shape-bias literature (children learning new words generalise by shape, not
colour -- Landau, Smith & Jones, 1988) and they validate it empirically: destroy
all colour information by painting every object uniform blue and Monty still
identifies 73.1% of 77 objects correctly.

A real log-likelihood cannot honour that commitment. By construction it is
unbounded below, so a bad colour match *can* drive a hypothesis arbitrarily
negative and kill it. You can clamp it to restore the commitment, but only by
choosing a clamp so wide that colour ends up outweighing shape by roughly ten to
one -- which inverts the very priority the commitment exists to protect.

So the finding is a negative one, and it is worth reporting as such: **a literal
probabilistic likelihood and Monty's shape-privileging commitment cannot both
hold in the current scoring path.** That is why the class actually wired into
experiments (``TelemetryGaussianCalculator``, at the bottom of this file) changes
nothing at all -- it hands back the shipped answer untouched and merely writes
the probabilistic answer to a side-channel for offline analysis.

An honest caveat about what is being claimed
--------------------------------------------
The reference publication explicitly *declines* the probabilistic reading. In
its own words the probabilistic description is "only to assist readers" and "no
normalized distributions ... are represented in Monty"; the authors prefer to
read the accumulating evidence as "akin to the accumulating membrane potential
in a neuron." This file therefore does not claim to have discovered that Monty
is secretly Bayesian. It claims something narrower and checkable: *if* you take
the authors' own analogy literally, exactly one term has the wrong shape, the
required change is well defined, and the cost of making it is measurable.

There is a pleasing consolation. A membrane potential that never leaks is a
perfect integrator, and a perfect integrator is precisely the probabilistically
optimal thing to do when the object you are looking at is not moving. Monty's
shipped settings (``past_weight = 1``, ``present_weight = 1`` in
``hypotheses_displacer.py``) sit exactly at that corner. The two readings agree
on the shipped configuration.

Three classes, in order of how safe they are to run
---------------------------------------------------
1. ``GaussianLogLikelihoodCalculator`` -- the honest probability calculation.
   Mathematically exact, but its output can fall below the -1 sentinel value the
   displacer uses to mean "no stored node was near enough to compare against,"
   which corrupts the comparison. Use for analysis, not for scoring.
2. ``FlooredGaussianLogLikelihoodCalculator`` -- the clamped variant. Fixes the
   sentinel collision at the cost described above. Specified so the cost can be
   quoted; NOT part of the executed work.
3. ``TelemetryGaussianCalculator`` -- **the one actually used.** Returns the
   shipped calculator's output byte-for-byte, so wiring it in cannot change any
   decision Monty makes, and records the probabilistic reading alongside.

WHERE THIS PLUGS IN
===================
Monty accepts ``feature_evidence_calculator`` as a normal constructor argument on
both ``DefaultHypothesesUpdater`` (``hypotheses_updater.py``) and
``BurstSamplingHypothesesUpdater`` (``burst_sampling.py``), and passes it down to
``DefaultFeatureEvidenceScorer`` (``feature_evidence/scorer.py``). No subclassing
of anything is required; this module only adds files and edits none.

Note that Monty is handed the *class*, never an instance -- it calls
``TheClass.calculate(...)`` directly. That is why every tunable below is a class
attribute configured by subclassing, rather than a constructor argument. The
shipped ``DefaultFeatureEvidenceCalculator`` does the same thing with its own
``SKIP_FEATURES`` / ``CIRCULAR_FEATURES`` / ``CATEGORICAL_FEATURES``, so this is
the house style, not a workaround.

Also note that the calculator runs at hypothesis *initialisation* as well as on
every step (``DefaultHypothesesUpdater._get_initial_hypothesis_space``), so a
scoring arm would change where the accumulator starts, not just how it grows.
Irrelevant for the telemetry class, which returns the shipped values.

WORKS CITED
===========
Abramowitz, M., & Stegun, I. A. (1964). *Handbook of Mathematical Functions.*
    National Bureau of Standards. -- modified Bessel function I0, used in the
    von Mises normaliser below.
Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer.
    -- Sec. 4.3.4 on the softmax as a normalised exponential of log-evidence,
    which is what turns the accumulator into a posterior.
Feldman, H., & Friston, K. J. (2010). "Attention, uncertainty, and free-energy."
    *Frontiers in Human Neuroscience, 4*, 215. -- precision as inverse variance,
    the concept the tolerance -> precision map imports.
Landau, B., Smith, L. B., & Jones, S. S. (1988). "The importance of shape in
    early lexical learning." *Cognitive Development, 3*(3), 299-321. -- the
    classic shape-bias result that motivates privileging shape over colour.
Leadholm, N., Clay, V., Knudstrup, S., Lee, H., & Hawkins, J. (2026). "Thousand
    Brains systems." *Neural Computation, 38*(6), 845-896.
    https://doi.org/10.1162/NECO.a.1508 -- the reference publication. Sec. 3.3
    (p. 858-860) declines the probabilistic reading and states the
    non-negativity commitment; p. 861 mentions extractable confidence; p. 871
    reports the 73.1% uniform-blue ablation.
Mardia, K. V., & Jupp, P. E. (2000). *Directional Statistics.* Wiley. -- the von
    Mises distribution used for hue, which is a circular quantity.

STATUS: research prototype, LLM-assisted starting point. Not yet run inside a
full Monty episode. See ``Private_Monty_journal/v4/AI_USE_LOG.md`` for the
verification record and ``c1_gaussian_likelihood_calculator.md`` for the full
write-up.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from typing import BinaryIO, ClassVar

import numpy as np
import numpy.typing as npt
from scipy.special import ive

from tbp.monty.frameworks.models.evidence_matching.feature_evidence.calculator import (
    DefaultFeatureEvidenceCalculator,
)


class GaussianLogLikelihoodCalculator:
    """Scores a stored node by probability rather than by a bounded 0-to-1 score.

    Plain language: for each candidate memory of an object, this asks "if this
    really were that object seen from this angle, how surprising would the thing
    I just observed be?" and answers on a logarithmic scale where 0 means
    "completely unsurprising" and large negative numbers mean "very surprising."

    Mirrors ``DefaultFeatureEvidenceCalculator``'s column layout, feature-kind
    taxonomy and circular-difference formula exactly. Only two things differ: the
    per-column kernel (a bell curve instead of a triangular tent), and the
    combination rule (a sum instead of an average).

    Not intended for scoring -- its output is unbounded below and collides with
    the displacer's -1 sentinel. See the module docstring and
    ``TelemetryGaussianCalculator``.

    Class attributes:
        SKIP_FEATURES: Features excluded from feature evidence. Pose is scored
            separately by the displacer, and notably has no tolerance at all, so
            the tolerance -> precision translation does not reach it.
        CIRCULAR_FEATURES: Features whose first sub-dimension wraps around.
            Hue is the example: 0.99 and 0.01 are both red and are *close*, even
            though they look far apart numerically.
        CATEGORICAL_FEATURES: Features that are either equal or not, with no
            meaningful notion of "close" (an object ID).
        CIRCULAR_RANGE: The period of the circular sub-dimension.
        P_MATCH: Probability assigned to a categorical match. AN UNCALIBRATED
            PLACEHOLDER, not a fitted value.
        SPAN: Output width used by the floored subclass only.
        FLOOR: If True, shift by the peak and clip into ``[0, SPAN]``.
    """

    SKIP_FEATURES = frozenset({"pose_vectors", "pose_fully_defined"})
    # Features whose first sub-dimension wraps around (circular features).
    CIRCULAR_FEATURES = frozenset({"hsv"})
    CATEGORICAL_FEATURES = frozenset({"object_id"})
    CIRCULAR_RANGE = 1
    P_MATCH = 0.999
    SPAN = 12.0
    FLOOR = False

    @classmethod
    def calculate(
        cls,
        channel_feature_array: np.ndarray,
        channel_feature_order: list[str],
        channel_feature_weights: dict,
        channel_query_features: dict,
        channel_tolerances: dict,
    ) -> npt.NDArray[np.float64]:
        """Calculate the log-likelihood feature evidence for all nodes in a graph.

        Args:
            channel_feature_array: Stored features for every node, shape
                ``(n_nodes, n_columns)``. One row per remembered point on the
                object's surface.
            channel_feature_order: Feature names in column order.
            channel_feature_weights: Per-feature combination weights.
            channel_query_features: What was just observed, keyed by feature name.
            channel_tolerances: Per-feature tolerance, read here as a precision
                ``Pi = 1 / tolerance**2``.

        Returns:
            Feature evidence for all nodes, shape ``(n_nodes,)``. For the strict
            arm this is the weighted sum of per-column log-densities (unbounded
            below); for the floored arm it is that value shifted by its
            hypothesis-independent peak and clipped into ``[0, SPAN]``.
        """
        # Step 1: work out which column of the stored array holds which feature,
        # and which of the three kinds each column is. This exactly reproduces
        # the shipped calculator's bookkeeping so the two are comparable.
        (
            tolerance,
            weight,
            query,
            numeric,
            circular,
            categorical,
        ) = cls._layout(
            channel_feature_array.shape[1],
            channel_feature_order,
            channel_feature_weights,
            channel_query_features,
            channel_tolerances,
        )

        # Step 2: how far is each stored value from what we just observed?
        diffs = cls._differences(
            channel_feature_array, query, numeric, circular, categorical
        )

        # Step 3: THE HINGE OF THE WHOLE THESIS. Reinterpret each tolerance as a
        # precision, Pi = 1 / tolerance^2. "How much slack do I allow?" becomes
        # "how much do I trust this measurement?" (Feldman & Friston, 2010).
        precision = np.zeros_like(tolerance)
        usable = np.isfinite(tolerance) & (tolerance > 0)
        precision[usable] = 1.0 / tolerance[usable] ** 2

        log_density = np.zeros_like(diffs)

        # Step 4a: ordinary measurements get a bell curve (Gaussian). Written in
        # logarithms, the familiar bell shape becomes a downward parabola: the
        # penalty grows with the *square* of the error, so being twice as wrong
        # is four times as costly. This is the term that lets one bad feature
        # veto a match, which the averaging rule cannot do.
        if numeric.any():
            p = precision[numeric]
            log_density[:, numeric] = (
                0.5 * np.log(p)
                - 0.5 * np.log(2 * np.pi)
                - 0.5 * p * diffs[:, numeric] ** 2
            )

        # Step 4b: hue lives on a circle, so a bell curve on a straight line is
        # the wrong shape -- it would call red-at-0.99 and red-at-0.01 opposites.
        # The von Mises distribution is the bell curve's circular counterpart
        # (Mardia & Jupp, 2000, Directional Statistics).
        if circular.any():
            kappa = precision[circular] / (2 * np.pi) ** 2
            theta = 2 * np.pi * diffs[:, circular]
            # I0 is a modified Bessel function (Abramowitz & Stegun, 1964) and
            # overflows for large kappa, so use the exponentially scaled `ive`:
            # log I0(k) = k + log ive(0, k).
            log_norm = np.log(2 * np.pi) + kappa + np.log(ive(0, kappa))
            log_density[:, circular] = kappa * np.cos(theta) - log_norm

        # Step 4c: an object ID either matches or it does not; there is no
        # "close." A coin-flip (Bernoulli) likelihood covers that case.
        if categorical.any():
            matched = diffs[:, categorical] == 0
            log_density[:, categorical] = np.where(
                matched, np.log(cls.P_MATCH), np.log1p(-cls.P_MATCH)
            )

        # Step 5: SUM, not average. This single choice is what makes the running
        # total a genuine joint log-likelihood, so that feeding it to a softmax
        # yields a real posterior probability (Bishop, 2006, Sec. 4.3.4).
        evidence = np.nansum(log_density * weight, axis=1)

        # Preserve the shipped calculator's contract: a node missing a stored
        # value gets NaN evidence rather than having that column quietly dropped.
        used = numeric | circular | categorical
        evidence[np.isnan(channel_feature_array[:, used]).any(axis=1)] = np.nan

        if not cls.FLOOR:
            return evidence
        peak = cls._peak(precision, weight, numeric, circular, categorical)
        return np.clip(evidence - peak + cls.SPAN, 0.0, cls.SPAN)

    @classmethod
    def _peak(
        cls,
        precision: npt.NDArray[np.float64],
        weight: npt.NDArray[np.float64],
        numeric: npt.NDArray[np.bool_],
        circular: npt.NDArray[np.bool_],
        categorical: npt.NDArray[np.bool_],
    ) -> float:
        """Return the best score any node could possibly achieve: a perfect match.

        Subtracting this slides every score down so that a perfect match sits at
        zero. It is safe because the shift is the same for every hypothesis, and
        adding the same number to all competitors leaves their relative
        probabilities untouched -- an additive constant cancels in the softmax.

        Args:
            precision: Per-column precision.
            weight: Per-column combination weight.
            numeric: Mask of numeric columns.
            circular: Mask of circular columns.
            categorical: Mask of categorical columns.

        Returns:
            The maximum attainable weighted log-density sum.
        """
        peak = np.zeros_like(precision)
        if numeric.any():
            p = precision[numeric]
            peak[numeric] = 0.5 * np.log(p) - 0.5 * np.log(2 * np.pi)
        if circular.any():
            kappa = precision[circular] / (2 * np.pi) ** 2
            peak[circular] = kappa - (np.log(2 * np.pi) + kappa + np.log(ive(0, kappa)))
        if categorical.any():
            peak[categorical] = np.log(cls.P_MATCH)
        return float(np.nansum(peak * weight))

    @classmethod
    def _layout(
        cls,
        n_cols: int,
        channel_feature_order: list[str],
        channel_feature_weights: dict,
        channel_query_features: dict,
        channel_tolerances: dict,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.bool_],
        npt.NDArray[np.bool_],
        npt.NDArray[np.bool_],
    ]:
        """Reproduce the shipped calculator's column layout and feature-kind masks.

        Stored features are packed side by side into one wide table, so this
        walks the feature list working out which columns belong to which feature
        (hue/saturation/value take three, the two curvatures take two, and so
        on). Kept deliberately identical to ``DefaultFeatureEvidenceCalculator``
        so that any difference in output is attributable to the kernel alone.

        Args:
            n_cols: Number of stored feature columns.
            channel_feature_order: Feature names in column order.
            channel_feature_weights: Per-feature combination weights.
            channel_query_features: Observed feature values.
            channel_tolerances: Per-feature tolerance.

        Returns:
            ``(tolerance, weight, query, numeric, circular, categorical)``, each
            of length ``n_cols``.
        """
        tolerance = np.full(n_cols, np.nan)
        weight = np.full(n_cols, np.nan)
        query = np.full(n_cols, np.nan)
        numeric = np.zeros(n_cols, dtype=bool)
        circular = np.zeros(n_cols, dtype=bool)
        categorical = np.zeros(n_cols, dtype=bool)

        start = 0
        for feature in channel_feature_order:
            if feature in cls.SKIP_FEATURES:
                continue
            value = channel_query_features[feature]
            length = len(value) if hasattr(value, "__len__") else 1
            end = start + length
            query[start:end] = value
            tolerance[start:end] = channel_tolerances[feature]
            weight[start:end] = channel_feature_weights[feature]

            if feature in cls.CIRCULAR_FEATURES:
                # Only H wraps around; S and V are ordinary numbers.
                circular[start] = True
                numeric[start + 1 : end] = True
            elif feature in cls.CATEGORICAL_FEATURES:
                categorical[start:end] = True
            else:
                numeric[start:end] = True
            start = end

        return tolerance, weight, query, numeric, circular, categorical

    @classmethod
    def _differences(
        cls,
        stored: npt.NDArray[np.float64],
        query: npt.NDArray[np.float64],
        numeric: npt.NDArray[np.bool_],
        circular: npt.NDArray[np.bool_],
        categorical: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float64]:
        """Compute per-column differences exactly as the shipped calculator does.

        Args:
            stored: Stored feature array, shape ``(n_nodes, n_columns)``.
            query: Observed values laid out per column.
            numeric: Mask of numeric columns.
            circular: Mask of circular columns.
            categorical: Mask of categorical columns.

        Returns:
            Absolute differences (numeric, circular) or 0/1 mismatches
            (categorical).
        """
        diffs = np.zeros_like(stored)
        diffs[:, numeric] = np.abs(stored[:, numeric] - query[numeric])
        node_c, query_c = stored[:, circular], query[circular]
        # Hue wraps: the shortest way round the colour wheel, not the arithmetic
        # difference. Checks going forwards, standing still, and backwards.
        diffs[:, circular] = np.min(
            [
                np.abs(cls.CIRCULAR_RANGE + node_c - query_c),
                np.abs(node_c - query_c),
                np.abs(node_c - (query_c + cls.CIRCULAR_RANGE)),
            ],
            axis=0,
        )
        diffs[:, categorical] = (stored[:, categorical] != query[categorical]).astype(
            stored.dtype
        )
        return diffs


class FlooredGaussianLogLikelihoodCalculator(GaussianLogLikelihoodCalculator):
    """SPECIFIED BUT NOT EXECUTED: the clamped variant, kept so its cost is quotable.

    Plain language: this slides every score down so a perfect match sits at the
    top, then refuses to let any score fall below zero. That restores Monty's
    rule that colour may support but never eliminate a hypothesis (Leadholm et
    al., 2026, p. 859).

    The catch, measured on Monty's real stored features: at ``SPAN = 1`` -- the
    obvious choice, matching the shipped 0-to-1 range -- 99.5% of nodes are
    flattened against the floor, destroying almost all the information. A usable
    ``SPAN`` is about 12, at which point the colour term outweighs the pose term
    (which lives in -1 to 1) by roughly ten to one, inverting the very priority
    the rule exists to protect. The cost IS the result; this arm is not run.
    """

    FLOOR = True
    SPAN = 12.0


class TelemetryGaussianCalculator(GaussianLogLikelihoodCalculator):
    """THE ARM ACTUALLY RUN: shipped behaviour untouched, probability logged beside it.

    Plain language: this is a wiretap, not a rewrite. It hands back exactly what
    the shipped calculator returned, so plugging it in cannot change a single
    decision Monty makes -- the benchmark numbers must come out bit-for-bit
    identical, and if they do not, the wiring is wrong. On the way past, it
    quietly writes down the probabilistic answer for offline analysis.

    That side-channel is what yields the calibrated confidence the reference
    publication gestures at -- "a classification and corresponding confidence
    can be extracted at any time" (Leadholm et al., 2026, p. 861) -- but
    deliberately never defines, since "no normalized distributions ... are
    represented in Monty" (p. 860). From the recorded values you can reconstruct,
    per step, the posterior probability over objects, the surprise, and the
    posterior entropy, none of which changes what Monty did.

    Note the deliberate asymmetry with the two classes above: those would change
    behaviour and suspend an architectural commitment; this one changes nothing
    and only adds a read-out.

    Persistence: ``run_parallel.py`` runs each episode in a freshly spawned
    process with ``maxtasksperchild=1``, so anything held only on the class dies
    with the worker. Setting ``MONTY_C1_TELEMETRY_DIR`` streams each record to a
    per-process ``.npy`` stream instead, which both survives the worker and keeps
    memory flat. With the variable unset the records accumulate in ``RECORDS``,
    which is fine for a single in-process run or a unit test but will grow
    without bound over a long episode.
    """

    #: Environment variable naming the directory telemetry is streamed to.
    TELEMETRY_DIR_ENV: ClassVar[str] = "MONTY_C1_TELEMETRY_DIR"

    #: In-memory fallback, used only when ``TELEMETRY_DIR_ENV`` is unset.
    RECORDS: ClassVar[list] = []

    _handle: ClassVar[BinaryIO | None] = None

    @classmethod
    def reset_records(cls) -> None:
        """Clear accumulated telemetry and close any open stream.

        Without this, one episode's numbers bleed into the next, because the
        store lives on the class rather than on an instance.
        """
        cls.RECORDS = []
        if cls._handle is not None:
            cls._handle.close()
            cls._handle = None

    @classmethod
    def _sink(cls) -> BinaryIO | None:
        """Return the open telemetry stream, opening it on first use.

        Returns:
            An open binary handle, or None if ``TELEMETRY_DIR_ENV`` is unset.
        """
        if cls._handle is not None:
            return cls._handle
        directory = os.environ.get(cls.TELEMETRY_DIR_ENV)
        if not directory:
            return None
        path = Path(directory).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        cls._handle = (path / f"c1_telemetry_{os.getpid()}.npy").open("wb")
        atexit.register(cls.reset_records)
        return cls._handle

    @classmethod
    def calculate(cls, **kwargs) -> npt.NDArray[np.float64]:
        """Return the shipped evidence verbatim; log the log-likelihood alongside.

        Args:
            **kwargs: Forwarded unchanged to both calculators. Keyword-only,
                matching how ``DefaultFeatureEvidenceScorer`` invokes the
                calculator (see ``feature_evidence/scorer.py``).

        Returns:
            Exactly ``DefaultFeatureEvidenceCalculator.calculate(**kwargs)``.
        """
        baseline = DefaultFeatureEvidenceCalculator.calculate(**kwargs)
        record = GaussianLogLikelihoodCalculator.calculate(**kwargs)
        sink = cls._sink()
        if sink is None:
            cls.RECORDS.append(record)
        else:
            # Concatenated .npy stream; read back with repeated np.load(handle).
            np.save(sink, record)
        return baseline

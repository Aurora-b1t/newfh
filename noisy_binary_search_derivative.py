"""
Derivative-based Noisy Binary Search for Hoprate Adjustment
============================================================

A variation of the Multiplicative Weights Update (MWU) noisy binary search
algorithm that uses the *derivative* (rate of change) metric based on BER
and hoprate, instead of simple BER increase/decrease, to make directional decisions.

Core idea: instead of using a binary "BER went up/down" signal, we calculate:

    metric = ΔBER_percent / Δhoprate

Where the deltas are taken between the *current* observation and the most recent
*prior* observation whose hoprate was **different** from the current one:

  - ΔBER_percent = (current_BER - reference_BER) × 100
    (BER change in percentage points, e.g., 0.1 → 0.02 gives ΔBER_percent = -8)
  - Δhoprate = current_hoprate - reference_hoprate

Comparing against the immediately previous step is meaningless when the hoprate
has not changed (Δhoprate = 0).  Instead we keep a small history and always
compute the gradient against the last observation taken at a *different* hoprate.
For example, given (500 Hz, 0.20), (520 Hz, 0.15), (520 Hz, 0.16), the third step
uses the first step as its reference: ΔBER% = (0.16 − 0.20)×100, Δhoprate = 20.

History is pruned to at most two records — the latest observation and the most
recent different-hoprate observation — because once a different-hoprate reference
exists, any earlier (and even earlier same-hoprate) records are unnecessary.

Decision rule (configurable threshold, default -0.002):

  metric > threshold   →  answer = "target is to the LEFT"
                          → h ≤ h_curr compatible  →  w × 2(1−p)
                          → h > h_curr incompatible →  w × 2p

  metric ≤ threshold   →  answer = "target is to the RIGHT"
                          → h ≥ h_curr compatible  →  w × 2(1−p)
                          → h < h_curr incompatible →  w × 2p

Because the reference always has a different hoprate than the current
observation, Δhoprate is never 0 and no division-by-zero fallback is needed.

The rest of the algorithm (weight normalisation, weighted-median query
selection, convergence criterion) is identical to the original implementation.
"""

import numpy as np
from typing import Tuple, Optional

import settings


class DerivativeNoisyBinarySearch:
    """
    Derivative-based noisy binary search over a hoprate range [hoprate_min, hoprate_max].

    Uses Δhoprate / ΔBER derivative instead of simple BER comparison to make
    directional decisions.  The derivative is always computed against the most
    recent prior observation taken at a *different* hoprate (not necessarily the
    immediately previous step).

    Parameters
    ----------
    hoprate_min : float
        Minimum candidate hoprate (Hz).
    hoprate_max : float
        Maximum candidate hoprate (Hz).
    hoprate_step : float
        Discretisation step (Hz).  Default 10 — matches ``_apply_hoprate``.
    p : float
        Assumed noise probability, 0 ≤ p < 0.5.  Smaller p → faster convergence
        but less tolerance to noisy BER readings.
    delta : float
        Confidence / convergence threshold, 0 < δ ≤ 1.  When the maximum weight
        reaches 1 − δ the algorithm reports convergence.
    derivative_threshold : float
        Decision threshold for the derivative metric.  Default -0.002.
        metric > threshold → LEFT move; metric ≤ threshold → RIGHT move.
    seed : int or None
        RNG seed for reproducible query randomisation.
    """

    def __init__(
        self,
        hoprate_min: float = 10.0,
        hoprate_max: float = 1000.0,
        hoprate_step: float = 10.0,
        p: float = 0.1,
        delta: float = 0.01,
        derivative_threshold: float = -0.002,
        seed: Optional[int] = None,
    ):
        if not (0.0 <= p < 0.5):
            raise ValueError(f"p must be in [0, 0.5), got {p}")
        if not (0.0 < delta <= 1.0):
            raise ValueError(f"delta must be in (0, 1], got {delta}")

        self.hoprate_min = float(hoprate_min)
        self.hoprate_max = float(hoprate_max)
        self.hoprate_step = float(hoprate_step)
        self.p = float(p)
        self.delta = float(delta)
        self.derivative_threshold = float(derivative_threshold)

        # ---- candidate grid ---------------------------------------------------
        n = int(round((self.hoprate_max - self.hoprate_min) / self.hoprate_step)) + 1
        self.candidates = np.linspace(
            self.hoprate_min, self.hoprate_max, n, dtype=np.float64
        )
        self.n_candidates = len(self.candidates)

        # ---- internal state ---------------------------------------------------
        self.weights = np.ones(self.n_candidates, dtype=np.float64) / self.n_candidates

        # The query selected for the next environment step (the hoprate that will
        # be, or just was, tested).  Set by ``_select_hoprate``.
        self._current_idx: Optional[int] = None
        self._current_hoprate: Optional[float] = None

        # Latest completed observation (hoprate actually tested + its BER).
        self._obs_hoprate: Optional[float] = None
        self._obs_ber: Optional[float] = None

        # Most recent completed observation whose hoprate differs from
        # ``_obs_hoprate`` — the reference against which the gradient is taken.
        # Its hoprate is guaranteed != ``_obs_hoprate`` whenever it is not None.
        self._ref_hoprate: Optional[float] = None
        self._ref_ber: Optional[float] = None

        self._step_count: int = 0

        self._rng = np.random.RandomState(seed)

        # ---- diagnostic state -------------------------------------------------
        self._last_derivative: Optional[float] = None
        self._last_delta_hoprate: Optional[float] = None
        self._last_delta_ber: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> float:
        """
        Reset all internal state and return the initial hoprate to test.

        Returns
        -------
        float
            The first hoprate selected by the (uniform) weighted median.
        """
        self.weights = np.ones(self.n_candidates, dtype=np.float64) / self.n_candidates
        self._current_idx = None
        self._current_hoprate = None
        self._obs_hoprate = None
        self._obs_ber = None
        self._ref_hoprate = None
        self._ref_ber = None
        self._step_count = 0
        self._last_derivative = None
        self._last_delta_hoprate = None
        self._last_delta_ber = None
        return self._select_hoprate()

    def seed_observation(self, hoprate: float, ber: float) -> None:
        """
        Seed the algorithm with an initial (hoprate, BER) observation.

        Use this when the first tested hoprate is chosen externally (not via
        ``reset()``).  The seeded pair is recorded as the latest observation; if
        a previous observation exists at a different hoprate it is kept as the
        reference, so the next ``step()`` can immediately compute a gradient.

        Parameters
        ----------
        hoprate : float
            The hoprate that was actually tested.
        ber : float
            The mean BER observed at that hoprate.
        """
        idx = int(np.argmin(np.abs(self.candidates - float(hoprate))))
        h_new = float(self.candidates[idx])
        ber = float(ber)

        if self._obs_ber is not None and h_new != self._obs_hoprate:
            # Promote the previous latest observation to the reference.
            self._ref_hoprate = self._obs_hoprate
            self._ref_ber = self._obs_ber
        # If h_new == _obs_hoprate, keep the existing reference and just refresh
        # the latest BER below.

        self._current_idx = idx
        self._current_hoprate = h_new
        self._obs_hoprate = h_new
        self._obs_ber = ber

    def step(self, ber: float) -> float:
        """
        Update weights using the observed BER and return the next hoprate.

        The BER is compared against the most recent *prior observation at a
        different hoprate* (the reference) via the derivative metric.  If the
        hoprate just tested matches the latest stored observation, only that
        observation's BER is refreshed (the reference is reused).  If the hoprate
        changed, the previous latest observation becomes the new reference.

        On the very first call (no prior observation exists) the weights stay
        uniform and only a new hoprate is selected.  No update is performed until
        two observations at *different* hoprates have been seen.

        Parameters
        ----------
        ber : float
            Mean BER observed during the environment step that just finished.

        Returns
        -------
        float
            The hoprate to use for the next environment step.
        """
        self._step_count += 1
        ber = float(ber)

        # Hoprate that was just tested (selected at the end of the previous step).
        h_new = self._current_hoprate

        # First observation: record it, no reference to compare against yet.
        if self._obs_ber is None:
            self._obs_hoprate = h_new
            self._obs_ber = ber
            return self._select_hoprate()

        if h_new == self._obs_hoprate:
            # Same hoprate as the latest observation: refresh its BER to the
            # latest reading (older readings at this hoprate are discarded).
            # The reference (different hoprate) is reused as-is.
            self._obs_ber = ber
        else:
            # Hoprate changed: the previous latest observation becomes the new
            # reference (its hoprate differs from h_new).  The old reference is
            # no longer needed and is discarded.
            self._ref_hoprate = self._obs_hoprate
            self._ref_ber = self._obs_ber
            self._obs_hoprate = h_new
            self._obs_ber = ber

        # Update weights whenever a different-hoprate reference is available.
        if (self._ref_ber is not None and self._ref_hoprate is not None
                and self._ref_hoprate != h_new):
            self._update_weights(ber, self._ref_ber, h_new, self._ref_hoprate)
        else:
            # No different-hoprate reference yet (only one hoprate seen so far):
            # a real gradient cannot be formed.  Force a move so the weighted
            # median shifts and a different hoprate gets sampled next — this is
            # the same bootstrap the original algorithm used for Δhoprate == 0.
            self._force_move()
        self._normalize()

        return self._select_hoprate()

    def get_best_hoprate(self) -> float:
        """Return the hoprate with the highest current weight (MAP estimate)."""
        return self.candidates[np.argmax(self.weights)]

    def get_weighted_average(self) -> float:
        """Return the weighted-average hoprate (soft estimate)."""
        return float(np.average(self.candidates, weights=self.weights))

    def is_converged(self) -> bool:
        """Return True when max(weight) ≥ 1 − δ."""
        return bool(np.max(self.weights) >= 1.0 - self.delta)

    def get_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(candidates, weights)`` for diagnostics / plotting."""
        return self.candidates.copy(), self.weights.copy()

    def get_last_derivative(self) -> Optional[float]:
        """Return the derivative value from the last weight update."""
        return self._last_derivative

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_hoprate(self) -> Optional[float]:
        return self._current_hoprate

    @property
    def current_idx(self) -> Optional[int]:
        return self._current_idx

    @property
    def step_count(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    # Internal: query selection  (paper Algorithm 3.1)
    # ------------------------------------------------------------------

    def _select_hoprate(self) -> float:
        """
        Weighted-median query selection with randomisation (paper Algorithm 3.1).

        1. Find index *k* s.t. cumulative weight crosses W/2.
        2. Randomise between *k* and *k+1* with probability α proportional to
           the weight imbalance around the median.
        """
        total = np.sum(self.weights)
        cumulative = np.cumsum(self.weights)

        # weighted median index
        k = int(np.searchsorted(cumulative, total / 2.0))
        k = min(k, self.n_candidates - 1)

        if k < self.n_candidates - 1 and self.weights[k] > 0:
            sum_left = cumulative[k] - self.weights[k]   # Σ₁ᵏ⁻¹
            sum_right = total - cumulative[k]             # Σₖ₊₁ⁿ

            # α = (Σ₁ᵏ − Σₖ₊₁ⁿ) / (2 w_k)
            alpha = (sum_left + self.weights[k] - sum_right) / (2.0 * self.weights[k])
            alpha = float(np.clip(alpha, 0.0, 1.0))

            chosen = k if self._rng.random() < alpha else k + 1
        else:
            chosen = k

        self._current_idx = int(chosen)
        self._current_hoprate = float(self.candidates[chosen])
        return self._current_hoprate

    # ------------------------------------------------------------------
    # Internal: MWU weight update — DERIVATIVE VERSION
    # ------------------------------------------------------------------

    def _update_weights(self, ber_curr: float, ber_ref: float,
                        h_curr: float, h_ref: float) -> None:
        """
        Apply the MWU weight update using the derivative metric.

        The gradient is taken between the current observation (``h_curr``,
        ``ber_curr``) and the most recent prior observation at a *different*
        hoprate (``h_ref``, ``ber_ref``).  The current hoprate *h_curr* is the
        query element (split point).  The directional answer is derived from:

            delta_ber_percent = (ber_curr - ber_ref) * 100
            delta_hoprate     = h_curr - h_ref          (guaranteed != 0)
            metric            = delta_ber_percent / delta_hoprate

            metric > threshold   →  answer = LEFT   →  h ≤ h_curr compatible
            metric ≤ threshold   →  answer = RIGHT  →  h ≥ h_curr compatible

        Because ``h_ref`` is always a different hoprate than ``h_curr``,
        ``delta_hoprate`` is never zero and no clamp is required.

        Compatible weights are multiplied by 2(1−p), incompatible by 2p.
        The split point is always the current hoprate index.
        """
        curr_idx = self._current_idx
        if curr_idx is None:
            return

        # Δhoprate is guaranteed non-zero: the reference hoprate always differs
        # from the current hoprate (enforced by ``step`` / ``seed_observation``).
        delta_hoprate = h_curr - h_ref
        delta_ber = ber_curr - ber_ref
        self._last_delta_hoprate = delta_hoprate
        self._last_delta_ber = delta_ber

        metric = (delta_ber * 100) / delta_hoprate
        self._last_derivative = metric

        self._apply_metric(metric, curr_idx)

    def _force_move(self) -> None:
        """
        Bootstrap fallback used before any different-hoprate reference exists.

        With only one hoprate observed so far, no real gradient can be formed.
        Force a RIGHT move (metric just below the threshold) so the weighted
        median shifts and a different hoprate is sampled on the next step —
        matching the original algorithm's behaviour for the Δhoprate == 0 case.
        """
        curr_idx = self._current_idx
        if curr_idx is None:
            return

        self._last_delta_hoprate = 0.0
        self._last_delta_ber = 0.0
        self._last_derivative = self.derivative_threshold - 0.001  # ≤ threshold → RIGHT

        self._apply_metric(self._last_derivative, curr_idx)

    def _apply_metric(self, metric: float, curr_idx: int) -> None:
        """Apply the MWU weight update for a given metric and split point."""
        indices = np.arange(self.n_candidates)

        # Decision based on threshold
        if metric > self.derivative_threshold:
            # metric > threshold → answer = LEFT
            # → left side favoured
            favoured = indices < curr_idx
        else:
            # metric ≤ threshold → answer = RIGHT
            # → current + right side favoured
            favoured = indices >= curr_idx

        n_fav = np.sum(favoured)
        if n_fav == 0 or n_fav == self.n_candidates:
            return

        self.weights[favoured] *= 2.0 * (1.0 - self.p)
        self.weights[~favoured] *= 2.0 * self.p

    def _normalize(self) -> None:
        """Renormalise weights to sum to 1."""
        s = np.sum(self.weights)
        if s > 0:
            self.weights /= s
        else:
            # Degenerate fallback — reset to uniform
            self.weights = np.ones(self.n_candidates, dtype=np.float64) / self.n_candidates


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    def _sim_ber(hoprate: float, optimal: float, max_hr: float,
                 baseline: float = 0.1, scale: float = 0.3,
                 rng: np.random.RandomState = None) -> float:
        """Simulated noisy BER: lower near *optimal* hoprate."""
        dist = abs(hoprate - optimal) / max_hr
        noise = rng.normal(0, 0.02) if rng else 0.0
        return max(0.0, min(1.0, baseline + scale * dist + noise))

    def run_test(label: str, optimal: float, steps: int = 80, threshold: float = -0.002):
        nbs = DerivativeNoisyBinarySearch(
            hoprate_min=10.0, hoprate_max=1000.0, hoprate_step=10.0,
            p=0.1, delta=0.05,
            derivative_threshold=threshold,
            seed=settings.RANDOM_SEED,
        )
        print(f"\n{'='*60}")
        print(f"Mode: {label}")
        print(f"True optimal hoprate: {optimal:.0f} Hz")
        print(f"Derivative threshold: {threshold}")
        print(f"{'='*60}")

        h = nbs.reset()
        print(f"Step {nbs.step_count:3d}  init hoprate = {h:6.0f} Hz")

        for i in range(steps):
            ber = _sim_ber(nbs.current_hoprate, optimal, nbs.hoprate_max,
                           rng=nbs._rng)
            h = nbs.step(ber)

            metric = nbs.get_last_derivative()
            metric_str = f"{metric:10.4f}" if metric is not None else "      N/A"

            if nbs.is_converged():
                best = nbs.get_best_hoprate()
                wavg = nbs.get_weighted_average()
                print(f"Step {nbs.step_count:3d}  CONVERGED  "
                      f"hop={h:6.0f}  best={best:6.0f}  wavg={wavg:6.0f}  "
                      f"metric={metric_str}  max_w={np.max(nbs.weights):.4f}")
                print(f"  → estimate = {best:.0f} Hz  "
                      f"(error = {abs(best - optimal):.0f} Hz,  "
                      f"steps = {nbs.step_count})")
                return nbs

            if i % 10 == 0 or i == steps - 1:
                best = nbs.get_best_hoprate()
                wavg = nbs.get_weighted_average()
                max_w = np.max(nbs.weights)
                print(f"Step {nbs.step_count:3d}  hop={h:6.0f}  BER={ber:.4f}  "
                      f"best={best:6.0f}  wavg={wavg:6.0f}  "
                      f"metric={metric_str}  max_w={max_w:.4f}")

        best = nbs.get_best_hoprate()
        wavg = nbs.get_weighted_average()
        print(f"Did not converge in {steps} steps.")
        print(f"  best={best:.0f} Hz  wavg={wavg:.0f} Hz  "
              f"error={abs(best - optimal):.0f} Hz")
        return nbs

    # Test derivative-based MWU mapping
    run_test("Derivative-based (ΔBER%/Δhoprate ≤ threshold → RIGHT)",
             optimal=350.0, threshold=-0.002)

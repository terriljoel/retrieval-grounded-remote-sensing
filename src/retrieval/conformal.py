"""Split-conformal thresholds: a guarantee instead of a tuned number.

A tuned threshold says "0.85 looked good on this data". A conformal threshold
says "auto-accepting above this score has an error rate of at most alpha, with
probability 1 - delta over the draw of the calibration set". That is a
statistical claim about future proposals, and it costs one quantile.

The construction here is the standard split-conformal one for **selective
classification with a bounded error rate**, applied per harm type:

  1. Take a calibration set the model never saw, with a binary "this admission
     would be harmful" label for the harm in question.
  2. For a candidate threshold t, the empirical harm rate among admissions is
     r(t). Choose the smallest t whose *upper confidence bound* on r(t) is
     still <= alpha.
  3. The upper bound is a Clopper-Pearson binomial bound at level delta, which
     is what turns "the observed rate was fine" into "the true rate is fine".

Using the raw empirical rate instead would be the mistake this module exists to
avoid: at high thresholds only a handful of proposals are admitted, and 0/8
harmful looks like a 0% error rate while being consistent with a true rate of
over 30%. The bound is what stops the threshold being chosen on noise.

Exchangeability is the assumption, and it holds here by construction --
calibration and test proposals are drawn from the same synthetic pool and split
by image. It will *not* hold when the detector's real proposals arrive, and the
guarantee has to be re-earned on those.
"""

from __future__ import annotations

import numpy as np

DELTA = 0.10          # confidence level for the bound: 1 - delta = 90%


def clopper_pearson_upper(harmful: int, total: int, delta: float = DELTA) -> float:
    """Upper 1-delta confidence bound on a binomial rate.

    Exact rather than normal-approximate: at the thresholds that matter the
    admitted count is small and a Wald interval there is meaningless (it can
    even reach below zero).
    """
    if total == 0:
        return 1.0
    if harmful >= total:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(1.0 - delta, harmful + 1, total - harmful))


def calibrate(
    scores: np.ndarray,
    harmful: np.ndarray,
    alpha: float,
    delta: float = DELTA,
    min_admitted: int = 1,
) -> dict:
    """Lowest score threshold whose bounded harm rate stays within alpha.

    `scores` higher means more confident it is a safe accept. `harmful` is True
    where admitting this proposal would be the harm being bounded.

    Returns the threshold, the coverage it buys, and both the empirical and
    bounded harm rates. Threshold `inf` with zero coverage means no threshold
    on this calibration set could support the guarantee.
    """
    order = np.argsort(-scores, kind="stable")
    ranked_harm = harmful[order].astype(np.int64)
    cumulative = np.cumsum(ranked_harm)
    admitted = np.arange(1, len(scores) + 1)

    bounds = np.array([
        clopper_pearson_upper(int(cumulative[index]), int(admitted[index]), delta)
        for index in range(len(scores))
    ])
    feasible = (bounds <= alpha) & (admitted >= min_admitted)
    if not feasible.any():
        return {
            "threshold": float("inf"), "coverage": 0.0, "n_admitted": 0,
            "empirical_rate": 0.0, "bounded_rate": 1.0, "alpha": alpha, "delta": delta,
        }

    # The largest admitted count that still satisfies the bound -- the most
    # work the gate can take while keeping the guarantee.
    best = int(np.flatnonzero(feasible)[-1])
    return {
        "threshold": float(scores[order][best]),
        "coverage": float(admitted[best] / len(scores)),
        "n_admitted": int(admitted[best]),
        "empirical_rate": float(cumulative[best] / admitted[best]),
        "bounded_rate": float(bounds[best]),
        "alpha": alpha,
        "delta": delta,
    }


def evaluate(scores: np.ndarray, harmful: np.ndarray, threshold: float) -> dict:
    """Apply a calibrated threshold to a fresh set and report what happened."""
    admitted = scores >= threshold
    count = int(admitted.sum())
    return {
        "threshold": threshold,
        "coverage": float(count / len(scores)),
        "n_admitted": count,
        "observed_rate": float(harmful[admitted].mean()) if count else 0.0,
        "n_harmful_admitted": int(harmful[admitted].sum()),
    }


def _self_check() -> None:
    rng = np.random.default_rng(0)

    # A well-ordered score: harm concentrated at the bottom.
    scores = np.linspace(1.0, 0.0, 1000)
    harmful = np.zeros(1000, dtype=bool)
    harmful[800:] = True
    result = calibrate(scores, harmful, alpha=0.05)
    # It should admit most of the clean prefix and stop before the harm block.
    assert 0.75 <= result["coverage"] <= 0.85, result
    assert result["bounded_rate"] <= 0.05, result
    assert result["empirical_rate"] <= result["bounded_rate"], result

    # A tighter alpha must never buy more coverage.
    tight = calibrate(scores, harmful, alpha=0.01)
    assert tight["coverage"] <= result["coverage"], (tight, result)

    # The bound is what stops a lucky short prefix being trusted. With 8
    # admissions and zero harm the empirical rate is 0, but the bound is not.
    assert clopper_pearson_upper(0, 8) > 0.20, clopper_pearson_upper(0, 8)
    assert clopper_pearson_upper(0, 1000) < 0.01
    # More data at the same rate must tighten the bound, never loosen it.
    assert clopper_pearson_upper(5, 100) > clopper_pearson_upper(50, 1000)

    # Pure noise cannot support a strict guarantee at meaningful coverage.
    noise = rng.random(500)
    coin = rng.random(500) < 0.4
    useless = calibrate(noise, coin, alpha=0.01)
    assert useless["coverage"] < 0.05, useless

    # An impossible request returns the sentinel rather than a bad threshold.
    always = calibrate(noise, np.ones(500, dtype=bool), alpha=0.01)
    assert always["threshold"] == float("inf") and always["coverage"] == 0.0

    # evaluate() must agree with calibrate() on the calibration set itself.
    applied = evaluate(scores, harmful, result["threshold"])
    assert applied["n_admitted"] == result["n_admitted"], (applied, result)

    print("conformal self-check OK")


if __name__ == "__main__":
    _self_check()

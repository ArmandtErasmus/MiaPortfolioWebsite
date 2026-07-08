"""Stochastic market-return models used to drive the smoothed bonus fund.

The engine supports three regimes of investment-return generation:

* ``Gaussian``          – classic i.i.d. log-normal style annual returns.
* ``Student-t``         – fat-tailed returns (better captures crash risk).
* ``Regime-Switching``  – a two-state model that intermittently drops the
                          portfolio into a high-volatility "crisis" regime.

All generators return an array of *arithmetic* annual returns with shape
``(n_paths, n_years)`` so the rest of the platform can stay model-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MarketAssumptions:
    """Capital-market assumptions for the underlying investment portfolio."""

    expected_return: float = 0.09       # long-run arithmetic mean (p.a.)
    volatility: float = 0.11            # annualised standard deviation
    model: str = "Regime-Switching"     # Gaussian | Student-t | Regime-Switching
    dof: int = 5                        # Student-t degrees of freedom
    crisis_prob: float = 0.04           # annual probability of entering a crisis
    crisis_return: float = -0.22        # mean return while in the crisis regime
    crisis_vol: float = 0.30            # volatility while in the crisis regime


def simulate_returns(
    assumptions: MarketAssumptions,
    n_paths: int,
    n_years: int,
    seed: int | None = 42,
) -> np.ndarray:
    """Generate a ``(n_paths, n_years)`` matrix of annual portfolio returns."""
    rng = np.random.default_rng(seed)
    mu, sigma = assumptions.expected_return, assumptions.volatility
    shape = (n_paths, n_years)

    if assumptions.model == "Gaussian":
        returns = rng.normal(mu, sigma, size=shape)

    elif assumptions.model == "Student-t":
        dof = max(assumptions.dof, 3)
        # Scale the raw t so that its variance matches the target volatility.
        scale = sigma / np.sqrt(dof / (dof - 2))
        returns = mu + scale * rng.standard_t(dof, size=shape)

    elif assumptions.model == "Regime-Switching":
        normal = rng.normal(mu, sigma, size=shape)
        crisis = rng.normal(assumptions.crisis_return, assumptions.crisis_vol, size=shape)
        in_crisis = rng.random(size=shape) < assumptions.crisis_prob
        returns = np.where(in_crisis, crisis, normal)

    else:  # pragma: no cover - guarded by the UI selectbox
        raise ValueError(f"Unknown market model: {assumptions.model!r}")

    # Floor at total wipe-out to keep the accounting sane.
    return np.clip(returns, -0.95, None)


def summarise_returns(returns: np.ndarray) -> dict[str, float]:
    """Realised summary statistics of a simulated return matrix."""
    flat = returns.reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "volatility": float(np.std(flat)),
        "skew": float(_skew(flat)),
        "kurtosis": float(_kurtosis(flat)),
        "worst_year": float(np.min(flat)),
        "best_year": float(np.max(flat)),
    }


def _skew(x: np.ndarray) -> float:
    m = x - x.mean()
    s = x.std()
    return np.mean(m**3) / s**3 if s > 0 else 0.0


def _kurtosis(x: np.ndarray) -> float:
    m = x - x.mean()
    s = x.std()
    return np.mean(m**4) / s**4 - 3.0 if s > 0 else 0.0

"""Deterministic stress scenarios for the smoothed bonus fund.

Each scenario is a hand-crafted sequence of annual market returns representing
a historically-inspired market path.  Running the fund through them shows how
the smoothing mechanism and Bonus Smoothing Reserve behave under named,
explainable shocks, complementing the probabilistic Monte Carlo view.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fund import BonusStrategy, FundResult, project_fund


@dataclass
class Scenario:
    name: str
    description: str
    returns: list[float]


STRESS_LIBRARY: list[Scenario] = [
    Scenario(
        "Soft Landing",
        "A shallow correction that the reserve absorbs comfortably, bonuses hold.",
        [-0.05, 0.06, 0.10, 0.09, 0.08],
    ),
    Scenario(
        "2008 Global Financial Crisis",
        "A sharp ~35% drawdown followed by a strong multi-year recovery rally.",
        [-0.35, 0.22, 0.14, 0.11, 0.09],
    ),
    Scenario(
        "COVID-19 Shock (2020)",
        "A violent single-year crash with an unusually rapid V-shaped rebound.",
        [-0.24, 0.28, 0.16, 0.10],
    ),
    Scenario(
        "1970s Stagflation",
        "A grinding low/negative real-return decade, the hardest test of smoothing.",
        [0.02, -0.06, 0.01, -0.04, 0.03, -0.02, 0.05],
    ),
    Scenario(
        "Prolonged Bear Market",
        "Several consecutive down years that steadily drain the reserve.",
        [-0.12, -0.09, -0.05, 0.02, 0.06, 0.10],
    ),
    Scenario(
        "Rising Rate Shock",
        "A bond-and-equity repricing followed by a slow normalisation.",
        [-0.14, -0.04, 0.07, 0.09, 0.08],
    ),
    Scenario(
        "Lost Decade",
        "Ten years of anaemic, choppy returns averaging near zero.",
        [0.03, -0.08, 0.05, -0.03, 0.02, -0.06, 0.04, 0.01, -0.02, 0.03],
    ),
    Scenario(
        "Depression (No Recovery)",
        "A severe, sustained multi-year drawdown that overwhelms the reserve.",
        [-0.30, -0.16, -0.10, -0.04, 0.01, 0.03],
    ),
]


def run_scenario(
    scenario_returns: list[float],
    strategy: BonusStrategy,
    initial_funding: float,
) -> FundResult:
    """Project the fund through a single deterministic return path."""
    returns = np.array(scenario_returns, dtype=float).reshape(1, -1)
    return project_fund(returns, strategy, initial_funding=initial_funding)


def scenario_verdict(result: FundResult, strategy: BonusStrategy) -> dict[str, object]:
    """Pass / watch / fail assessment for a single deterministic scenario."""
    funding = result.funding_level[0]
    bonus = result.bonus_rate[0]
    min_funding = float(funding.min())
    end_funding = float(funding[-1])
    removed = bool(np.any(bonus < 0))
    recovered = end_funding >= strategy.lower_corridor

    # A deep trough during a crash is expected, smoothing protects the member's
    # *bonus*, not the funding level. What matters is whether the reserve absorbs
    # the shock and the fund recovers. FAIL = left underfunded, or a near-solvency
    # trough it never climbs back from.
    if not recovered or min_funding < 0.70:
        status, tone = "FAIL", "error"
    elif min_funding < strategy.lower_corridor or removed:
        status, tone = "WATCH", "warning"
    else:
        status, tone = "PASS", "success"

    return {
        "status": status,
        "tone": tone,
        "min_funding": min_funding,
        "end_funding": end_funding,
        "bonus_removed": removed,
        "mean_bonus": float(bonus.mean()),
        "recovered": recovered,
    }

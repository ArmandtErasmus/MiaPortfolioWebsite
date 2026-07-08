"""Smoothed bonus fund mechanics.

A smoothed bonus fund holds member savings at a *book value* that grows with
declared bonuses, while the assets backing those savings grow with volatile
market returns.  The ratio of the two is the **funding level**:

    funding level  =  market value of assets  /  book value of liabilities

The bonus smoothing algorithm withholds return in strong years (building the
Bonus Smoothing Reserve) and releases it in weak years, delivering members a
far smoother return stream than the underlying market.  This module implements
that algorithm in a fully vectorised way so it runs across thousands of Monte
Carlo paths at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BonusStrategy:
    """Parameters governing bonus declaration and reserve management."""

    target_funding: float = 1.00        # funding level the fund steers toward
    smoothing_years: float = 5.0        # horizon over which shocks are absorbed
    bonus_cap: float = 0.16             # maximum annual bonus that can be declared
    bonus_floor: float = 0.00           # normal minimum (vested bonuses can't fall)
    non_vested_removal: float = 0.10    # max bonus clawback allowed under stress
    lower_corridor: float = 0.925       # below this, non-vested bonus is removed
    upper_corridor: float = 1.075       # above this, special bonuses are boosted
    management_charge: float = 0.010    # annual charge deducted from assets


@dataclass
class FundResult:
    """Path-wise output of a smoothed fund projection.

    Every array is shaped ``(n_paths, n_years)`` unless noted otherwise.
    """

    funding_level: np.ndarray           # funding level after each declaration
    bonus_rate: np.ndarray              # declared smoothed bonus each year
    smoothed_return: np.ndarray         # EWMA of underlying market return
    market_return: np.ndarray           # raw market return that was fed in
    asset_value: np.ndarray             # market value of assets
    book_value: np.ndarray              # smoothed book value of liabilities

    @property
    def n_paths(self) -> int:
        return self.funding_level.shape[0]

    @property
    def n_years(self) -> int:
        return self.funding_level.shape[1]


def project_fund(
    returns: np.ndarray,
    strategy: BonusStrategy,
    initial_funding: float = 1.02,
) -> FundResult:
    """Roll the smoothed bonus fund forward one year at a time.

    Parameters
    ----------
    returns
        ``(n_paths, n_years)`` matrix of annual market returns.
    strategy
        Bonus declaration / reserving rules.
    initial_funding
        Funding level at outset (assets / book value).
    """
    n_paths, n_years = returns.shape

    book = np.ones(n_paths)                      # book value starts at 1.0
    assets = np.full(n_paths, initial_funding)   # assets carry the opening surplus
    # The smoothed state is tracked as a *log* return so the declared bonus targets
    # the geometric (not arithmetic) growth rate, this removes the volatility-drag
    # leak that would otherwise erode the funding level even in a fair market.
    smoothed_log = np.full(n_paths, _seed_bonus(strategy))

    f_hist = np.empty((n_paths, n_years))
    b_hist = np.empty((n_paths, n_years))
    s_hist = np.empty((n_paths, n_years))
    a_hist = np.empty((n_paths, n_years))
    v_hist = np.empty((n_paths, n_years))

    alpha = 1.0 / max(strategy.smoothing_years, 1.0)   # EWMA weight on new info

    for t in range(n_years):
        r = returns[:, t]

        # 1. Market moves the assets (net of the management charge).
        assets = assets * (1.0 + r - strategy.management_charge)

        # 2. Smoothed view of the underlying return (this is what creates the
        #    smoothing: bonuses track a moving average, not the latest jump). We
        #    smooth log returns so the central bonus tracks geometric growth.
        smoothed_log = alpha * np.log1p(r - strategy.management_charge) \
            + (1.0 - alpha) * smoothed_log
        smoothed = np.expm1(smoothed_log)            # net-of-charge smoothed return

        # 3. Funding level assessed *after* provisionally crediting the fair
        #    (smoothed) bonus. Measuring surplus against this, rather than the
        #    raw assets/book ratio, prevents the current year's market move
        #    from being double-counted as distributable surplus.
        funding_assessed = assets / (book * (1.0 + smoothed))

        # 4. Funding correction, amortise any genuine surplus/deficit back to
        #    target gradually over the smoothing horizon.
        correction = (funding_assessed - strategy.target_funding) / strategy.smoothing_years

        target_bonus = smoothed + correction

        # 5. Corridor management. When funding is weak, allow non-vested bonus
        #    to be removed (a negative floor); when strong, lift the cap.
        floor = np.where(
            funding_assessed < strategy.lower_corridor,
            -strategy.non_vested_removal,
            strategy.bonus_floor,
        )
        cap = np.where(
            funding_assessed > strategy.upper_corridor,
            strategy.bonus_cap * 1.25,
            strategy.bonus_cap,
        )
        bonus = np.clip(target_bonus, floor, cap)

        # 6. Book value grows with the declared bonus; funding recomputed.
        book = book * (1.0 + bonus)
        funding_after = assets / book

        f_hist[:, t] = funding_after
        b_hist[:, t] = bonus
        s_hist[:, t] = smoothed
        a_hist[:, t] = assets
        v_hist[:, t] = book

    return FundResult(
        funding_level=f_hist,
        bonus_rate=b_hist,
        smoothed_return=s_hist,
        market_return=returns,
        asset_value=a_hist,
        book_value=v_hist,
    )


def _seed_bonus(strategy: BonusStrategy) -> float:
    """A sensible opening value for the smoothed *log*-return state variable."""
    return float(np.log1p(0.08 - strategy.management_charge))


# --------------------------------------------------------------------------- #
# Aggregate analytics derived from a Monte Carlo FundResult
# --------------------------------------------------------------------------- #
def fund_analytics(result: FundResult, strategy: BonusStrategy) -> dict[str, float]:
    """Sustainability, risk and smoothing metrics across all simulated paths."""
    f = result.funding_level
    b = result.bonus_rate
    terminal = f[:, -1]

    # Volatility reduction: how much smoother is the member return vs market?
    member_vol = float(np.std(b))
    market_vol = float(np.std(result.market_return))
    vol_reduction = 1.0 - member_vol / market_vol if market_vol > 0 else 0.0

    # Path-wise minimum funding level (worst point each path ever reaches).
    path_min_funding = f.min(axis=1)

    return {
        "prob_underfunded_terminal": float(np.mean(terminal < 1.0)),
        "prob_breach_lower_corridor": float(np.mean(path_min_funding < strategy.lower_corridor)),
        "prob_bonus_removal": float(np.mean(np.any(b < 0.0, axis=1))),
        "prob_insolvent": float(np.mean(path_min_funding < 0.85)),
        "median_terminal_funding": float(np.median(terminal)),
        "funding_var95": float(np.percentile(terminal, 5)),   # 95% VaR (5th pct)
        "funding_cvar95": float(np.mean(terminal[terminal <= np.percentile(terminal, 5)])),
        "mean_bonus": float(np.mean(b)),
        "median_bonus": float(np.median(b)),
        "min_bonus": float(np.min(b)),
        "member_return_vol": member_vol,
        "market_return_vol": market_vol,
        "volatility_reduction": float(vol_reduction),
        "sustainable_bonus": float(np.percentile(b, 25)),      # prudent declarable level
    }


def percentile_bands(array: np.ndarray, levels=(5, 25, 50, 75, 95)) -> dict[int, np.ndarray]:
    """Return per-year percentile bands for a ``(n_paths, n_years)`` array."""
    return {lvl: np.percentile(array, lvl, axis=0) for lvl in levels}

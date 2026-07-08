"""Automated investment-commentary generation.

Two engines are provided:

* :func:`generate_narrative`, a deterministic, always-available engine that
  turns simulation analytics into client-ready commentary, risk summaries and
  management insights.  No API key required.
* :func:`generate_llm_narrative`, an optional wrapper that hands the same
  structured analytics to the Anthropic API for a richer, free-form write-up.

The rule-based engine is intentionally the default so the platform is fully
functional offline, while the LLM path demonstrates production AI integration.
"""
from __future__ import annotations

import json


def _pct(x: float, dp: int = 1) -> str:
    return f"{x * 100:.{dp}f}%"


def _funding_grade(median_terminal: float) -> str:
    if median_terminal >= 1.05:
        return "strongly funded"
    if median_terminal >= 1.00:
        return "adequately funded"
    if median_terminal >= 0.95:
        return "marginally funded"
    return "under strain"


def generate_narrative(
    analytics: dict,
    market_summary: dict,
    horizon: int,
    n_paths: int,
    strategy_summary: dict,
) -> str:
    """Compose a structured investment report from simulation analytics."""
    a = analytics
    grade = _funding_grade(a["median_terminal_funding"])
    vol_red = _pct(a["volatility_reduction"], 0)

    # ---- Executive summary ------------------------------------------------ #
    exec_summary = (
        f"Across **{n_paths:,} Monte Carlo paths** over a **{horizon}-year** horizon, "
        f"the smoothed bonus portfolio is projected to remain **{grade}**, with a median "
        f"terminal funding level of **{a['median_terminal_funding']:.1%}**. The smoothing "
        f"mechanism reduced member-experienced volatility by **{vol_red}** relative to the "
        f"underlying market, the core value proposition of the product is intact under the "
        f"modelled assumptions."
    )

    # ---- Bonus sustainability -------------------------------------------- #
    removal_risk = a["prob_bonus_removal"]
    if removal_risk < 0.05:
        sustain_tone = "highly sustainable"
    elif removal_risk < 0.15:
        sustain_tone = "broadly sustainable, with a modest tail risk"
    else:
        sustain_tone = "under pressure"

    bonus_section = (
        f"The declared bonus is **{sustain_tone}**. The central (median) declared bonus is "
        f"**{_pct(a['median_bonus'])}** p.a., and a prudently sustainable declaration, the "
        f"25th percentile of simulated bonuses, sits at **{_pct(a['sustainable_bonus'])}**. "
        f"There is a **{_pct(removal_risk)}** probability of having to remove non-vested "
        f"bonuses at some point over the horizon to protect the reserve."
    )

    # ---- Risk & funding --------------------------------------------------- #
    risk_section = (
        f"Funding risk is quantified by a 95% terminal Value-at-Risk of "
        f"**{a['funding_var95']:.1%}** (CVaR **{a['funding_cvar95']:.1%}**). The probability of "
        f"finishing underfunded (below 100%) is **{_pct(a['prob_underfunded_terminal'])}**, the "
        f"probability of breaching the lower reserve corridor at any point is "
        f"**{_pct(a['prob_breach_lower_corridor'])}**, and the probability of a severe funding "
        f"event (below 85%) is **{_pct(a['prob_insolvent'])}**. The underlying market was "
        f"modelled with a mean return of **{_pct(market_summary['mean'])}** and volatility of "
        f"**{_pct(market_summary['volatility'])}**, exhibiting excess kurtosis of "
        f"**{market_summary['kurtosis']:.2f}** (tail-risk sensitivity)."
    )

    # ---- Management insights --------------------------------------------- #
    insights: list[str] = []
    if a["prob_breach_lower_corridor"] > 0.15:
        insights.append(
            "Reserve resilience is a watch item, consider a more conservative bonus "
            "declaration or a de-risked strategic asset allocation to widen the buffer."
        )
    else:
        insights.append(
            "The Bonus Smoothing Reserve provides a comfortable buffer under current "
            "assumptions; the current declaration philosophy can be maintained."
        )
    if a["median_terminal_funding"] > 1.08:
        insights.append(
            "The fund is projected to accumulate surplus above the upper corridor, there is "
            "scope to enhance member outcomes via special or catch-up bonuses."
        )
    if a["prob_bonus_removal"] > 0.10:
        insights.append(
            "Non-vested bonus removal is a live possibility in adverse paths; ensure member "
            "communications and PPFM disclosures set expectations accordingly."
        )
    if a["volatility_reduction"] < 0.5:
        insights.append(
            "Smoothing is delivering less volatility dampening than typical, review the "
            "smoothing horizon and corridor width to strengthen the smoothing effect."
        )
    insights.append(
        f"Governance: monitor the funding level against the {strategy_summary['lower_corridor']:.1%}–"
        f"{strategy_summary['upper_corridor']:.1%} corridor at each declaration date."
    )

    insight_block = "\n".join(f"- {i}" for i in insights)

    report = f"""### Executive Summary
{exec_summary}

### Bonus Declaration & Sustainability
{bonus_section}

### Risk & Funding Assessment
{risk_section}

### Management Insights & Recommendations
{insight_block}

---
*Generated by the Smoothed Bonus Analytics commentary engine. Figures are model
outputs conditional on the stated capital-market assumptions and are not advice.*
"""
    return report


def build_llm_context(
    analytics: dict,
    market_summary: dict,
    horizon: int,
    n_paths: int,
    strategy_summary: dict,
) -> str:
    """Serialise the analytics into a compact JSON brief for an LLM."""
    return json.dumps(
        {
            "horizon_years": horizon,
            "monte_carlo_paths": n_paths,
            "analytics": analytics,
            "market": market_summary,
            "strategy": strategy_summary,
        },
        indent=2,
        default=float,
    )


def generate_llm_narrative(
    api_key: str,
    analytics: dict,
    market_summary: dict,
    horizon: int,
    n_paths: int,
    strategy_summary: dict,
    model: str = "claude-opus-4-8",
) -> str:
    """Optional: generate commentary with the Anthropic API.

    Raises if the ``anthropic`` package or key is unavailable so the caller can
    fall back to the deterministic engine.
    """
    from anthropic import Anthropic  # imported lazily; optional dependency

    context = build_llm_context(
        analytics, market_summary, horizon, n_paths, strategy_summary
    )
    prompt = (
        "You are an actuarial investment analyst writing client-ready commentary "
        "for a smoothed bonus fund. Using ONLY the JSON analytics below, write a "
        "concise report in Markdown with these sections: Executive Summary, Bonus "
        "Declaration & Sustainability, Risk & Funding Assessment, and Management "
        "Insights & Recommendations. Be precise with the numbers, use a measured "
        "institutional tone, and do not invent data.\n\n"
        f"```json\n{context}\n```"
    )

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1600,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if block.type == "text")

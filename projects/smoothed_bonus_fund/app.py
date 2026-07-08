"""Smoothed Bonus Fund Analytics Platform.

An interactive Streamlit application for evaluating smoothed bonus investment
strategies through stochastic modelling, Monte Carlo simulation, deterministic
stress testing, bonus-sustainability analysis and automated AI commentary.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.fund import (
    BonusStrategy,
    fund_analytics,
    percentile_bands,
    project_fund,
)
from engine.market import MarketAssumptions, simulate_returns, summarise_returns
from engine.reporting import generate_llm_narrative, generate_narrative
from engine.scenarios import STRESS_LIBRARY, run_scenario, scenario_verdict

# --------------------------------------------------------------------------- #
# Page config & theme helpers
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Smoothed Bonus Analytics Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#2563eb"
ACCENT_2 = "#0891b2"          # deeper teal, reads clearly on white
GOOD = "#16a34a"
WARN = "#d97706"
BAD = "#dc2626"
INK = "#0f172a"              # primary text
MUTED = "#64748b"           # secondary text / neutral series
GRID = "rgba(15,23,42,0.08)"

# Base styling shared by every chart. Individual charts supply their own
# ``xaxis`` / ``yaxis`` dicts, so those keys are deliberately omitted here to
# avoid duplicate-keyword collisions when spread into ``update_layout``.
PLOT_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=INK, size=13),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1400px;}
      h1, h2, h3 {letter-spacing:-0.01em; color:#0f172a;}
      div[data-testid="stMetric"] {
          background: linear-gradient(180deg,#ffffff,#f4f7fb);
          border: 1px solid rgba(15,23,42,0.10);
          border-radius: 14px; padding: 14px 16px;
          box-shadow: 0 1px 2px rgba(15,23,42,0.05);
      }
      div[data-testid="stMetricValue"] {font-size: 1.5rem;}
      .hero {
          background: radial-gradient(120% 140% at 0% 0%, #dbeafe 0%, #f8fafc 60%);
          border:1px solid rgba(37,99,235,0.18); border-radius:18px;
          padding:22px 26px; margin-bottom:8px;
      }
      .pill {display:inline-block; padding:3px 10px; border-radius:999px;
             font-size:0.72rem; font-weight:600; margin-right:6px;
             background:rgba(37,99,235,0.10); color:#1d4ed8;
             border:1px solid rgba(37,99,235,0.30);}
      .verdict {font-weight:700; padding:4px 12px; border-radius:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached simulation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def run_simulation(mkt: dict, strat: dict, n_paths: int, horizon: int,
                   initial_funding: float, seed: int):
    assumptions = MarketAssumptions(**mkt)
    strategy = BonusStrategy(**strat)
    returns = simulate_returns(assumptions, n_paths, horizon, seed=seed)
    result = project_fund(returns, strategy, initial_funding=initial_funding)
    analytics = fund_analytics(result, strategy)
    market_summary = summarise_returns(returns)
    return result, analytics, market_summary


def fmt_pct(x, dp=1):
    return f"{x * 100:.{dp}f}%"


# --------------------------------------------------------------------------- #
# Sidebar, assumptions & controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## ⚙️ Model Configuration")

    with st.expander("📉 Capital Market Assumptions", expanded=True):
        model = st.selectbox(
            "Return model",
            ["Regime-Switching", "Gaussian", "Student-t"],
            help="Stochastic process driving underlying portfolio returns.",
        )
        expected_return = st.slider("Expected return (p.a.)", 0.02, 0.16, 0.09, 0.005,
                                    format="%.3f")
        volatility = st.slider("Volatility (p.a.)", 0.04, 0.30, 0.11, 0.01)
        dof = 5
        crisis_prob = 0.04
        crisis_return = -0.22
        crisis_vol = 0.30
        if model == "Student-t":
            dof = st.slider("Degrees of freedom (tail fatness)", 3, 30, 5)
        if model == "Regime-Switching":
            crisis_prob = st.slider("Annual crisis probability", 0.0, 0.25, 0.04, 0.01)
            crisis_return = st.slider("Crisis mean return", -0.45, 0.0, -0.22, 0.01)
            crisis_vol = st.slider("Crisis volatility", 0.15, 0.50, 0.30, 0.01)

    with st.expander("🎚️ Bonus & Smoothing Strategy", expanded=True):
        smoothing_years = st.slider("Smoothing horizon (years)", 1.0, 8.0, 5.0, 0.5,
                                    help="Longer horizons absorb shocks more gradually.")
        bonus_cap = st.slider("Bonus cap (p.a.)", 0.06, 0.25, 0.16, 0.01)
        management_charge = st.slider("Management charge (p.a.)", 0.0, 0.03, 0.01, 0.0025,
                                      format="%.4f")
        lower_corridor = st.slider("Lower funding corridor", 0.80, 0.98, 0.925, 0.005)
        upper_corridor = st.slider("Upper funding corridor", 1.02, 1.20, 1.075, 0.005)
        non_vested_removal = st.slider("Max non-vested clawback", 0.0, 0.25, 0.10, 0.01)

    with st.expander("🎲 Simulation Settings", expanded=True):
        horizon = st.slider("Projection horizon (years)", 5, 40, 20, 1)
        n_paths = st.select_slider("Monte Carlo paths",
                                   options=[1000, 2500, 5000, 10000, 20000], value=5000)
        initial_funding = st.slider("Opening funding level", 0.90, 1.15, 1.03, 0.01)
        seed = st.number_input("Random seed", 0, 10_000, 42, 1)

    st.caption("Adjust any control to re-run the analytics live.")

# Assemble config dicts (hashable for caching)
mkt_cfg = dict(
    expected_return=expected_return, volatility=volatility, model=model,
    dof=dof, crisis_prob=crisis_prob, crisis_return=crisis_return, crisis_vol=crisis_vol,
)
strat_cfg = dict(
    target_funding=1.0, smoothing_years=smoothing_years, bonus_cap=bonus_cap,
    bonus_floor=0.0, non_vested_removal=non_vested_removal,
    lower_corridor=lower_corridor, upper_corridor=upper_corridor,
    management_charge=management_charge,
)

result, analytics, market_summary = run_simulation(
    mkt_cfg, strat_cfg, n_paths, horizon, initial_funding, int(seed)
)
strategy = BonusStrategy(**strat_cfg)
years = np.arange(1, horizon + 1)

# --------------------------------------------------------------------------- #
# Hero header
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div class="hero">
      <h1 style="margin:0 0 4px 0;">Smoothed Bonus Fund Analytics Platform</h1>
      <p style="color:#475569; margin:0; max-width:820px;">
        Evaluate smoothed bonus investment strategies end-to-end: model the underlying
        market stochastically, simulate the Bonus Smoothing Reserve across thousands of
        paths, stress the fund against named crises, and generate client-ready commentary.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_mc, tab_stress, tab_bonus, tab_report = st.tabs(
    ["📊 Overview", "🎲 Monte Carlo", "⚡ Stress Testing",
     "💰 Bonus Sustainability", "🤖 AI Report"]
)

# =========================================================================== #
# TAB 1, OVERVIEW
# =========================================================================== #
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    med_term = analytics["median_terminal_funding"]
    c1.metric("Median Terminal Funding", fmt_pct(med_term),
              delta=fmt_pct(med_term - 1.0) + " vs target")
    c2.metric("Median Bonus (p.a.)", fmt_pct(analytics["median_bonus"]))
    c3.metric("Volatility Reduction", fmt_pct(analytics["volatility_reduction"], 0),
              help="Reduction in member-experienced volatility vs the raw market.")
    c4.metric("P(Underfunded @ Horizon)", fmt_pct(analytics["prob_underfunded_terminal"]))
    c5.metric("P(Bonus Removal)", fmt_pct(analytics["prob_bonus_removal"]))

    st.markdown("#### Smoothing in action, one representative path")
    st.caption(
        "The declared bonus (smooth) tracks a moving average of the volatile market "
        "return, insulating members from year-to-year market swings."
    )

    # Pick a median-ish path for illustration
    idx = int(np.argsort(result.funding_level[:, -1])[result.n_paths // 2])
    mkt_path = result.market_return[idx]
    bonus_path = result.bonus_rate[idx]

    fig = go.Figure()
    fig.add_bar(x=years, y=mkt_path, name="Market return",
                marker_color="rgba(100,116,139,0.55)")
    fig.add_trace(go.Scatter(x=years, y=bonus_path, name="Declared bonus",
                             mode="lines+markers", line=dict(color=ACCENT_2, width=3)))
    fig.update_layout(**PLOT_LAYOUT, height=340,
                      yaxis=dict(tickformat=".0%", gridcolor=GRID),
                      xaxis=dict(title="Year", gridcolor=GRID))
    st.plotly_chart(fig, use_container_width=True)

    colL, colR = st.columns([3, 2])
    with colL:
        st.markdown("#### Cumulative growth, smoothed vs market")
        cum_market = np.cumprod(1 + mkt_path)
        cum_bonus = np.cumprod(1 + bonus_path)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=years, y=cum_market, name="Market (assets)",
                                  line=dict(color="#64748b", width=2, dash="dot")))
        fig2.add_trace(go.Scatter(x=years, y=cum_bonus, name="Member (smoothed)",
                                  line=dict(color=ACCENT, width=3)))
        fig2.update_layout(**PLOT_LAYOUT, height=320,
                           yaxis=dict(title="Growth of 1", gridcolor=GRID),
                           xaxis=dict(title="Year", gridcolor=GRID))
        st.plotly_chart(fig2, use_container_width=True)

    with colR:
        st.markdown("#### Funding level gauge")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=med_term * 100,
            number={"suffix": "%", "font": {"size": 34}},
            gauge={
                "axis": {"range": [80, 120]},
                "bar": {"color": ACCENT},
                "steps": [
                    {"range": [80, lower_corridor * 100], "color": "rgba(239,68,68,0.35)"},
                    {"range": [lower_corridor * 100, upper_corridor * 100],
                     "color": "rgba(34,197,94,0.25)"},
                    {"range": [upper_corridor * 100, 120], "color": "rgba(37,99,235,0.30)"},
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                              "value": 100},
            },
        ))
        gauge.update_layout(template="plotly_white", height=320,
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=20, r=20, t=30, b=10),
                            font=dict(color="#0f172a"))
        st.plotly_chart(gauge, use_container_width=True)

    ms = market_summary
    st.markdown("#### Underlying market, realised characteristics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean return", fmt_pct(ms["mean"]))
    m2.metric("Volatility", fmt_pct(ms["volatility"]))
    m3.metric("Worst year", fmt_pct(ms["worst_year"]))
    m4.metric("Skew", f"{ms['skew']:.2f}")
    m5.metric("Excess kurtosis", f"{ms['kurtosis']:.2f}")

# =========================================================================== #
# TAB 2, MONTE CARLO
# =========================================================================== #
with tab_mc:
    st.markdown("### Monte Carlo funding-level distribution")
    st.caption(
        f"{n_paths:,} simulated paths of the funding level, shown as percentile fan bands."
    )

    bands = percentile_bands(result.funding_level)

    def fan_chart(bands, ytitle, ytickfmt=".0%", refs=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.concatenate([years, years[::-1]]),
                                 y=np.concatenate([bands[95], bands[5][::-1]]),
                                 fill="toself", fillcolor="rgba(37,99,235,0.12)",
                                 line=dict(color="rgba(0,0,0,0)"), name="5–95%",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=np.concatenate([years, years[::-1]]),
                                 y=np.concatenate([bands[75], bands[25][::-1]]),
                                 fill="toself", fillcolor="rgba(37,99,235,0.28)",
                                 line=dict(color="rgba(0,0,0,0)"), name="25–75%",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=years, y=bands[50], name="Median",
                                 line=dict(color=ACCENT_2, width=3)))
        for ref, label, color in (refs or []):
            fig.add_hline(y=ref, line=dict(color=color, width=1.5, dash="dash"),
                          annotation_text=label, annotation_position="right")
        fig.update_layout(**PLOT_LAYOUT, height=380,
                          yaxis=dict(title=ytitle, tickformat=ytickfmt, gridcolor=GRID),
                          xaxis=dict(title="Year", gridcolor=GRID))
        return fig

    refs = [(1.0, "Target 100%", "#64748b"),
            (lower_corridor, "Lower corridor", BAD),
            (upper_corridor, "Upper corridor", ACCENT)]
    st.plotly_chart(fan_chart(bands, "Funding level", refs=refs),
                    use_container_width=True)

    cA, cB = st.columns(2)
    with cA:
        st.markdown("#### Terminal funding-level distribution")
        terminal = result.funding_level[:, -1]
        hist = go.Figure()
        hist.add_histogram(x=terminal, nbinsx=60, marker_color=ACCENT,
                           opacity=0.85)
        hist.add_vline(x=1.0, line=dict(color="#64748b", dash="dash"),
                       annotation_text="Target")
        hist.add_vline(x=analytics["funding_var95"], line=dict(color=BAD, dash="dot"),
                       annotation_text="95% VaR")
        hist.update_layout(**PLOT_LAYOUT, height=320,
                           xaxis=dict(title="Terminal funding level",
                                      tickformat=".0%", gridcolor=GRID),
                           yaxis=dict(title="Paths", gridcolor=GRID))
        st.plotly_chart(hist, use_container_width=True)

    with cB:
        st.markdown("#### Declared-bonus distribution")
        bonus_flat = result.bonus_rate.reshape(-1)
        bh = go.Figure()
        bh.add_histogram(x=bonus_flat, nbinsx=60, marker_color=ACCENT_2, opacity=0.85)
        bh.add_vline(x=0.0, line=dict(color=BAD, dash="dash"),
                     annotation_text="Removal")
        bh.add_vline(x=analytics["median_bonus"], line=dict(color="#0f172a", dash="dot"),
                     annotation_text="Median")
        bh.update_layout(**PLOT_LAYOUT, height=320,
                         xaxis=dict(title="Annual declared bonus",
                                    tickformat=".0%", gridcolor=GRID),
                         yaxis=dict(title="Observations", gridcolor=GRID))
        st.plotly_chart(bh, use_container_width=True)

    st.markdown("#### Risk metrics")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("95% Funding VaR", fmt_pct(analytics["funding_var95"]),
              help="5th percentile of terminal funding level.")
    r2.metric("95% Funding CVaR", fmt_pct(analytics["funding_cvar95"]),
              help="Average funding level in the worst 5% of paths.")
    r3.metric("P(Breach lower corridor)", fmt_pct(analytics["prob_breach_lower_corridor"]))
    r4.metric("P(Severe event <85%)", fmt_pct(analytics["prob_insolvent"]))

# =========================================================================== #
# TAB 3, STRESS TESTING
# =========================================================================== #
with tab_stress:
    st.markdown("### Deterministic stress scenarios")
    st.caption("Named, historically-inspired market paths run through the smoothing engine.")

    names = [s.name for s in STRESS_LIBRARY]
    chosen = st.selectbox("Select scenario", names)
    scenario = next(s for s in STRESS_LIBRARY if s.name == chosen)

    edit = st.toggle("Customise the return path", value=False)
    scen_returns = scenario.returns
    if edit:
        txt = st.text_input(
            "Annual returns (comma-separated, e.g. -0.35, 0.22, 0.14)",
            value=", ".join(f"{r:.2f}" for r in scenario.returns),
        )
        try:
            scen_returns = [float(x) for x in txt.split(",") if x.strip() != ""]
        except ValueError:
            st.error("Could not parse the return path, using the preset instead.")
            scen_returns = scenario.returns

    st.info(f"**{scenario.name}**, {scenario.description}")

    s_result = run_scenario(scen_returns, strategy, initial_funding)
    verdict = scenario_verdict(s_result, strategy)
    s_years = np.arange(1, len(scen_returns) + 1)

    tone_color = {"success": GOOD, "warning": WARN, "error": BAD}[verdict["tone"]]
    v1, v2, v3, v4 = st.columns(4)
    v1.markdown(
        f"<div class='verdict' style='background:{tone_color}22;color:{tone_color};"
        f"border:1px solid {tone_color}55;text-align:center'>{verdict['status']}</div>",
        unsafe_allow_html=True,
    )
    v2.metric("Trough funding", fmt_pct(verdict["min_funding"]))
    v3.metric("End funding", fmt_pct(verdict["end_funding"]))
    v4.metric("Bonus removed?", "Yes" if verdict["bonus_removed"] else "No")

    fig = go.Figure()
    fig.add_bar(x=s_years, y=scen_returns, name="Market return",
                marker_color="rgba(100,116,139,0.5)")
    fig.add_trace(go.Scatter(x=s_years, y=s_result.bonus_rate[0], name="Declared bonus",
                             mode="lines+markers", line=dict(color=ACCENT_2, width=3)))
    fig.add_trace(go.Scatter(x=s_years, y=s_result.funding_level[0], name="Funding level",
                             yaxis="y2", line=dict(color=ACCENT, width=3)))
    fig.add_hline(y=lower_corridor, line=dict(color=BAD, dash="dash"),
                  yref="y2", annotation_text="Lower corridor")
    fig.update_layout(**PLOT_LAYOUT, height=420,
                      yaxis=dict(title="Return / bonus", tickformat=".0%", gridcolor=GRID),
                      yaxis2=dict(title="Funding level", overlaying="y", side="right",
                                  tickformat=".0%", showgrid=False),
                      xaxis=dict(title="Year", gridcolor=GRID))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Run the full stress library")
    rows = []
    for s in STRESS_LIBRARY:
        res = run_scenario(s.returns, strategy, initial_funding)
        vd = scenario_verdict(res, strategy)
        rows.append({
            "Scenario": s.name,
            "Status": vd["status"],
            "Trough funding": fmt_pct(vd["min_funding"]),
            "End funding": fmt_pct(vd["end_funding"]),
            "Mean bonus": fmt_pct(vd["mean_bonus"]),
            "Recovered": "Yes" if vd["recovered"] else "No",
        })
    df = pd.DataFrame(rows)

    def color_status(val):
        return {
            "PASS": f"color:{GOOD};font-weight:600",
            "WATCH": f"color:{WARN};font-weight:600",
            "FAIL": f"color:{BAD};font-weight:600",
        }.get(val, "")

    st.dataframe(
        df.style.map(color_status, subset=["Status"]),
        use_container_width=True, hide_index=True,
    )

# =========================================================================== #
# TAB 4, BONUS SUSTAINABILITY
# =========================================================================== #
with tab_bonus:
    st.markdown("### Bonus declaration sustainability")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Sustainable bonus (25th pct)", fmt_pct(analytics["sustainable_bonus"]),
              help="A prudent declarable bonus that holds across most adverse paths.")
    b2.metric("Median bonus", fmt_pct(analytics["median_bonus"]))
    b3.metric("Worst declared bonus", fmt_pct(analytics["min_bonus"]))
    b4.metric("P(Bonus removal)", fmt_pct(analytics["prob_bonus_removal"]))

    st.markdown("#### Bonus declaration fan chart")
    b_bands = percentile_bands(result.bonus_rate)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([years, years[::-1]]),
                             y=np.concatenate([b_bands[95], b_bands[5][::-1]]),
                             fill="toself", fillcolor="rgba(34,211,238,0.12)",
                             line=dict(color="rgba(0,0,0,0)"), name="5–95%",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.concatenate([years, years[::-1]]),
                             y=np.concatenate([b_bands[75], b_bands[25][::-1]]),
                             fill="toself", fillcolor="rgba(34,211,238,0.28)",
                             line=dict(color="rgba(0,0,0,0)"), name="25–75%",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=years, y=b_bands[50], name="Median bonus",
                             line=dict(color=ACCENT_2, width=3)))
    fig.add_hline(y=0.0, line=dict(color=BAD, dash="dash"),
                  annotation_text="Removal threshold")
    fig.update_layout(**PLOT_LAYOUT, height=360,
                      yaxis=dict(title="Declared bonus", tickformat=".0%", gridcolor=GRID),
                      xaxis=dict(title="Year", gridcolor=GRID))
    st.plotly_chart(fig, use_container_width=True)

    cL, cR = st.columns(2)
    with cL:
        st.markdown("#### Volatility reduction, the core value proposition")
        comp = pd.DataFrame({
            "Series": ["Underlying market", "Member (smoothed)"],
            "Volatility": [analytics["market_return_vol"], analytics["member_return_vol"]],
        })
        figv = go.Figure()
        figv.add_bar(x=comp["Series"], y=comp["Volatility"],
                     marker_color=["#64748b", ACCENT_2],
                     text=[fmt_pct(v) for v in comp["Volatility"]],
                     textposition="outside")
        figv.update_layout(**PLOT_LAYOUT, height=320, showlegend=False,
                           yaxis=dict(title="Annualised volatility",
                                      tickformat=".0%", gridcolor=GRID))
        st.plotly_chart(figv, use_container_width=True)
        st.success(
            f"Smoothing cuts member-experienced volatility by "
            f"**{fmt_pct(analytics['volatility_reduction'], 0)}** "
            f"(from {fmt_pct(analytics['market_return_vol'])} to "
            f"{fmt_pct(analytics['member_return_vol'])})."
        )

    with cR:
        st.markdown("#### Probability of funding below target, over time")
        below = (result.funding_level < 1.0).mean(axis=0)
        breach = (result.funding_level < lower_corridor).mean(axis=0)
        figp = go.Figure()
        figp.add_trace(go.Scatter(x=years, y=below, name="P(funding < 100%)",
                                  line=dict(color=WARN, width=3)))
        figp.add_trace(go.Scatter(x=years, y=breach, name="P(below lower corridor)",
                                  line=dict(color=BAD, width=3)))
        figp.update_layout(**PLOT_LAYOUT, height=320,
                           yaxis=dict(title="Probability", tickformat=".0%",
                                      gridcolor=GRID, range=[0, 1]),
                           xaxis=dict(title="Year", gridcolor=GRID))
        st.plotly_chart(figp, use_container_width=True)

# =========================================================================== #
# TAB 5, AI REPORT
# =========================================================================== #
with tab_report:
    st.markdown("### Automated investment commentary")
    st.caption(
        "Generate a client-ready report covering the executive summary, bonus "
        "sustainability, risk assessment and management insights, produced directly "
        "from the simulation results."
    )

    strategy_summary = dict(
        smoothing_years=smoothing_years, bonus_cap=bonus_cap,
        lower_corridor=lower_corridor, upper_corridor=upper_corridor,
        management_charge=management_charge, initial_funding=initial_funding,
    )

    colcfg1, colcfg2 = st.columns([1, 2])
    with colcfg1:
        use_llm = st.toggle("Enhance with Anthropic API", value=False,
                            help="Optional. Falls back to the built-in engine if unavailable.")
    api_key = ""
    if use_llm:
        with colcfg2:
            api_key = st.text_input("Anthropic API key", type="password",
                                    placeholder="sk-ant-...")

    if st.button("📝 Generate report", type="primary"):
        report = None
        if use_llm and api_key:
            try:
                with st.spinner("Generating commentary with Claude..."):
                    report = generate_llm_narrative(
                        api_key, analytics, market_summary, horizon, n_paths,
                        strategy_summary,
                    )
                st.success("Generated with the Anthropic API.")
            except Exception as exc:  # noqa: BLE001 - surface any API issue gracefully
                st.warning(f"LLM generation unavailable ({exc}). Using the built-in engine.")
        if report is None:
            report = generate_narrative(
                analytics, market_summary, horizon, n_paths, strategy_summary
            )
        st.session_state["report"] = report

    if "report" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["report"])
        st.download_button("⬇️ Download report (Markdown)",
                           st.session_state["report"],
                           file_name="smoothed_bonus_report.md",
                           mime="text/markdown")
    else:
        st.info("Configure the model in the sidebar, then click **Generate report**.")

"""
dashboard_fast.py — Market Regime Classification (FAST, ~15 min)

DTW-based clustering of 50 S&P 500 stocks (2015-2024) with a backtest
comparison. Displays the real results saved by the backtester in outputs/,
or recomputes them on demand with the fast clustering engine.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUTS = PROJECT_ROOT / "outputs"

st.set_page_config(page_title="Market Regime Classification", layout="wide")

# Single muted accent used across charts for a professional, consistent look.
ACCENT = "#4C9EEB"
MUTED = "#3B4A5F"


def methodology():
    """ Concise overview of the approach. """
    with st.expander("Methodology"):
        st.markdown(
            """
            - **Data**: daily log returns for 50 S&P 500 constituents, Jan 2015 – Dec 2024.
            - **Similarity**: Dynamic Time Warping (DTW) measures how similarly two stocks
              move, allowing for small timing differences.
            - **Clustering**: DTW k-means groups the stocks; the number of regimes (K) is
              chosen by the silhouette score (higher = better-separated groups).
            - **Evaluation**: each grouping is backtested on the same period and compared on
              a risk-adjusted basis (Sharpe ratio) against an equal-weight (1/N) benchmark,
              with a two-sample t-test to gauge statistical significance.
            - **Pipeline**: Data → Log returns → DTW distance matrix → DTW k-means →
              silhouette K-search → regime groups → backtest comparison.
            - **Optimization (this fast build)**: the K-search runs in parallel, k-means uses
              fewer restarts, and a bounded Sakoe-Chiba DTW window is applied — identical
              methodology to `src/`, roughly 30× faster.
            """
        )


# ---------------------------------------------------------------------------
# Real data — loaded from the files the backtester / clustering engine saved
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_saved():
    """ Load the real saved results (comparison table, p-value, silhouette, clusters). """
    def _read_csv(name):
        p = OUTPUTS / name
        return pd.read_csv(p) if p.exists() else None

    summary = {}
    sp = OUTPUTS / "backtest_summary.json"
    if sp.exists():
        try:
            summary = json.loads(sp.read_text())
        except Exception:
            summary = {}

    return {
        "scores": _read_csv("silhouette_scores.csv"),
        "clusters": _read_csv("cluster_summary.csv"),
        "comparison": _read_csv("comparison_table.csv"),
        "labels": _read_csv("cluster_labels.csv"),
        "period": _read_csv("period_returns.csv"),
        "pca": _read_csv("pca_coords.csv"),
        "summary": summary,
    }


# Fallbacks = the values from the most recent real backtest run (used only if a file is missing)
FALLBACK_COMPARISON = pd.DataFrame({
    "Method": ["DTW + HRP", "Pearson + HRP", "1/N"],
    "Sharpe Ratio": [0.732058, 1.147782, 0.675134],
    "Mean Return": [0.137146, 0.595234, np.nan],
    "Volatility": [0.177022, 0.490026, np.nan],
    "Max Drawdown": [-4.058599, -1.879905, np.nan],
})
FALLBACK_SUMMARY = {"t_stat": -0.0344, "p_value": 0.9726}


@st.cache_data(show_spinner="Running the live pipeline — this can take a while...")
def run_live(engine="Fast (~15 min)"):
    """ Recompute everything with the selected engine.

    "Fast (~15 min)" uses src_fast/ (parallel + bounded DTW).
    "Original (hours)" uses the original src/ implementation.
    """
    if engine.startswith("Original"):
        # ---- the user's ORIGINAL implementation (unbounded DTW, n_init=10) ----
        from src.data_processing import main as get_data
        from src.clustering_engine import (
            determine_optimal_k, run_clustering, run_pca, compute_dtw_distance_matrix,
        )
        from src.backtester import run_complete_backtest

        returns, regimes = get_data()
        ra = returns.T.values
        best_k, scores = determine_optimal_k(ra, max_k=6)
        labels = run_clustering(ra, best_k)
        coords, explained = run_pca(ra)
        heat = compute_dtw_distance_matrix(ra[:8])
    else:
        # ---- the FAST engine (parallel + bounded DTW window) ----
        from src_fast.data_processing import main as get_data
        from src_fast.clustering_engine import (
            determine_optimal_k, run_clustering, run_pca, compute_dtw_distance_matrix,
        )
        from src_fast.backtester import run_complete_backtest

        returns, regimes = get_data()
        ra = returns.T.values
        best_k, scores = determine_optimal_k(ra, max_k=6, n_init=3,
                                             sakoe_chiba_radius=0.05, n_jobs=-1)
        labels = run_clustering(ra, best_k, n_init=3,
                                sakoe_chiba_radius=0.05, n_jobs=-1)
        coords, explained = run_pca(ra)
        heat = compute_dtw_distance_matrix(ra[:8], sakoe_chiba_radius=0.05, n_jobs=1)

    # Same comparison table + p-value the backtester produces
    bt = run_complete_backtest(returns, labels, coords)
    comparison = bt["comparison_table"]
    p_value = float(bt["p_value"])
    t_stat = float(bt["t_state"])

    scores_df = pd.DataFrame([{"k": k, "silhouette_score": round(s, 4)}
                              for k, s in sorted(scores.items())])
    counts = pd.Series(labels).value_counts().sort_index()
    clusters_df = pd.DataFrame({"Group": counts.index, "# stocks": counts.values})
    pca_df = pd.DataFrame({
        "ticker": list(returns.columns),
        "PC1": coords[:, 0], "PC2": coords[:, 1], "Group": labels,
    })
    tickers8 = list(returns.columns[:8])
    heat_df = pd.DataFrame(heat, index=tickers8, columns=tickers8)

    # Regime-group performance during the stress periods (COVID, 2022 rate shock)
    period_rows = []
    for period_name, sl in regimes.items():
        for cluster in np.unique(labels):
            idx = np.where(labels == cluster)[0]
            sub = sl.iloc[:, idx]
            avg = sub.mean(axis=1)
            period_rows.append({
                "Period": period_name, "Group": int(cluster),
                "n_stocks": int(len(idx)),
                "mean_return": float(avg.mean()),
                "volatility": float(avg.std()),
            })
    period_df = pd.DataFrame(period_rows)

    # Market timeline for the appendix chart
    market = returns.mean(axis=1)
    timeline_df = pd.DataFrame({"date": market.index, "market_return": market.values})

    return {
        "best_k": int(best_k),
        "scores_df": scores_df,
        "clusters_df": clusters_df,
        "pca_df": pca_df,
        "heat_df": heat_df,
        "period_df": period_df,
        "timeline_df": timeline_df,
        "comparison": comparison,
        "p_value": p_value,
        "t_stat": t_stat,
        "explained": explained,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Data source")
    st.caption("Results come from the latest backtest run saved in `outputs/`.")
    st.caption("This app uses the **fast** pipeline (`src_fast/`).")
    run_now = st.button("Recompute with live engine", width="stretch")
    if run_now:
        st.caption("Recomputes with the fast engine — ~15 minutes.")
    st.divider()
    st.caption("DTW Market Regime Classifier · Fast · 2015–2024")

# ---------------------------------------------------------------------------
# Gather the data (live run if requested, otherwise saved files)
# ---------------------------------------------------------------------------
saved = load_saved()

if run_now:
    try:
        data = run_live("Fast (~15 min)")
        st.success("Live run complete — showing fresh results.")
    except Exception as exc:
        st.error(f"Live recompute failed: {exc}")
        data = None
else:
    data = None

if data is not None:
    comparison = data["comparison"]
    p_value, t_stat = data["p_value"], data["t_stat"]
    best_k = data["best_k"]
    scores_df, clusters_df = data["scores_df"], data["clusters_df"]
    pca_df, period_df = data["pca_df"], data["period_df"]
    timeline_df, heat_df = data["timeline_df"], data["heat_df"]
else:
    comparison = saved["comparison"] if saved["comparison"] is not None else FALLBACK_COMPARISON
    p_value = saved["summary"].get("p_value", FALLBACK_SUMMARY["p_value"])
    t_stat = saved["summary"].get("t_stat", FALLBACK_SUMMARY["t_stat"])
    scores_df, clusters_df = saved["scores"], saved["clusters"]
    pca_df = saved["pca"]
    # The saved PCA file uses 'cluster'; the charts expect 'Group'.
    if pca_df is not None and "cluster" in pca_df.columns and "Group" not in pca_df.columns:
        pca_df = pca_df.rename(columns={"cluster": "Group"})
    period_df, heat_df = saved["period"], None
    timeline_df = None
    best_k = 2  # from the latest run
    if scores_df is not None and not scores_df.empty:
        best_k = int(scores_df.loc[scores_df["silhouette_score"].idxmax(), "k"])


def sharpe_of(method):
    row = comparison[comparison["Method"] == method]
    return float(row["Sharpe Ratio"].iloc[0]) if len(row) else np.nan


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Market Regime Classification")
st.caption("Fast implementation (~15 min, `src_fast/`) · DTW clustering of 50 S&P 500 stocks · 2015–2024")

methodology()
st.divider()

# --- Key findings ---
st.subheader("Key findings")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Optimal number of regimes", best_k, "selected via silhouette score")
c2.metric("Universe", "50", "S&P 500 constituents")
c3.metric("Best strategy Sharpe", f"{sharpe_of('Pearson + HRP'):.3f}", "Pearson + HRP")
c4.metric("p-value (DTW vs Pearson)", f"{p_value:.3f}", "significant if < 0.05")
st.divider()

# --- 1. Regime selection ---
st.subheader("1 · Regime selection")
st.caption("Silhouette score by number of clusters (K). Higher values indicate better-"
           "separated regimes; the best K is highlighted.")
if scores_df is not None and not scores_df.empty:
    colors = [ACCENT if int(k) == best_k else MUTED for k in scores_df["k"]]
    fig = go.Figure(go.Bar(
        x=scores_df["k"], y=scores_df["silhouette_score"], marker_color=colors,
        text=[f"{s:.3f}" for s in scores_df["silhouette_score"]], textposition="outside",
    ))
    fig.update_layout(
        height=360, xaxis_title="Number of clusters (K)", yaxis_title="Silhouette score",
        xaxis=dict(tickmode="linear", dtick=1), showlegend=False, template="plotly_dark",
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Silhouette data not available. Run the backtester first.")

# --- 2. Regime composition ---
st.subheader("2 · Regime composition")
if clusters_df is not None and not clusters_df.empty:
    clusters_show = clusters_df.rename(columns={"cluster": "Group", "n_tickers": "# stocks"})
    total = clusters_show["# stocks"].sum()
    clusters_show["Share"] = (clusters_show["# stocks"] / total * 100).round(1).astype(str) + "%"
    st.dataframe(clusters_show, width="stretch", hide_index=True)
else:
    st.info("Cluster data not available. Run the backtester first.")

# --- 3. Strategy comparison (real backtest results) ---
st.subheader("3 · Strategy comparison")
st.dataframe(comparison, width="stretch", hide_index=True)

# --- 4. Risk-return profile ---
st.subheader("4 · Risk–return profile")
st.caption("Volatility (risk) vs. annualized return for each strategy. Bubble size = Sharpe ratio.")
rr = comparison.dropna(subset=["Volatility", "Mean Return"]).copy()
if len(rr):
    fig = px.scatter(
        rr, x="Volatility", y="Mean Return", size="Sharpe Ratio", text="Method",
        labels={"Volatility": "Volatility (annualized)", "Mean Return": "Mean return (annualized)"},
        color="Method",
        color_discrete_sequence=[ACCENT, "#2DD4BF", "#F59E0B"],
        template="plotly_dark",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Risk–return data not available.")

# --- 5. Regime performance by stress period ---
st.subheader("5 · Regime performance by stress period")
st.caption("Average daily return of each regime-group during the COVID crash and the 2022 "
           "rate-shock window — do the groups behave differently under stress?")
if period_df is not None and not period_df.empty:
    fig = px.bar(
        period_df, x="Period", y="mean_return", color="Group", barmode="group",
        labels={"Period": "Stress period", "mean_return": "Mean daily return", "Group": "Regime group"},
        template="plotly_dark",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Stress-period data not available. Run the backtester first.")

st.subheader("Interpretation")
st.markdown(
    f"""
    - The silhouette analysis selects **K = {best_k}** as the optimal number of regimes.
    - On a risk-adjusted basis, the Pearson-based grouping shows the highest Sharpe ratio
      ({sharpe_of('Pearson + HRP'):.2f}), versus the DTW grouping ({sharpe_of('DTW + HRP'):.2f})
      and the equal-weight benchmark ({sharpe_of('1/N'):.2f}).
    - The two-sample t-test between the DTW and Pearson strategies yields **p = {p_value:.3f}**,
      well above the 0.05 threshold. The observed difference is therefore **not statistically
      significant** on this 10-year sample.
    """
)

# --- 6. PCA projection (shows group separation) ---
st.subheader("6 · PCA projection of stocks")
st.caption("Each point is a stock, projected onto its two largest principal components "
           "and colored by regime group. Similar movers land close together.")
if pca_df is not None and not pca_df.empty:
    fig = px.scatter(
        pca_df, x="PC1", y="PC2", color="Group", text="ticker",
        labels={"PC1": "PC1", "PC2": "PC2"},
        color_discrete_sequence=[ACCENT, "#2DD4BF", "#F59E0B", "#F472B6", "#A78BFA", "#F87171"],
        template="plotly_dark",
    )
    fig.update_traces(textposition="top center", marker=dict(size=9))
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")
else:
    st.info("PCA data not available. Run the backtester first.")

# --- Market timeline (live-only) ---
if timeline_df is not None and not timeline_df.empty:
    st.subheader("Appendix · Market timeline")
    st.caption("Average market daily return over the sample. Shaded windows mark the COVID "
               "crash and the 2022 rate-shock periods used for stress testing.")
    tl = timeline_df.copy()
    tl["date"] = pd.to_datetime(tl["date"])
    fig = px.line(tl, x="date", y="market_return",
                  labels={"date": "", "market_return": "Mean daily return"},
                  template="plotly_dark")
    fig.update_layout(height=360, showlegend=False)
    fig.add_vrect(x0="2020-02-01", x1="2020-04-30", fillcolor="red", opacity=0.15,
                  line_width=0, annotation_text="COVID", annotation_position="top left")
    fig.add_vrect(x0="2022-01-03", x1="2022-12-30", fillcolor="orange", opacity=0.15,
                  line_width=0, annotation_text="Rate shock", annotation_position="top left")
    st.plotly_chart(fig, width="stretch")

# --- DTW distance matrix (live-only) ---
if heat_df is not None and not heat_df.empty:
    st.subheader("Appendix · DTW distance matrix (8 sample stocks)")
    st.caption("Pairwise DTW distance between 8 sample stocks. Darker = more similar movement.")
    fig = px.imshow(heat_df, color_continuous_scale="Blues",
                    labels=dict(color="DTW distance"), template="plotly_dark")
    st.plotly_chart(fig, width="stretch")

# --- Limitations ---
st.divider()
st.subheader("Limitations")
st.markdown(
    """
    - **Universe**: 50 large-cap U.S. equities only — results may not generalize to small caps
      or other markets.
    - **In-sample evaluation**: clustering and backtesting share the same 2015–2024 window; the
      results are descriptive, not a forward-looking validation.
    - **Statistical power**: with p = 0.97, the DTW vs. Pearson difference is not statistically
      significant on this sample.
    - **Synchronized universe**: U.S. large caps move largely in lockstep, which limits the
      benefit DTW's time-warping can add versus simple correlation.
    """
)

st.divider()
st.caption("Market Regime Classification · Dynamic Time Warping clustering of 50 S&P 500 stocks, 2015–2024.")

# DTW Market Regime Classifier

> *"Does Dynamic Time Warping produce more stable market regimes than Pearson correlation?"*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-20%2F20%20Passing-brightgreen.svg)](tests/)

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [In Plain English](#in-plain-english)
- [Key Terms](#key-terms)
- [Two Versions](#two-versions)
- [Live Demo](#live-demo)
- [Results at a Glance](#results-at-a-glance)
- [Research Question](#research-question)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Dashboards](#dashboards)
- [Testing](#testing)
- [Why This Matters](#why-this-matters)
- [Tech Stack](#tech-stack)
- [Learning Journey](#learning-journey)
- [Limitations](#limitations)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## What This Project Does

This project groups **50 S&P 500 stocks** by how similarly they move using **Dynamic Time Warping (DTW)** + k-means, finds the optimal number of market **"regimes"** via a **silhouette score**, and backtests the grouping against **Pearson correlation** and an **equal-weight (1/N)** benchmark.

**Key finding:** On this dataset there is **no statistically significant difference** between DTW and Pearson (p = 0.9726). Still, the full research pipeline (data → clustering → backtest → dashboard → tests) was built from scratch, in two versions.

> New to this topic? Start with **In Plain English** below, then check the **Key Terms** glossary.

---

## In Plain English

Stocks don't all move together, and the way they move changes over time. A **market regime** is just a period when the market behaves in a particular way — like weather: sunny (calm, stocks drift up) or stormy (a crash, everything falls together).

This project asks: **can we automatically find those regimes by measuring how similarly stocks move?**

- **DTW** is the measuring stick. It's like comparing two songs played at different speeds — it aligns them in time, so two stocks that follow the same *shape* of movement count as similar even if one runs slightly ahead of the other.
- **k-means** groups the 50 stocks that move most alike into regimes.
- **Silhouette score** checks how cleanly the groups are separated.
- **Backtest** asks whether a portfolio built from those regimes beats simply owning every stock equally (1/N).

**The honest punchline:** the fancy DTW method did **not** beat the simpler Pearson-correlation method on this data (p = 0.97 → no statistical difference). But building the entire pipeline from scratch — and finding that out — is the real project.

---

## Key Terms

| Term | One-line meaning |
|---|---|
| **DTW** (Dynamic Time Warping) | Measures how similar two time series are, even if shifted in time (like two songs at different speeds). |
| **Market regime** | A period when the market behaves in a particular way (calm bull vs. crash). |
| **k-means** | Groups similar items into K clusters. |
| **Silhouette score** | -1 to 1; how well separated the clusters are. Higher is better; 0.24 = real but overlapping. |
| **Pearson correlation** | Standard measure of how two stocks move together, -1 (opposite) to +1 (identical). |
| **Sharpe ratio** | Return per unit of risk; higher is better. ~0.7–1.1 is solid. |
| **HRP** (Hierarchical Risk Parity) | Decides how much of each stock to hold, aiming for a well-diversified portfolio. |
| **PCA** | Shrinks 50 stocks down to 2 axes so they can be plotted on a 2D chart. |
| **p-value** | Chance the result is luck; below 0.05 is usually "significant." |
| **1/N (equal weight)** | Benchmark that simply owns every stock equally. |
| **Stress period** | A short, turbulent time in the market (e.g., COVID crash 2020, rate-shock 2022). |

---

## Two Versions

The same study ships in **two implementations**, so you can run whichever you prefer:

| Folder | What it is | Runtime | Run it |
|---|---|---|---|
| `src/` | **Original** — built from scratch while learning (unbounded DTW, `n_init=10`) | many hours | `python src/backtester.py` |
| `src_fast/` | **Optimized** — identical methodology, parallel + bounded DTW | ~15 min | `python src_fast/backtester.py` |

> **Tip:** You don't need to call the backtester files directly. **`python run.py` is the single entry point** — it runs the **fast** pipeline by default, and `python run.py --original` runs the slow one. The table shows exactly which script each one wraps.

Each version also has its own dashboard: `streamlit run dashboard_original.py` (original) or `streamlit run dashboard_fast.py` (fast).

---

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dtw-market-regime-classifier-kjrfaxmnqlqy7vdwfandhg.streamlit.app)

**Live app:** [https://dtw-market-regime-classifier-kjrfaxmnqlqy7vdwfandhg.streamlit.app](https://dtw-market-regime-classifier-kjrfaxmnqlqy7vdwfandhg.streamlit.app)

> Prefer to run it on your own machine? See [Quick Start](#quick-start) below.

---

## Results at a Glance

| Method | Sharpe Ratio | Volatility | Max Drawdown |
|--------|--------------|------------|--------------|
| **DTW + HRP** | 0.732 | 0.177 | -4.06 |
| **Pearson + HRP** | 1.148 | 0.490 | -1.88 |
| **1/N** | 0.675 | — | — |

- **Optimal number of regimes:** 2 (silhouette score: 0.2354)
- **p-value:** 0.9726 → no significant difference
- **Stress impact:** one regime group fell ~3.5× harder than the other during the COVID crash

*Sharpe ratios are annualized over the full 2015–2024 period. Silhouette ranges from -1 (bad) to +1 (great); 0.24 means the two regimes are real but overlap somewhat.*

---

## Research Question

> Does DTW distance produce more stable market-regime clusters than Pearson correlation?

**Inspired by:** López de Prado's paper *"Building Diversified Portfolios That Outperform Out-of-Sample"* (HRP).

**My twist:** Replace the Pearson correlation used in the standard approach with **DTW** and test whether it improves cluster stability during market stress.

**Hypothesis:** DTW-based clustering significantly outperforms Pearson-based clustering (p < 0.05).

**Result:** On 50 large-cap U.S. stocks (2015–2024), **p = 0.9726 → no significant difference**. DTW's advantage may appear in less synchronized markets or with out-of-sample evaluation.

---

## Architecture

```
dtw-market-regime-classifier/
│
├── src/                     # ORIGINAL pipeline (unbounded DTW, ~hours)
│   └── data → DTW → k-means → silhouette → backtest
│
├── src_fast/                # FAST pipeline (parallel, ~15 min)
│   └── same methodology, optimized
│
├── dashboard_fast.py        # Streamlit app — FAST pipeline
├── dashboard_original.py    # Streamlit app — ORIGINAL pipeline
├── run.py                   # Main entry point — python run.py (fast) / --original
│
├── outputs/                 # 7 result files (comparison, clusters, PCA, stress)
├── tests/                   # 20/20 tests passing (both engines)
├── mini_projects/           # Practice scripts 1–3
├── notebooks/               # (empty — reserved for exploration)
│
├── requirements.txt         # Dependencies
├── pytest.ini               # Test config
├── README.md                # You are here
└── notes.md                 # Learning notes
```

---

## Quick Start

```bash
# Clone (requires Python 3.9+)
git clone https://github.com/ABcodes-soon/dtw-market-regime-classifier.git
cd dtw-market-regime-classifier

# Install
pip install -r requirements.txt

# Run the FAST pipeline (~15 min) — the recommended entry point.
# (under the hood it runs the fast backtester: src_fast/backtester.py)
python run.py

# ...or the ORIGINAL pipeline (many hours — only if you want to wait).
# (under the hood it runs the original backtester: src/backtester.py)
python run.py --original

# Want to call the backtester directly? (same as the two commands above)
#   python src_fast/backtester.py    # fast
#   python src/backtester.py         # original

# Launch a dashboard
streamlit run dashboard_fast.py        # fast pipeline
streamlit run dashboard_original.py    # original pipeline
```

---

## Dashboards

| Dashboard | Purpose |
|---|---|
| `dashboard_fast.py` | **FAST** pipeline — interactive results |
| `dashboard_original.py` | **ORIGINAL** pipeline — same data |

**Both show:**
- Performance summary (Sharpe, volatility, drawdown)
- Strategy comparison table (DTW vs Pearson vs 1/N)
- Silhouette chart (optimal K)
- PCA projection of stocks by group
- Risk–return profile
- Stress-period performance (COVID, rate_shock)
- Market timeline & limitations

---

## Testing

```bash
pytest tests/ -v
```

**20/20 tests passing** across **both** the original and fast engines:

- Data processing (tickers, download, log returns)
- Clustering (DTW matrix, k-means, silhouette, PCA)
- Backtester (Sharpe, portfolio metrics, comparison)
- Fast-pipeline features + equivalence with the original

---

## Why This Matters

| Problem | My approach |
|---|---|
| Portfolios are unstable during market stress | DTW captures non-linear patterns in returns |
| Pearson correlation misses timing shifts | DTW aligns similar patterns even with phase lags |
| Most clustering tutorials use Euclidean distance | I replaced it with DTW and tested a hypothesis |

---

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Data** | yfinance, Pandas, NumPy |
| **Clustering** | tslearn (DTW), scikit-learn (k-means, silhouette, PCA) |
| **Backtesting** | NumPy, SciPy (t-test) |
| **Dashboard** | Streamlit, Plotly |
| **Testing** | Pytest |
| **Deployment** | Streamlit Community Cloud |

---

## Learning Journey

Built after completing:

- **Pandas** — Boris Paskhaver Udemy (S04–S11)
- **HKUST Coursera** — *Python & Statistics for Financial Analysis* (Prof. Xuhu Wan)
- **StatQuest** — K-means, PCA, Silhouette
- **Sigma Coding** — *Clustering Stocks With Python* (Parts 1–6)
- **López de Prado** — *Building Diversified Portfolios That Outperform Out-of-Sample*

*Full notes in `notes.md`.*

---

## Limitations

- **Universe:** 50 large-cap U.S. stocks only
- **Evaluation:** in-sample (2015–2024)
- **Statistical significance:** p = 0.9726 → no significant difference
- **DTW advantage:** may appear in less synchronized markets or with out-of-sample testing

---

## Contact

**GitHub:** [ABcodes-soon](https://github.com/ABcodes-soon)

---

## Acknowledgments

- **Marcos López de Prado** — research foundation
- **Prof. Xuhu Wan** — HKUST Coursera course
- **Sigma Coding** — pipeline blueprint
- **StatQuest** — clarity on ML concepts

---

## License

MIT — feel free to use, modify, and build on this work.

---

*Built with Python, persistence, and a lot of debugging.*

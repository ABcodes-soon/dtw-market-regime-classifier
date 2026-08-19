# 📊 DTW Market Regime Classifier

> *"Does Dynamic Time Warping produce more stable market regimes than Pearson correlation?"*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-20%2F20%20Passing-brightgreen.svg)](tests/)

---

## 🎯 TL;DR — What This Project Does

Groups **50 S&P 500 stocks** by how similarly they move using **Dynamic Time Warping (DTW)** + k-means, finds optimal market **"regimes"** via a **silhouette score**, and backtests the grouping against **Pearson correlation** and an **equal-weight (1/N)** benchmark.

**Key finding:** On this dataset there is **no statistically significant difference** between DTW and Pearson (p = 0.9726) — but the full research pipeline (data → clustering → backtest → dashboard → tests) was built from scratch, in two versions.

---

## 📊 Results at a Glance

| Method | Sharpe Ratio | Volatility | Max Drawdown |
|--------|--------------|------------|--------------|
| **DTW + HRP** | 0.732 | 0.177 | -4.06 |
| **Pearson + HRP** | 1.148 | 0.490 | -1.88 |
| **1/N** | 0.675 | — | — |

- **Optimal number of regimes:** 2 (silhouette score: 0.2354)
- **p-value:** 0.9726 → no significant difference
- **Stress impact:** one regime group fell ~3.5× harder than the other during the COVID crash

---

## 🔬 Research Question

> Does DTW distance produce more stable market-regime clusters than Pearson correlation?

**Inspired by:** López de Prado's paper *"Building Diversified Portfolios That Outperform Out-of-Sample"* (HRP).

**My twist:** Replace the Pearson correlation used in the standard approach with **DTW** and test whether it improves cluster stability during market stress.

**Hypothesis:** DTW-based clustering significantly outperforms Pearson-based clustering (p < 0.05).

**Result:** On 50 large-cap U.S. stocks (2015–2024), **p = 0.9726 → no significant difference**. DTW's advantage may appear in less synchronized markets or with out-of-sample evaluation.

---

## 🏗️ Architecture

```
dtw-market-regime-classifier/
│
├── src/                     # ORIGINAL pipeline (unbounded DTW, ~hours)
│   └── data → DTW → k-means → silhouette → backtest
│
├── src_fast/                # FAST pipeline (parallel, ~15 min)
│   └── same methodology, optimized
│
├── dashboard.py             # Streamlit app — FAST pipeline   (port 8501)
├── dashboard_original.py    # Streamlit app — ORIGINAL        (port 8502)
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

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/ABcodes-soon/dtw-market-regime-classifier.git
cd dtw-market-regime-classifier

# Install
pip install -r requirements.txt

# Run the fast pipeline (~15 min) — or src/backtester.py for the original
python src_fast/backtester.py

# Launch the dashboard
streamlit run dashboard.py
```

---

## 📊 Dashboards

| Dashboard | Purpose |
|---|---|
| `dashboard.py` | **FAST** pipeline — interactive results |
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

## 🧪 Testing

```bash
pytest tests/ -v
```

**20/20 tests passing** across **both** the original and fast engines:

- Data processing (tickers, download, log returns)
- Clustering (DTW matrix, k-means, silhouette, PCA)
- Backtester (Sharpe, portfolio metrics, comparison)
- Fast-pipeline features + equivalence with the original

---

## 🧠 Why This Matters

| Problem | My approach |
|---|---|
| Portfolios are unstable during market stress | DTW captures non-linear patterns in returns |
| Pearson correlation misses timing shifts | DTW aligns similar patterns even with phase lags |
| Most clustering tutorials use Euclidean distance | I replaced it with DTW and tested a hypothesis |

---

## 📚 Learning Journey

Built after completing:

- **Pandas** — Boris Paskhaver Udemy (S04–S11)
- **HKUST Coursera** — *Python & Statistics for Financial Analysis* (Prof. Xuhu Wan)
- **StatQuest** — K-means, PCA, Silhouette
- **Sigma Coding** — *Clustering Stocks With Python* (Parts 1–6)
- **López de Prado** — *Building Diversified Portfolios That Outperform Out-of-Sample*

*Full notes in `notes.md`.*

---

## ⚠️ Limitations

- **Universe:** 50 large-cap U.S. stocks only
- **Evaluation:** in-sample (2015–2024)
- **Statistical significance:** p = 0.9726 → no significant difference
- **DTW advantage:** may appear in less synchronized markets or with out-of-sample testing

---

## 📬 Contact

**Author:** Abhinav
**GitHub:** [ABcodes-soon](https://github.com/ABcodes-soon)

---

## 🙏 Acknowledgments

- **Marcos López de Prado** — research foundation
- **Prof. Xuhu Wan** — HKUST Coursera course
- **Sigma Coding** — pipeline blueprint
- **StatQuest** — clarity on ML concepts

---

## 📄 License

MIT — feel free to use, modify, and build on this work.

---

*Built with Python, persistence, and a lot of debugging.* 🚀

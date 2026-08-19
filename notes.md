
**One Sentence Summary:** Recursive bisection is a top-down process that repeatedly splits a portfolio into two parts until each part is a single stock.

---

### Q24: Why are these steps important?

**My Original Answer:** Quasi-diagonalization organizes the tree, finds similar stocks, groups them by similarity, and finds the investments. Recursive bisection breaks each group of clustered stocks down to each stock.

**Correct Answer:** Quasi-diagonalization reorganizes the covariance matrix so similar stocks are grouped together, making clusters visible. Recursive bisection then walks down the tree, splitting clusters until reaching individual stocks. Together, they turn the dendrogram into actual portfolio weights — without needing to invert a covariance matrix.

**One Sentence Summary:** These steps turn the dendrogram into actual portfolio weights by organizing similar stocks together and recursively splitting clusters.

---

## 7. Monte Carlo Results

> **Stage goal:** Empirically compare HRP against traditional methods out-of-sample.

### Q25: What methods does he compare in the results table?

**Answer:** He compares three methods:

| Method | What it is |
|--------|-----------|
| **HRP** | Hierarchical Risk Parity — his new method |
| **CLA** | Critical Line Algorithm — Markowitz's traditional method |
| **IVP** | Inverse Variance Portfolio — traditional risk parity |

**Results:**
| Method | Out-of-sample variance |
|--------|------------------------|
| CLA | 0.1157 (highest) |
| IVP | 0.0928 (middle) |
| **HRP** | **0.0671 (lowest — best!)** |

> 💡 **Key insight:** CLA looks good on paper but fails in real life. HRP works better out-of-sample.

**Project Connection:** I'm testing a similar comparison: DTW-based clusters vs Pearson-based clusters vs 1/N baseline.

---

### Q26: What performance metrics does he use?

**Answer:** He uses:
1. **Variance (σ²)** — measures risk (how much returns fluctuate)
2. **Sharpe Ratio** — measures risk-adjusted return

**Results:**
| Method | Variance |
|--------|----------|
| CLA | 0.1157 (highest risk) |
| IVP | 0.0928 (middle risk) |
| **HRP** | **0.0671 (lowest risk — best!)** |

> 💡 **Key insight:** All methods have similar returns. HRP wins on risk, not return. HRP improves Sharpe ratio by 31.3% over CLA.

**Project Connection:** I'll use the same metrics — variance and Sharpe ratio — to compare DTW vs Pearson.

---

### Q27: How does HRP compare to Markowitz's method?

**My Answer:** HRP significantly outperforms CLA out-of-sample: 72.47% lower variance and +31.3% improvement in Sharpe ratio. CLA tries to minimize variance but fails out-of-sample (Markowitz's curse). HRP uses a tree structure to create more stable portfolios.

**Simplified:** Variance: HRP has ~72% less risk than CLA. Sharpe Ratio: HRP is ~31% more efficient than CLA.

> 💡 **Key insight:** CLA fails in the real world. HRP works in the real world.

---

### Q28: Copy the table structure you'll use for your project

| Method | Sharpe Ratio | Max Drawdown | Volatility |
|--------|--------------|--------------|------------|
| **DTW + HRP** | [Your result] | [Your result] | [Your result] |
| **Pearson + HRP** | [Your result] | [Your result] | [Your result] |
| **1/N** | [Your result] | [Your result] | [Your result] |

**What Each Column Means:**
- **Sharpe Ratio:** return per unit of risk — higher is better
- **Max Drawdown:** largest peak-to-trough decline — lower is better
- **Volatility:** how much returns fluctuate — lower is better

**What Each Method Represents:**
- **DTW + HRP:** your method (hypothesis)
- **Pearson + HRP:** baseline (comparison)
- **1/N:** benchmark (must beat)

---

## 8. Conclusions

### Q29: What are the 3 key takeaways from this paper?

**1. HRP replaces covariance with a tree structure** — More stable than traditional methods, fully utilizes covariance information

**2. HRP does not require covariance matrix inversion** — Works even during market stress when matrices break down, impossible for traditional methods

**3. HRP delivers lower out-of-sample variance** — Beats CLA and IVP in real-world performance, even though CLA is designed to minimize variance!

**One Sentence Summary:** HRP replaces unstable covariance matrices with a tree structure, avoiding matrix inversion and delivering better real-world performance.

---

### Q30: How does HRP overcome Markowitz's curse?

**My Answer:** HRP uses trees and breaks down each cluster to find how to organize, while Markowitz breaks it down but goes by half and goes to each stock. However, Markowitz is unstable, so HRP's stability is a strong benefit during market stress.

**Correct Answer:** HRP overcomes Markowitz's curse by replacing the unstable covariance matrix with a tree structure. Traditional methods try to mathematically find the optimal portfolio but fail during stress because covariance matrices break down. HRP uses a dendrogram to cluster similar stocks first, then allocates weights by walking down the tree. This does not require matrix inversion, so it remains stable even during market stress.

**One Sentence Summary:** HRP uses a tree structure instead of a covariance matrix, making it stable during market stress.

---

### Q31: Why is HRP better for larger investment universes?

**My Answer:** Hierarchical risk parity is better because Monte Carlo experiments show HRP delivers lower out-of-sample variance than CLA or traditional risk parity methods (IVP).

**Correct Answer:** HRP's out-of-sample outperformance becomes even more substantial for larger investment universes. Traditional methods suffer from estimation errors that grow with more assets — the more stocks you have, the more correlations you need to estimate. HRP avoids this by using a tree structure instead of relying on correlation estimates. So the bigger the universe, the bigger HRP's advantage.

**One Sentence Summary:** HRP's advantage grows with more assets because it avoids the estimation errors that plague traditional methods.

**Project Connection:** My 150-stock universe is exactly where this advantage should show up — if DTW clustering stays stable at scale.

---

## ⚡ Quick Reference — HRP, CLA, and IVP

| Method | What It Does | Why It Matters |
|--------|--------------|----------------|
| **HRP** | Uses tree clustering + recursive bisection | Best out-of-sample performance |
| **CLA** | Markowitz's method — tries to minimize variance mathematically | Fails out-of-sample (Markowitz's curse) |
| **IVP** | Allocates weights inversely to variance | Ignores correlations between stocks |

**Project Connection:** HRP = my DTW method, CLA = Pearson baseline, IVP = 1/N benchmark

---

## 🧠 Key Takeaways

| Concept | Summary |
|---------|---------|
| **HRP** | A portfolio construction method that uses tree structures instead of complete graphs |
| **Markowitz's Curse** | More diversification = more instability due to estimation errors |
| **Complete Graph** | Every stock connected to every other stock (messy, unstable) |
| **Tree Graph** | Stocks connected only to nearest neighbors (organized, stable) |
| **Market Stress** | Correlation matrices break down, but trees remain stable |
| **DTW Connection** | Replaces Pearson correlation to create more stable clusters |
| **Distance Formula** | $d_{i,j} = \sqrt{0.5 \times (1 - \rho_{i,j})}$ → my version: $d_{i,j} = DTW(X_i, X_j)$ |
| **Quasi-Diagonalization** | Reorders covariance so similar stocks form blocks along the diagonal |
| **Recursive Bisection** | Top-down splitting of clusters into weights — no matrix inversion |
| **HRP vs CLA vs IVP** | HRP lowest out-of-sample variance (0.0671); CLA highest (0.1157) |
| **HRP Wins On** | Risk, not return — ~72% less variance, +31.3% Sharpe vs CLA |

---

## 🗺️ Study Map — From Question to Concept

| Questions | Concept | Your Project Role |
|-----------|---------|-------------------|
| Q1–Q4 | What HRP solves | Why you're building this |
| Q9–Q12 | Markowitz's curse | The problem DTW might fix |
| Q13–Q16 | Tree vs complete graph | Why you build a dendrogram |
| Q17–Q21 | Tree clustering (distance → dendrogram) | Swap Pearson → DTW here |
| Q22–Q24 | Quasi-diagonalization + recursive bisection | Turn clusters into weights |
| Q25–Q28 | Monte Carlo results | Your comparison table (Sharpe / MaxDD / Vol) |
| Q29–Q31 | Conclusions | What you're trying to prove |

---

# 📊 Project Findings — DTW vs Pearson

### Research Hypothesis (from LdP Paper)

> "We test H₀ that DTW and Pearson clusters produce equal out-of-sample Sharpe ratios. We reject H₀ at p < 0.05, showing DTW-based clustering significantly outperforms Pearson correlation-based clustering for Hierarchical Risk Parity portfolios."

### What I Found

| Method | Sharpe Ratio | Mean Return | Volatility | Max Drawdown |
|--------|--------------|-------------|------------|--------------|
| DTW + HRP | 0.732 | 0.137 | 0.177 | -4.06% |
| Pearson + HRP | 1.148 | 0.595 | 0.490 | -1.88% |
| 1/N | 0.675 | — | — | — |

**Key Result:**
- **p-value:** 0.9726 → **No statistically significant difference**
- **Optimal K:** 2 (silhouette score: 0.2354)

### Why DTW Didn't Beat Pearson (In This Dataset)

1. **Equities are highly synchronized.** DTW's edge is in matching time-warped shapes — useful when series move similarly but with phase lags. Large-cap U.S. stocks move in lockstep day-to-day, so Pearson already captures nearly all the structure.

2. **The gap is not statistically meaningful.** With p = 0.97, the Sharpe difference (0.732 vs 1.148) is pure noise on a single 10-year window — not evidence that Pearson is genuinely better.

3. **Coarse clustering.** With K=2, the market is split into just two big groups. DTW's advantage tends to show up when regimes are distinct and numerous (crises vs. calm), which 2 clusters can't really capture.

4. **In-sample evaluation.** The backtest was performed on the same data used to train the clusters (2015-2024). This introduces lookahead bias and inflates performance metrics.

### What This Means

**The hypothesis (why I expected DTW to win):**
DTW measures how similar two return paths are, even if one is shifted or stretched in time. The theory was: tickers that "behave alike through a crisis, but with slightly different timing" would cluster together under DTW but not under Pearson — giving better regime separation → better diversification → higher risk-adjusted return.

**Why it didn't show up:**
U.S. large caps are too synchronized. There's little phase lag for DTW's warping to exploit, so it mostly aligns noise while Pearson grabs the real signal. DTW's advantage is biggest in markets with genuine lead-lag structure (e.g., bonds, commodities, international equities).

**One window, no significance:**
A single 2015-2024 backtest gives p = 0.97 — the difference is statistically indistinguishable from a coin flip. It says nothing reliable about which method is better.

### What Would Actually Make DTW More Likely to Win

1. **Out-of-sample / rolling evaluation:** Train clusters on 2015-2021, test on 2022-2024 — this is where DTW's edge, if real, becomes detectable.

2. **Evaluate on stress slices:** COVID and rate_shock periods — DTW's regime-separation value is most visible during crises, not on a 10-year average.

3. **More tickers / finer K:** Test up to K=5-6 (though silhouette data showed K=2 was genuinely best here).

4. **Less synchronized assets:** Test on bonds, commodities, or international equities where lead-lag relationships exist.

### Conclusion

The result I got isn't a failure — it's the honest answer from this dataset. DTW and Pearson produce similar clusters for these highly-correlated large-caps, and a single noisy window can't tell them apart.

**The hypothesis that DTW would outperform Pearson on this dataset is not supported. A null result is still a valid research finding.**

---

## 🧠 Key Takeaways

| Concept | Summary |
|---------|---------|
| **DTW vs Pearson** | No statistically significant difference (p = 0.9726) |
| **Optimal K** | 2 clusters (silhouette = 0.2354) |
| **Why DTW didn't win** | Large-cap U.S. stocks are too synchronized; little phase lag for DTW to exploit |
| **What would help** | Out-of-sample evaluation, stress-period testing, less synchronized assets |
| **The real value** | I built an end-to-end research pipeline from scratch |

---

## 📚 What I Learned from This Project

1. **DTW clustering on financial time series** — I can apply DTW to stocks and evaluate clusters.

2. **Silhouette score for optimal K** — I can find the best number of clusters automatically.

3. **Sharpe ratio and backtesting** — I can measure risk-adjusted performance.

4. **In-sample vs out-of-sample** — I understand why honest evaluation requires testing on unseen data.

5. **p-value interpretation** — I know that p > 0.05 means "no significant difference," not "failure."

6. **Research pipeline design** — I built a complete system from data → clustering → backtesting → dashboard.

7. **Connecting theory to code** — I read LdP's paper and implemented a variant of it.

8. **Handling null results** — I learned that a null result is still a valid contribution.
import yfinance as yf
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from sklearn.metrics import silhouette_score


# Use yfinance to download historical stock data for a list of tickers
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"]

print("Downloading historical stock data...")

prices_data = yf.download(tickers, start = "2015-01-01", end = "2024-12-31", auto_adjust = True)

current_prices = prices_data["Close"].dropna(axis = 1, how = 'all')

#### Part 2:

returns = np.log(current_prices / current_prices.shift(1)).dropna()
print(f"Return shape: {returns.shape}")
# result sin 2514 rows and 10 columns, which means we have 2514 daily returns for each of the 10 tickers.


# Prepare for tslearn: (nlearns, n_timestamp)
returns_array = returns.T.values  # Transpose to have shape (n_tickers, n_days)
# tslearn expects: (n_series, n_timestamps) — each row is a ticker, each column is a day.
print(f"Array shape for tslearn: {returns_array.shape}")

# DTW Matrix + silhouette loop


#Term	What It Means in Plain English
# cdist_dtw	Cross-Distance with Dynamic Time Warping — computes the distance between every pair of time series
# returns_array	Your stock returns: shape is (10 tickers, 2265 days)
# dtw_dist	A 10×10 matrix where each cell is the DTW distance between two stocks

dtw_dist = cdist_dtw(returns_array)

scores = {}
# its a dictionary to store the silhouette scores for different values of k (number of clusters). The keys are the values of k, and the values are the corresponding silhouette scores.
for k in range(2,6):

    model = TimeSeriesKmeans(n_clusters = k, metric = "dtw", n_init = 10, random_state = 42)

    #5B fit and predict and assigns each stock to a cluster
    labels = model.fit_predict(returns_array)

    # the silhouette score is a measure of how similar an object is to its own cluster compared to other clusters. It ranges from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.
    #  It measures how well each stock fits into its assigned cluster compared to other clusters.
    # the precomputed metric means that we are using the DTW distance matrix we computed earlier instead of calculating distances again.
    score = silhouette_score(dtw_dist, labels, metric = "precomputed")

    # for each value of k, we store the silhouette score in the scores dictionary. The key is the value of k, and the value is the corresponding silhouette score.
    scores[k] = score


    print(f"Silhouette score for k={k}: {score:.4f}")

# returns the dictionary key that has the highest associated value
#4a best K for silhouette score
best_k = max(scores, key = scores.get)
print(f"\n🌟 Optimal K: {best_k} (silhouette={scores[best_k]:.4f})")

model = TimeSeriesKmeans(n_clusters = k, metric = "dtw", n_init = 10, random_state = 42)
final_labels = model.fit_predict(returns_array)


print("\nFinal cluster assignments:")
for ticker, label in zip(tickers, final_labels):
    # so it will check returns the ticker and its corresponding cluster label, and then prints them in a formatted string.
    print(f"{ticker}: Cluster {label}")


tech_clusters = set([label for ticker, label in zip(tickers, final_labels) if ticker in ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]])

# energy sectors
energy_clusters = set([label for ticker, label in zip(tickers, final_labels) if ticker in ["XOM", "CVX", "COP", "SLB", "BP"]])



# tech and energy sectors
print("\nTech sector clusters:", tech_clusters)
print("Energy sector clusters:", energy_clusters)

# quick check
if tech_clusters.intersection(energy_clusters):
    print("\n⚠️ Warning: Tech and Energy sectors share clusters!")
elif not tech_clusters.intersection(energy_clusters):
    print("\n✅ Tech and Energy sectors are in separate clusters.")
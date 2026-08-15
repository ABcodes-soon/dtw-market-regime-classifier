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




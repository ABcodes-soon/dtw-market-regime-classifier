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






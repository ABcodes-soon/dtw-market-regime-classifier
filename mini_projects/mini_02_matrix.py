# 5 ticker returns matrix

import yfinance as yf
import numpy as np
import pandas as pd

# Assign the tickers to a list
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

current_data = yf.download(tickers, start="2020-01-01", end="2023-01-01", auto_adjust=True)

prices = current_data["Close"]

# so this will give us a matrix of returns for all 5 tickers and find the log returns for each ticker
returns = np.log(prices / prices.shift(1)).dropna()

# Display
print(f"COVID period 2020-01-01 to 2023-01-01 for {tickers}")
print(returns.head())

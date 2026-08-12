# Mini project 1: AAPL returns during the COVID period

import yfinance as yf
import numpy as np

ticker = "AAPL"

# NOTE (yfinance 1.x): Yahoo removed "Adj Close" from its API.
# With auto_adjust=True (the default), "Close" is already split/dividend-adjusted.
# Columns are a MultiIndex (Price, Ticker) even for a single ticker.
df = yf.download(ticker, start="2020-01-01", end="2023-01-01", auto_adjust=True)
prices = df["Close"][ticker]

# Log returns: ln(P_t / P_{t-1}); shift(1) moves prices back 1 day so we can
# compute each day's return, then dropna() removes the first NaN row.
returns = np.log(prices / prices.shift(1)).dropna()
returns = returns[returns != 0]  # remove any zero returns (e.g., weekends, holidays)

print(f"COVID period 2020-01-01 to 2023-01-01 for {ticker}")
print(f"Mean log return: {returns.mean():.6f}")

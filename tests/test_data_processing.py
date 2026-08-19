"""Tests for the data pipeline — run against BOTH the original (src/) and the
fast (src_fast/) implementations via the `data_processing` fixture."""

import pandas as pd
import numpy as np


def test_get_tickers(data_processing):
    """The S&P 500 ticker list is non-empty and is a list."""
    tickers = data_processing.get_sp500_tickers()
    assert len(tickers) > 0
    assert isinstance(tickers, list)


def test_download_data(data_processing):
    """Historical data downloads correctly (needs internet)."""
    tickers = ["AAPL", "MSFT"]
    prices = data_processing.download_stock_data(tickers, "2020-01-01", "2020-12-31")
    assert prices.shape[1] == 2  # 2 tickers
    assert prices.shape[0] > 0   # has data


def test_calculate_returns(data_processing):
    """Log returns are computed correctly (first row dropped, no NaN)."""
    prices = pd.DataFrame({"AAPL": [100, 105, 102], "MSFT": [200, 210, 205]})
    returns = data_processing.calculate_returns(prices)
    assert returns.shape[0] == 2  # drops first row
    assert not returns.isnull().any().any()  # no NaN values
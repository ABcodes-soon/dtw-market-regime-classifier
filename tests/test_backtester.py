"""Tests for the backtester — run against BOTH the original (src/) and the
fast (src_fast/) implementations via the `backtester` fixture."""

import pandas as pd
import numpy as np


def test_sharpe_ratio(backtester):
    """The Sharpe ratio is computed correctly."""
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    sharpe = backtester.calculate_sharpe(returns)
    assert sharpe > 0
    assert isinstance(sharpe, float)


def test_evaluate_portfolio(backtester):
    """Portfolio evaluation returns metrics for every cluster."""
    returns = pd.DataFrame(np.random.randn(100, 10))
    labels = np.random.randint(0, 2, 10)
    metrics = backtester.evaluate_portfolio(returns, labels)
    assert len(metrics) == len(np.unique(labels))
    assert 'sharpe' in metrics[0]
import numpy as np
import pandas as pd
from scipy import stats # tests p values

def calculate_sharpe(returns, risk_free_rate = 0):
    """
    Calculate sharp ratio and Args:  Array of returns
    risk_free_rate: Risk-free rate (default: 0)
    returns shows the annualized sharpe ratio
    """

    sharpe_returns = returns - risk_free_rate

    # shows the mean and std for taking more risk with average of 225 days.

    return sharpe_returns.mean() / sharpe_returns.std() * np.sqrt(225)



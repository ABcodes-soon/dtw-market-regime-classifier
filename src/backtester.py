import numpy as np
import pandas as pd
from scipy import stats  # used for the t-test p-value


def calculate_sharpe(returns, risk_free_rate=0):
    """
    Compute the (annualized) Sharpe ratio of a return series.

    The Sharpe ratio measures how much return you get per unit of risk taken.
    Higher = better risk-adjusted performance.

    Args:
        returns:         Array/Series of daily returns
        risk_free_rate:  Risk-free return to subtract (default: 0)

    Returns:
        Annualized Sharpe ratio
    """
    # Step 1: "Excess returns" = what the strategy earned above the risk-free rate
    excess_returns = returns - risk_free_rate

    # Step 2: Sharpe = mean return / volatility (std), then scale up to a year.

    # sqrt(225) annualizes daily figures using ~225 trading days per year.

    return excess_returns.mean() / excess_returns.std() * np.sqrt(225)

def evaluate_portfolio(returns, labels):
    """
    Compute risk/return metrics for each cluster of tickers.

    Args:
        returns:  DataFrame of daily returns (rows = days, columns = tickers)
        labels:   Array of cluster IDs, one per ticker (same order as the columns)

    Returns:
        Dictionary: { cluster_id: {sharpe, mean_return, volatility, max_drawdown, n_clusters} }
    """
    metrics = {}  # cluster_id -> metrics dict

    for cluster in np.unique(labels):
        # 1) Which tickers (column positions) belong to this cluster?
        ticker_idx = np.where(labels == cluster)[0]

        # 2) Slice the returns table down to just those tickers (keep all days)
        cluster_returns = returns.iloc[:, ticker_idx]

        # 3) Average the tickers together each day -> one daily return series for the cluster
        cluster_avg = cluster_returns.mean(axis=1)

        metrics[cluster] = {
            'sharpe':       calculate_sharpe(cluster_avg),
            'mean_return':  cluster_avg.mean() * 252,         # annualized return
            'volatility':   cluster_avg.std() * np.sqrt(252), # annualized risk
            'max_drawdown': (cluster_avg / cluster_avg.cummax() - 1).min(),
                            # worst peak-to-trough drop in the cluster's value
            'n_clusters':   len(ticker_idx),  # this is really the # of TICKERS in the cluster
        }

    return metrics

# Compare the two clustering/weighting methods statistically

def compare_dtw_vs_pearson(dtw_returns, pearson_returns):
    """
    Run an independent t-test to see if the DTW strategy's returns
    are statistically different from the Pearson strategy's returns.

    Args:
        dtw_returns:      Daily returns from the DTW-based strategy
        pearson_returns:  Daily returns from the Pearson-based strategy

    Returns:
        (t_stat, p_value)
          - t_stat:  how far apart the two means are (relative to variance)
          - p_value: probability that the difference is just random chance
                     (small p, usually < 0.05, means a real difference)
    """
    t_stat, p_value = stats.ttest_ind(dtw_returns, pearson_returns)

    return t_stat, p_value


# 1/N benchmark: give every stock an equal weight (1/N each)

def calculate_one_in_sharp(returns):
    """
    Sharpe ratio of the equal-weight (1/N) portfolio — our baseline/benchmark.

    Args:
        returns:  DataFrame of daily returns (days x tickers)

    Returns:
        Annualized Sharpe ratio of the equal-weight portfolio
    """
    # Average all tickers together each day -> the 1/N portfolio's daily returns
    equal_weights_returns = returns.mean(axis=1)

    
    # Reuse the same Sharpe function so all strategies are compared on equal footing
    return calculate_sharpe(equal_weights_returns)




# Build a summary table comparing all three strategies

def build_comparison_table(dtw_metrics, pearson_metrics, one_n_shape):
    """
    Combine the results of all methods into one comparison table.

    Args:
        dtw_metrics:      metrics dict from evaluate_portfolio for the DTW method
        pearson_metrics:  metrics dict from evaluate_portfolio for the Pearson method
        one_n_shape:      Sharpe ratio of the 1/N benchmark (from calculate_one_in_sharp)

    Returns:
        DataFrame with one row per method: DTW+HRP, Pearson+HRP, 1/N
    """
    # Pick the best-performing cluster for each method (highest Sharpe ratio)
    dtw_best = max(dtw_metrics, key=lambda c: dtw_metrics[c]['sharpe'])

    pearson_best = max(pearson_metrics, key=lambda c: pearson_metrics[c]['sharpe'])

    # Grab that cluster's metrics so the table isn't a wall of nested indexing
    dtw_row = dtw_metrics[dtw_best]
    pearson_row = pearson_metrics[pearson_best]

    table = pd.DataFrame({
        'Method': ['DTW + HRP', 'Pearson + HRP', '1/N'],
        'Sharpe Ratio': [dtw_row['sharpe'],       pearson_row['sharpe'],       one_n_shape],
        'Mean Return':  [dtw_row['mean_return'],  pearson_row['mean_return'],  None],
        'Volatility':   [dtw_row['volatility'],   pearson_row['volatility'],   None],
        'Max Drawdown': [dtw_row['max_drawdown'], pearson_row['max_drawdown'], None],
    })
    return table

##### Pause ######

# returns the dataframe for days and tickers and cluster labels and pearson labels
def run_complete_backtest(returns, dtw_labels, pearson_labels):

    print("=" * 60)
    print("RUNNING THE FULL COMPLETE BACKTEST ")
    print("=" * 60)

    print("\n Evaluating DTW clusters: ")

    dtw_metric = evaluate_portfolio(returns,dtw_labels)

    print("Now Evaluating Pearson cluster")

    pearson_metric = evaluate_portfolio(returns, pearson_labels)

    print("Calculating 1/N benchmark ")

    one_n_sharpe = calculate_one_in_sharp(returns)


    # get the mean for each cluster
    dtw_returns = [value['mean_return'] for value in dtw_metric.values()]
    pearson_returns = [value['mean_return'] for value in pearson_metric.values()]

    t_stat, p_value = compare_dtw_vs_pearson(dtw_returns, pearson_returns)

    print(f" t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")


    # build a table for comparision
    print("\n Building the comparison table: ")

    table = build_comparison_table(dtw_metric, pearson_metric, one_n_sharpe)

    print("\n" + "+" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    # Allows for organized format
    print("\n" + table.to_string(index=False))

    print(f"\n DTW vs Pearson t-test: ")
    print(f"t-statistic: {t_stat:.4f}")
    print(f" p-value: {p_value:.4f}")

    # now we compare

    if p_value < 0.05:
        print("Significant change (if p < 0.05)")

        dtw_best = max(dtw_metric, key=lambda c: dtw_metric[c]['sharpe'])
        
        pearson_best = max(pearson_metric, key=lambda c: pearson_metric[c]['sharpe'])

        # now here it the sharpe ratio change
        if (dtw_metric[dtw_best]['Sharpe'] > pearson_metric[pearson_best]['sharpe']):
            print("DTW preforms better than Pearson")
        else:
            print("Pearson outperforms DTW")
        
    else:
        print("No signficance can be found")

    return {
        'dtw_metrics': dtw_metric,
        'pearson_metric': pearson_metric,
        'one_n_sharpe': one_n_sharpe,
        'comparison_table': table,
        't_state': t_stat,
        'p_value': p_value
    }

def main():
    




    



    













    



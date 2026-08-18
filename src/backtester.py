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

def evaluate_portfolio(returns, labels):
    """parameters hods returns of days and tickers and 
    labels hold cluster for each ticker"""
    metrics = {} # holds for key value for cluster and metric

    # find unique clusters
    unique_clusters = np.unique(labels)

    for each_cluster in unique_clusters:

        tickers_in_cluster = [each for each, label in enumerate(labels) if label == each_cluster]
        # so basically find for each cluster and assign a label for each value

        cluster_returns = returns.iloc[:, tickers_in_cluster]
        # finds the all of the rows and finds the column for just that.

        # find the cluster average
        cluster_avg = cluster_returns.mean(axis = 1)
        # average returns of stocks in cluster and this is horizontal anf finds the average.

        sharpe = calculate_sharpe(cluster_avg)

        mean_return = cluster_avg.mean() * 252 # for mean annual return

        volatility = cluster_avg.std() * np.sqrt(252)

        # measures the largest peak to decline in your investment value.
        max_drawdown = (cluster_avg / cluster_avg.cummax() - 1).min()

        # make a dictionary
        metrics[each_cluster] = {

            'sharpe': sharpe,
            'mean_return': mean_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'n_clusters': len(tickers_in_cluster)


        }

    return metrics

# compare the Dynamic Time Warping and also Pearson 
def compare_dtw_vs_pearson(dtw_returns, pearson_returns):
    """Compares the DTW vs Pearson methods works"""

    """ The Independent T test compares DTW vs pearson. """

    """     Part	What It Does
    stats.ttest_ind()	Independent t-test — compares two groups
    dtw_returns	Group 1 (DTW strategy)
    pearson_returns	Group 2 (Pearson strategy)
    t_stat	t-statistic — measures the difference
    p_value	p-value — probability the difference is random"""

    t_stat, p_value = stats.ttest_ind(dtw_returns, pearson_returns)

    return t_stat, p_value


# This is the benchmark 1/N weight to make sure every stock gets equal weight

def calculate_one_in_sharp(returns):
    """Calculate Sharpe ration for (1/N) to make sure its equal for each """
    """ Args: Return Dataframe (days x returns)"""
    """ returns the Sharpe ratio """

    equal_weights_returns = returns.mean(axis = 1)
    # goes back to calculate sharpe function to find and subtracts to find the return on from the risk.
    return calculate_sharpe(equal_weights_returns)




# matches the different comparsions table and will build the table
def build_comparison_table(dtw_metrics, pearson_metrics, one_n_shape):

    dtw_best = max(dtw_metrics, key = lambda x: dtw_metrics[x]['sharpe'])
    # this will basically use key to value pairs and  and the key finds the highest value from the key function for sharpe.

    pearson_best = max(pearson_metrics, key = lambda x: pearson_metrics[x]['sharpe'])

    # use that key to value pairs and means  it will use dtw and pearson metric format.

    table = pd.DataFrame({ 'Method': ['DTW + HRP', 'Pearson + HRP', '1/N'], 
                          'Sharpe Ratio': [dtw_metrics[dtw_best]['sharpe'], pearson_metrics[pearson_best]['sharpe'], one_n_shape],
                          'Mean Return': [dtw_metrics[dtw_best]['mean_return'], pearson_metrics[pearson_best]['mean_return'], None],
                          'Volatility': [dtw_metrics[dtw_best]['volatility'], pearson_metrics[pearson_best]['volatility'], None],
                          'Max Drawdown': [dtw_metrics[dtw_best]['max_drawdown'], pearson_metrics[pearson_best]['max_drawdown'], None]

    })
    return table

##### Pause ######











    



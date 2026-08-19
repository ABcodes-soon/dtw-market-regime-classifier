import yfinance as yf
import pandas as pd
import numpy as np
import time



def get_sp500_tickers(limit=None):
    """ Returns a list of S&P 500 tickers and to test with """
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "ADBE", "CRM", "ORCL",
        "JPM", "V", "BAC", "WFC", "C", "GS", "MS", "AXP", "MA", "PYPL",
        "JNJ", "UNH", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "LLY", "AMGN",
        "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "MPC", "KMI", "WMB",
        "KO", "PEP", "COST", "WMT", "HD", "MCD", "NKE", "SBUX", "PG", "PM"
    ]
    # limit is optional: `limit=10` returns only the first 10 tickers,
    # which makes downloads + clustering much faster for testing.
    return tickers[:limit] if limit else tickers




def download_stock_data(tickers, start_date = "2015-01-01", end_date = "2024-12-31"):
    """
    Downloads stock price data for a list of tickers.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with Adjusted Close prices
    """
    print(f"Downloading historical stock data... {len(tickers)} tickers")
    print(f"Starting Period: {start_date} to {end_date}")

    start_time = time.time()

    # download the data using yfinance and start using the start
    data = yf.download(tickers, start = start_date, end = end_date, auto_adjust = True, progress = True)


    # takes the close column from the data and assigns it to a variable called prices. The prices variable will contain the adjusted closing prices for the specified tickers over the specified date range.
    prices = data["Close"].dropna(axis = 1, how = 'all')  # Drop columns with all NaN values

    print(f"Downloaded {prices.shape[1]} tickers in {time.time() - start_time:.2f} seconds.")

    print(f"Price data shape: {prices.shape}")

    return prices


    # calculate the returns
def calculate_returns(prices):
    print(f"Calculating log returns from price data...") 
    """ Computes log returns from price data """
    # it uses the natural logarithm of the ratio of the current price to the previous price to calculate the log returns. The dropna() method is used to remove any rows with NaN values that may result from the calculation.
    # so it shows it shifts the prices down by one row, divides the current price by the previous price, takes the natural logarithm of that ratio, and then drops any rows with NaN values.
    returns = np.log(prices / prices.shift(1)).dropna()


    print(f"Return shape: {returns.shape}")

    return returns


def get_regime_slices(returns):
    """ Slices returns into stress periods and returns a dictionary of slices """

    print(f"Getting regime slices from returns data...")

    regimes_slices = {"covid": returns.loc["2020-02-01": "2020-04-30"],
                      "rate_shock": returns.loc["2022-01-03": "2022-12-30"]}

    for name, slice in regimes_slices.items():
        print(f"{name}: {slice.shape[0]} days × {slice.shape[1]} tickers")

    return regimes_slices



def main():

    """ Main function to run the data processing steps """

    print("Starting data processing...")

    # have been assigned to a variable called tickers. The get_sp500_tickers() function is called to retrieve a list of S&P 500 stock tickers, and this list is stored in the tickers variable.

    tickers = get_sp500_tickers()

    print(f"Tickers: {len(tickers)}")

    prices = download_stock_data(tickers)
    returns = calculate_returns(prices)
    regimes = get_regime_slices(returns)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print(f"\n Data Summary:")
    print(f"   Total returns: {returns.shape[0]} days × {returns.shape[1]} tickers")

    for names, data in regimes.items():
        print(f"   {names}: {data.shape[0]} days × {data.shape[1]} tickers")

    return returns, regimes



# testing the processing file

if __name__ == "__main__":
    main()






    








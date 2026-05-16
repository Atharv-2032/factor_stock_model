import yfinance as yf
import pandas as pd
import os

TICKERS = [
    # technology
    "AAPL", "MSFT", "IBM", "INTC", "CSCO", "TXN", "ADI", "QCOM", "HPQ", "GLW",
    # financials
    "JPM", "GS", "BAC", "WFC", "MS", "C", "AXP", "USB", "PNC", "MET",
    # healthcare
    "JNJ", "PFE", "ABT", "MDT", "UNH", "BMY", "LLY", "AMGN", "GILD", "BAX",
    # consumer staples
    "PG", "WMT", "KO", "PEP", "CL", "GIS", "K", "MKC", "SYY", "CLX",
    # industrials
    "BA", "CAT", "GE", "MMM", "HON", "UPS", "FDX", "EMR", "ETN", "ITW",
]

START_DATE = "2000-01-01"
END_DATE   = "2023-12-31"

def download_prices(tickers, start, end):
    """
    Download daily adjusted closing prices for a list of tickers.
    Returns a DataFrame where rows are dates and columns are tickers.
    """
    print(f"Downloading price data for {len(tickers)} tickers...")
    
    raw = yf.download(
        tickers,
        start = start,
        end = end,
        auto_adjust= True,
        progress= False
    )

    prices = raw["Close"]
    print(f"Raw data shape: {prices.shape}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    return prices

def clean_prices(prices):
    """
    Clean the price DataFrame:
    - Forward fill missing values (e.g. public holidays where one exchange closed)
    - Drop any stock that has more than 20% missing data
    """
    # how much data is missing per stock
    missing_pct = prices.isnull().mean()
    bad_tickers = missing_pct[missing_pct > 0.20].index.tolist()

    if bad_tickers:
        print(f"Dropping tickers with >20% missing data: {bad_tickers}")
        prices = prices.drop(columns=bad_tickers)

    # forward fill remaining gaps (e.g. one missing day between two trading days)
    prices = prices.ffill()
    prices = prices.bfill()
    # drop any remaining rows where all values are NaN
    prices = prices.dropna(how="all")

    print(f"Clean data shape: {prices.shape}")
    print(f"Missing values remaining: {prices.isnull().sum().sum()}")

    return prices

def save_prices(prices, path = "data/prices.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prices.to_csv(path)
    print(f"Saved to {path}")

def load_prices(path="data/prices.csv"):
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"Loaded prices: {prices.shape}")
    return prices

if __name__ == "__main__":
    prices = download_prices(TICKERS, START_DATE, END_DATE)
    prices = clean_prices(prices)
    save_prices(prices)

    # reload and verify
    prices = load_prices()
    print(prices)
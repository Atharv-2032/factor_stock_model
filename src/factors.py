import pandas as pd
import numpy as np

def get_monthly_prices(prices):
    """
    Resample daily prices to monthly.
    We take the last trading day of each month as the monthly price.
    """
    monthly = prices.resample("ME").last()
    return monthly

def compute_momentum(monthly_prices):
    """
    Momentum: 12 month trailing return excluding the most recent month.
    For January 2010: return from January 2009 to December 2009.
    
    We exclude last month because very recent returns exhibit reversal
    (which we capture separately), not momentum.
    """
    # shift by 1 to exclude last month, then compute 12 month return
    momentum = monthly_prices.shift(1).pct_change(12)
    return momentum
def compute_reversal(monthly_prices):
    """
    Reversal: last month's return, inverted.
    Stocks that did badly last month tend to bounce back.
    We invert so higher score = more likely to outperform.
    """
    reversal = -monthly_prices.pct_change(1)
    return reversal

def compute_volatility(daily_prices):
    """
    Volatility: standard deviation of daily returns over past 60 days, inverted.
    Computed from daily data then resampled to monthly.
    Lower volatility = higher score (inverted).
    """
    daily_returns = daily_prices.pct_change()

    # rolling 60 day std, then take last value of each month
    rolling_vol = daily_returns.rolling(60).std()
    monthly_vol = rolling_vol.resample("ME").last()

    # invert so lower volatility = higher score
    volatility = -monthly_vol
    return volatility
def compute_size(monthly_prices):
    """
    Size: log of price used as a proxy for size, inverted.
    Smaller companies tend to outperform larger ones.
    
    Note: ideally this uses market cap = price x shares outstanding.
    We use log price as a proxy since shares outstanding needs
    fundamental data. This is a known simplification.
    """
    size = -np.log(monthly_prices)
    return size

def cross_sectional_normalize(df):
    """
    Normalize each factor cross-sectionally every month.
    
    This means: for each month, rank all stocks by their factor score
    and scale those ranks to lie between 0 and 1.
    
    Why: removes the effect of the factor's absolute level changing
    over time. The model only sees relative rankings, not raw values.
    """
    return df.rank(axis=1, pct=True)


def compute_target(monthly_prices):
    """
    Target: next month's excess return.
    
    Step 1 - compute each stock's return this month
    Step 2 - subtract the average return across all stocks (excess return)
    Step 3 - shift backwards by 1 month so each row's target is NEXT month's value
    
    This is the most important function to get right.
    The shift(−1) is what ensures we never leak future information.
    """
    # step 1: monthly return for each stock
    monthly_returns = monthly_prices.pct_change()

    # step 2: subtract cross-sectional mean each month to get excess return
    excess_returns = monthly_returns.sub(monthly_returns.mean(axis=1), axis=0)

    # step 3: shift back by 1 so january's row gets february's excess return
    # this is the target — what we're trying to predict
    target = excess_returns.shift(-1)

    return target


def build_factor_table(daily_prices):
    """
    Master function. Takes daily prices, returns the full factor table.
    
    Output: one row per stock per month with columns:
    date, ticker, momentum, reversal, volatility, size, target
    """
    print("resampling to monthly prices...")
    monthly_prices = get_monthly_prices(daily_prices)

    print("computing factors...")
    momentum   = compute_momentum(monthly_prices)
    reversal   = compute_reversal(monthly_prices)
    volatility = compute_volatility(daily_prices)
    size       = compute_size(monthly_prices)
    target     = compute_target(monthly_prices)

    print("normalizing factors cross-sectionally...")
    momentum   = cross_sectional_normalize(momentum)
    reversal   = cross_sectional_normalize(reversal)
    volatility = cross_sectional_normalize(volatility)
    size       = cross_sectional_normalize(size)

    print("stacking into long format table...")
    # each factor is currently a wide table: rows=months, cols=tickers
    # we stack them so each row is one stock in one month
    def stack(df, name):
        return df.stack().rename(name)

    factor_table = pd.concat([
        stack(momentum,   "momentum"),
        stack(reversal,   "reversal"),
        stack(volatility, "volatility"),
        stack(size,       "size"),
        stack(target,     "target")
    ], axis=1)

    factor_table.index.names = ["date", "ticker"]
    factor_table = factor_table.reset_index()

    # drop rows where any factor or target is missing
    before = len(factor_table)
    factor_table = factor_table.dropna()
    after = len(factor_table)
    print(f"dropped {before - after} rows with missing values")
    print(f"final table shape: {factor_table.shape}")

    return factor_table


if __name__ == "__main__":
    import os
    from src.data import load_prices

    prices = load_prices()

    factor_table = build_factor_table(prices)

    os.makedirs("features", exist_ok=True)
    factor_table.to_csv("features/factors.csv", index=False)
    print("saved to features/factors.csv")
    print(factor_table.head(10))
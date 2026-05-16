import pandas as pd
import matplotlib.pyplot as plt

def validate_prices(path="data/prices.csv"):
    prices = pd.read_csv(path, index_col=0, parse_dates=True)

    # check 1 - shape
    print(f"shape: {prices.shape}")
    print(f"expected: ~5800 rows (trading days), 10 columns (tickers)\n")

    # check 2 - date range
    print(f"first date: {prices.index[0].date()}")
    print(f"last date:  {prices.index[-1].date()}\n")

    # check 3 - missing values
    print("missing values per ticker:")
    print(prices.isnull().sum())
    print()

    # plot apple price history
    plt.figure(figsize=(12, 4))
    plt.plot(prices["AAPL"], color="steelblue", linewidth=1)
    plt.title("AAPL adjusted closing price 2000-2023")
    plt.xlabel("date")
    plt.ylabel("price (USD)")
    plt.axvspan("2008-09-01", "2009-03-01", alpha=0.15, color="red", label="2008 crisis")
    plt.axvspan("2020-02-01", "2020-04-01", alpha=0.15, color="orange", label="2020 crash")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    validate_prices()
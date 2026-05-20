import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests

START_DATE = "2000-01-01"
END_DATE   = "2023-12-31"


def get_sp500_tickers():
    
    """
    S&P 500 tickers hardcoded - Wikipedia scraping blocked.
    This is the S&P 500 composition as of 2023.
    """
    tickers = [
        "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL",
        "A", "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT",
        "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP",
        "AXP", "AIG", "AMT", "AWK", "AMP", "AME", "AMGN", "APH", "ADI",
        "ANSS", "AON", "APA", "AAPL", "AMAT", "APTV", "ACGL", "ADM", "ANET",
        "AJG", "AIZ", "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY",
        "AXON", "BKR", "BALL", "BAC", "BK", "BBWI", "BAX", "BDX", "BRK-B",
        "BBY", "BIO", "TECH", "BIIB", "BLK", "BX", "BA", "BCR", "BSX",
        "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR", "BG", "CDNS", "CZR",
        "CPT", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CTLT", "CAT",
        "CBOE", "CBRE", "CDW", "CE", "COR", "CNC", "CNX", "CDAY", "CF",
        "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD", "CI", "CINF",
        "CTAS", "CSCO", "C", "CFG", "CLX", "CME", "CMS", "KO", "CTSH",
        "CL", "CMCSA", "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT",
        "GLW", "CPAY", "CTVA", "CSGP", "COST", "CTRA", "CCI", "CSX",
        "CMI", "CVS", "DHR", "DRI", "DVA", "DAY", "DE", "DAL", "XRAY",
        "DVN", "DXCM", "FANG", "DLR", "DFS", "DG", "DLTR", "D", "DPZ",
        "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "EMN", "ETN", "EBAY",
        "ECL", "EIX", "EW", "EA", "ELV", "LLY", "EMR", "ENPH", "ETR",
        "EOG", "EPAM", "EQT", "EFX", "EQIX", "EQR", "ESS", "EL", "ETSY",
        "EG", "EVRST", "ES", "EXC", "EXPE", "EXPD", "EXR", "XOM", "FFIV",
        "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB", "FSLR", "FE",
        "FI", "FMC", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN", "FCX",
        "GRMN", "IT", "GE", "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS",
        "GM", "GPC", "GILD", "GS", "HAL", "HIG", "HAS", "HCA", "DOC",
        "HSIC", "HSY", "HES", "HPE", "HLT", "HOLX", "HD", "HON", "HRL",
        "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX",
        "IDXX", "ITW", "INCY", "IR", "PODD", "INTC", "ICE", "IFF", "IP",
        "IPG", "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL",
        "JKHY", "J", "JNJ", "JCI", "JPM", "JNPR", "K", "KVUE", "KDP",
        "KEY", "KEYS", "KMB", "KIM", "KMI", "KLAC", "KHC", "KR", "LHX",
        "LH", "LRCX", "LW", "LVS", "LDOS", "LEN", "LII", "LLY", "LIN",
        "LYV", "LKQ", "LMT", "L", "LOW", "LULU", "LYB", "MTB", "MRO",
        "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", "MA", "MTCH", "MKC",
        "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", "MGM", "MCHP",
        "MU", "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", "MPWR",
        "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX",
        "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS",
        "NOC", "NCLH", "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY",
        "ODFL", "OMC", "ON", "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PANW",
        "PH", "PAYX", "PAYC", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM",
        "PSX", "PNW", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR",
        "PLD", "PRU", "PEG", "PTC", "PSA", "PHM", "QRVO", "PWR", "QCOM",
        "DGX", "RL", "RJF", "RTX", "O", "REG", "REGN", "RF", "RSG", "RMD",
        "RVTY", "ROK", "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM", "SBAC",
        "SLB", "STX", "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SW",
        "SNA", "SOLV", "SO", "LUV", "SWK", "SBUX", "STT", "STLD", "STE",
        "SYK", "SMCI", "SYF", "SNPS", "SYY", "TMUS", "TROW", "TTWO", "TPR",
        "TRGP", "TGT", "TEL", "TDY", "TFX", "TER", "TSLA", "TXN", "TXT",
        "TMO", "TJX", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC", "TYL",
        "TSN", "USB", "UBER", "UDR", "ULTA", "UNP", "UAL", "UPS", "URI",
        "UNH", "UHS", "VLO", "VTR", "VLTO", "VRSN", "VRSK", "VZ", "VRTX",
        "VTRS", "VICI", "V", "VST", "VMC", "WRB", "GWW", "WAB", "WBA",
        "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST",
        "WDC", "WY", "WHR", "WMB", "WTW", "WYNN", "XEL", "XYL", "YUM",
        "ZBRA", "ZBH", "ZTS"
    ]

    print(f"loaded {len(tickers)} tickers")
    return tickers


def download_prices(tickers, start, end, batch_size=100):
    """
    Download daily adjusted closing prices.
    Downloads in batches to avoid yfinance rate limits.
    """
    all_data = []
    
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    
    print(f"downloading {len(tickers)} tickers in {len(batches)} batches...")
    
    for i, batch in enumerate(batches):
        print(f"  batch {i+1}/{len(batches)}...")
        
        raw = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True
        )
        
        if "Close" in raw.columns:
            prices = raw["Close"]
        else:
            prices = raw
            
        all_data.append(prices)
    
    combined = pd.concat(all_data, axis=1)
    
    # drop duplicate columns if any
    combined = combined.loc[:, ~combined.columns.duplicated()]
    
    print(f"raw data shape: {combined.shape}")
    return combined


def clean_prices(prices, max_missing=0.20):
    """
    Clean price data:
    - drop tickers with more than max_missing % of data missing
    - forward fill then backward fill remaining gaps
    """
    missing_pct = prices.isnull().mean()
    bad_tickers = missing_pct[missing_pct > max_missing].index.tolist()
    
    if bad_tickers:
        print(f"dropping {len(bad_tickers)} tickers with >{max_missing:.0%} missing data")
        prices = prices.drop(columns=bad_tickers)
    
    prices = prices.ffill()
    prices = prices.bfill()
    
    remaining_missing = prices.isnull().sum().sum()
    
    print(f"clean data shape: {prices.shape}")
    print(f"missing values remaining: {remaining_missing}")
    print(f"date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    return prices


def save_prices(prices, path="data/prices.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prices.to_csv(path)
    print(f"saved to {path}")


def load_prices(path="data/prices.csv"):
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"loaded prices: {prices.shape}")
    return prices


if __name__ == "__main__":
    tickers = get_sp500_tickers()
    prices  = download_prices(tickers, START_DATE, END_DATE)
    prices  = clean_prices(prices)
    save_prices(prices)
    prices  = load_prices()
    print(prices.tail())
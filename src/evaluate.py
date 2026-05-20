import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def compute_rank_ic(predictions):
    """
    Compute monthly rank IC and t-statistic.
    """
    monthly_ic = []

    for date, group in predictions.groupby("date"):
        if len(group) < 3:
            continue
        ic, _ = spearmanr(group["predicted"], group["target"])
        monthly_ic.append(ic)

    monthly_ic = pd.Series(monthly_ic)
    mean_ic    = monthly_ic.mean()
    std_ic     = monthly_ic.std()
    n          = len(monthly_ic)
    t_stat     = mean_ic / (std_ic / np.sqrt(n))
    pct_pos    = (monthly_ic > 0).mean()

    print(f"\nrank IC results:")
    print(f"  mean IC:      {mean_ic:.4f}")
    print(f"  IC std:       {std_ic:.4f}")
    print(f"  t-statistic:  {t_stat:.4f}  (above 2.0 = statistically significant)")
    print(f"  IC > 0:       {pct_pos:.1%}")
    print(f"  months:       {n}")

    return monthly_ic


def backtest(predictions, top_n=10):
    """
    Simulate long-short portfolio.
    Long: top_n stocks by predicted score.
    Short: bottom_n stocks by predicted score.
    """
    monthly_returns = []

    for date, group in predictions.groupby("date"):
        if len(group) < top_n * 2:
            continue

        group           = group.sort_values("predicted", ascending=False)
        long_portfolio  = group.head(top_n)
        short_portfolio = group.tail(top_n)

        monthly_returns.append({
            "date":        date,
            "long_return": long_portfolio["target"].mean(),
            "short_return": short_portfolio["target"].mean(),
            "ls_return":   long_portfolio["target"].mean() - short_portfolio["target"].mean()
        })

    returns_df          = pd.DataFrame(monthly_returns).set_index("date")
    return returns_df


def compute_performance(returns_df):
    """
    Compute overall and yearly performance metrics.
    """
    ls             = returns_df["ls_return"]
    n_months       = len(ls)
    overall_return = (1 + ls).prod() - 1
    ann_return     = (1 + overall_return) ** (12 / n_months) - 1
    ann_vol        = ls.std() * np.sqrt(12)
    sharpe         = ann_return / ann_vol
    win_rate       = (ls > 0).mean()

    cumulative     = (1 + ls).cumprod()
    rolling_max    = cumulative.cummax()
    drawdown       = (cumulative - rolling_max) / rolling_max
    max_drawdown   = drawdown.min()

    print(f"\noverall performance (long-short portfolio):")
    print(f"  period:        {ls.index[0].strftime('%Y-%m')} to {ls.index[-1].strftime('%Y-%m')}")
    print(f"  months:        {n_months}")
    print(f"  total return:  {overall_return:.1%}")
    print(f"  annual return: {ann_return:.1%}")
    print(f"  annual vol:    {ann_vol:.1%}")
    print(f"  sharpe ratio:  {sharpe:.3f}")
    print(f"  max drawdown:  {max_drawdown:.1%}")
    print(f"  win rate:      {win_rate:.1%}")

    print(f"\nyearly performance:")
    print(f"  {'year':<6} {'return':>8} {'sharpe':>8} {'win rate':>10}")
    print(f"  {'-'*36}")

    returns_df = returns_df.copy()
    returns_df["year"] = returns_df.index.year

    for year, group in returns_df.groupby("year"):
        yr_ls      = group["ls_return"]
        yr_return  = (1 + yr_ls).prod() - 1
        yr_vol     = yr_ls.std() * np.sqrt(12)
        yr_sharpe  = yr_return / yr_vol if yr_vol > 0 else 0
        yr_winrate = (yr_ls > 0).mean()
        print(f"  {year:<6} {yr_return:>8.1%} {yr_sharpe:>8.3f} {yr_winrate:>10.1%}")

    return {
        "annual_return": ann_return,
        "sharpe":        sharpe,
        "max_drawdown":  max_drawdown,
        "win_rate":      win_rate
    }
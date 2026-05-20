import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr

def load_factors(path = "features/factors.csv"):
    df = pd.read_csv(path,parse_dates = ["date"])
    print(f"loaded factor table: {df.shape}")
    return df

def walk_forward(df, train_years = 3):
    """
    Walk-forward validation loop.
    
    For each month in the test period:
      - train on all data strictly before that month
      - predict that month's stock scores
      - store predictions
    
    train_years: minimum years of data before we start predicting
    """
    features = ["momentum", "reversal", "volatility", "size"]

    all_months = sorted(df["date"].unique())
    start_idx = train_years*12
    test_months = all_months[start_idx:]

    print(f"total months available:  {len(all_months)}")
    print(f"first training ends:     {all_months[start_idx - 1].strftime('%Y-%m')}")
    print(f"first prediction month:  {test_months[0].strftime('%Y-%m')}")
    print(f"last prediction month:   {test_months[-1].strftime('%Y-%m')}")
    print(f"total prediction months: {len(test_months)}")
    print(f"\nrunning walk-forward loop...")

    all_predictions = []

    for i, test_month in enumerate(test_months):
        train_data = df[df["date"] < test_month]
        test_data = df[df["date"] == test_month]
        if len(test_data) == 0:
            continue

        X_train = train_data[features]
        y_train = train_data["target"]
        X_test = test_data[features]

        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=5,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train,y_train)

        preds = model.predict(X_test)

        result = test_data[["date", "ticker", "target"]].copy()
        result["predicted"] = preds
        all_predictions.append(result)

        if (i + 1) % 12 == 0:
            print(f"  completed {i + 1}/{len(test_months)} months "
                  f"— up to {test_month.strftime('%Y-%m')}")

    predictions = pd.concat(all_predictions, ignore_index=True)
    print(f"\nwalk-forward complete")
    print(f"total predictions: {len(predictions)}")

    return predictions

def compute_rank_ic(predictions):
    """
    Compute monthly rank IC across all prediction months.
    Then compute t-statistic to test if mean IC is significantly above zero.
    """

    monthly_ic = []

    for data, group in predictions.groupby("date"):
        if len(group) < 3:
            continue

        ic, _ = spearmanr(group["predicted"], group["target"])
        monthly_ic.append(ic)
    monthly_ic = pd.Series(monthly_ic)

    mean_ic  = monthly_ic.mean()
    std_ic   = monthly_ic.std()
    n        = len(monthly_ic)
    t_stat   = mean_ic / (std_ic / np.sqrt(n))
    pct_pos  = (monthly_ic > 0).mean()

    print(f"\nrank IC results:")
    print(f"  mean IC:      {mean_ic:.4f}")
    print(f"  IC std:       {std_ic:.4f}")
    print(f"  t-statistic:  {t_stat:.4f}  (above 2.0 = statistically significant)")
    print(f"  IC > 0:       {pct_pos:.1%}")
    print(f"  months:       {n}")

    return monthly_ic

def backtest(predictions, top_n = 10):
     """
    Simulate a long-short portfolio every month.
    
    Long:  top_n stocks by predicted score
    Short: bottom_n stocks by predicted score
    
    Monthly return = avg return of long portfolio - avg return of short portfolio
    """
     
     monthly_returns = []
     for date, group in predictions.groupby("date"):
         if len(group) < top_n*2:
             continue
         
         group = group.sort_values("predicted", ascending=False)
         long_portfolio = group.head(top_n)
         short_portfolio = group.tail(top_n)

         long_return = long_portfolio["target"].mean()
         short_return = short_portfolio["target"].mean()

         ls_return = long_return - short_return

         monthly_returns.append({
            "date":         date,
            "long_return":  long_return,
            "short_return": short_return,
            "ls_return":    ls_return
        })

     returns_df = pd.DataFrame(monthly_returns).set_index("date")
     return returns_df

def compute_performance(returns_df):
    """
    Compute overall and yearly performance metrics.
    """
    ls = returns_df["ls_return"]

    # overall metrics
    n_months        = len(ls)
    overall_return  = (1 + ls).prod() - 1
    ann_return      = (1 + overall_return) ** (12 / n_months) - 1
    ann_vol         = ls.std() * np.sqrt(12)
    sharpe          = ann_return / ann_vol
    win_rate        = (ls > 0).mean()

    # max drawdown
    cumulative      = (1 + ls).cumprod()
    rolling_max     = cumulative.cummax()
    drawdown        = (cumulative - rolling_max) / rolling_max
    max_drawdown    = drawdown.min()

    print(f"\noverall performance (long-short portfolio):")
    print(f"  period:          {ls.index[0].strftime('%Y-%m')} to {ls.index[-1].strftime('%Y-%m')}")
    print(f"  months:          {n_months}")
    print(f"  total return:    {overall_return:.1%}")
    print(f"  annual return:   {ann_return:.1%}")
    print(f"  annual vol:      {ann_vol:.1%}")
    print(f"  sharpe ratio:    {sharpe:.3f}")
    print(f"  max drawdown:    {max_drawdown:.1%}")
    print(f"  win rate:        {win_rate:.1%}")

    # yearly metrics
    print(f"\nyearly performance:")
    print(f"  {'year':<6} {'return':>8} {'sharpe':>8} {'win rate':>10}")
    print(f"  {'-'*36}")

    returns_df["year"] = returns_df.index.year

    for year, group in returns_df.groupby("year"):
        yr_ls       = group["ls_return"]
        yr_return   = (1 + yr_ls).prod() - 1
        yr_vol      = yr_ls.std() * np.sqrt(12)
        yr_sharpe   = yr_return / yr_vol if yr_vol > 0 else 0
        yr_winrate  = (yr_ls > 0).mean()

        print(f"  {year:<6} {yr_return:>8.1%} {yr_sharpe:>8.3f} {yr_winrate:>10.1%}")

    return {
        "annual_return": ann_return,
        "sharpe":        sharpe,
        "max_drawdown":  max_drawdown,
        "win_rate":      win_rate
    }

if __name__ == "__main__":
    import os

    df = load_factors()

    # walk-forward loop
    predictions = walk_forward(df, train_years=3)

    # save predictions
    os.makedirs("outputs", exist_ok=True)
    predictions.to_csv("outputs/predictions.csv", index=False)
    print("saved predictions to outputs/predictions.csv")

    # evaluate model
    monthly_ic = compute_rank_ic(predictions)

    # backtest strategy
    returns_df = backtest(predictions, top_n=10)

    # performance metrics
    metrics = compute_performance(returns_df)
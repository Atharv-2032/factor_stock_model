import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data    import load_prices
from src.factors import build_factor_table
from src.model   import walk_forward, save_predictions, load_predictions
from src.evaluate import compute_rank_ic, backtest, compute_performance
import os

REBUILD_DATA    = False   # set True to re-download prices
REBUILD_FACTORS = False   # set True to recompute factors
REBUILD_PREDS   = False   # set True to re-run walk-forward loop

if __name__ == "__main__":

    # step 1: load prices
    print("=" * 50)
    print("step 1: prices")
    print("=" * 50)
    prices = load_prices()

    # step 2: compute factors
    print("\n" + "=" * 50)
    print("step 2: factors")
    print("=" * 50)
    if REBUILD_FACTORS or not os.path.exists("features/factors.csv"):
        import pandas as pd
        factor_table = build_factor_table(prices)
        factor_table.to_csv("features/factors.csv", index=False)
    else:
        import pandas as pd
        factor_table = pd.read_csv("features/factors.csv", parse_dates=["date"])
        print(f"loaded factor table: {factor_table.shape}")

    # step 3: walk-forward
    print("\n" + "=" * 50)
    print("step 3: walk-forward validation")
    print("=" * 50)
    if REBUILD_PREDS or not os.path.exists("outputs/predictions.csv"):
        predictions = walk_forward(factor_table, train_years=3)
        save_predictions(predictions)
    else:
        predictions = load_predictions()

    # step 4: evaluate
    print("\n" + "=" * 50)
    print("step 4: evaluation")
    print("=" * 50)
    monthly_ic = compute_rank_ic(predictions)
    returns_df = backtest(predictions, top_n=10)
    metrics    = compute_performance(returns_df)
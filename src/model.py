import pandas as pd
import numpy as np
import lightgbm as lgb
import os


FEATURES = ["momentum", "reversal", "volatility", "size"]


def get_model():
    return lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=5,
        random_state=42,
        verbose=-1
    )


def walk_forward(df, train_years=3):
    """
    Walk-forward validation loop.
    Train on all data strictly before each test month.
    Predict that month's stock scores.
    Returns full prediction series.
    """
    all_months  = sorted(df["date"].unique())
    start_idx   = train_years * 12
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
        test_data  = df[df["date"] == test_month]

        if len(test_data) == 0:
            continue

        X_train = train_data[FEATURES]
        y_train = train_data["target"]
        X_test  = test_data[FEATURES]

        model = get_model()
        model.fit(X_train, y_train)

        preds  = model.predict(X_test)
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


def save_predictions(predictions, path="outputs/predictions.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    predictions.to_csv(path, index=False)
    print(f"saved predictions to {path}")


def load_predictions(path="outputs/predictions.csv"):
    predictions = pd.read_csv(path, parse_dates=["date"])
    print(f"loaded predictions: {predictions.shape}")
    return predictions
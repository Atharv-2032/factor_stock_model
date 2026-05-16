import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr

def load_factors(path="features/factors.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"loaded factor table: {df.shape}")
    return df


def train_test_split(df, split_date="2019-01-01"):
    """
    Simple time-based split.
    Everything before split_date is training.
    Everything after is test.
    """
    train = df[df["date"] < split_date]
    test  = df[df["date"] >= split_date]

    print(f"train: {train['date'].min().date()} to {train['date'].max().date()} — {len(train)} rows")
    print(f"test:  {test['date'].min().date()} to {test['date'].max().date()} — {len(test)} rows")

    return train, test

def get_features_target(df):
    features = ["momentum", "reversal", "volatility", "size"]
    X = df[features]
    y = df["target"]
    return X, y

def train_model(X_train, y_train):
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=5,
        random_state=42,
        verbose=-1        # silences lightgbm output
    )
    model.fit(X_train, y_train)
    print("model trained")
    return model
def compute_rank_ic(df, predictions):
    """
    Rank IC: for each month, compute the spearman correlation between
    predicted rankings and actual return rankings.
    Then average across all months.

    Spearman correlation is just pearson correlation applied to ranks
    instead of raw values — more robust to outliers.
    """
    df = df.copy()
    df["predicted"] = predictions

    monthly_ic = []

    for date, group in df.groupby("date"):
        if len(group) < 3:
            # need at least 3 stocks to compute a meaningful correlation
            continue

        ic, _ = spearmanr(group["predicted"], group["target"])
        monthly_ic.append(ic)

    monthly_ic = pd.Series(monthly_ic)

    print(f"\nrank IC results:")
    print(f"mean IC:     {monthly_ic.mean():.4f}")
    print(f"IC std:      {monthly_ic.std():.4f}")
    print(f"IC > 0 pct:  {(monthly_ic > 0).mean():.1%}  (months where model ranked correctly)")
    print(f"min IC:      {monthly_ic.min():.4f}")
    print(f"max IC:      {monthly_ic.max():.4f}")

    return monthly_ic


def evaluate_feature_importance(model, feature_names):
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    print(f"\nfeature importance:")
    for feat, imp in importance.items():
        bar = "█" * int(imp / importance.max() * 20)
        print(f"  {feat:<12} {bar} {imp:.0f}")

    return importance


if __name__ == "__main__":
    # load data
    df = load_factors()

    # split
    train, test = train_test_split(df, split_date="2019-01-01")

    # get features and target
    X_train, y_train = get_features_target(train)
    X_test,  y_test  = get_features_target(test)

    # train
    model = train_model(X_train, y_train)

    # predict on test set
    predictions = model.predict(X_test)

    # evaluate
    monthly_ic = compute_rank_ic(test, predictions)

    # feature importance
    evaluate_feature_importance(model, ["momentum", "reversal", "volatility", "size"])
    
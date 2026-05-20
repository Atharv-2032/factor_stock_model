# Stock Return Prediction Using ML on Classical Quantitative Factors

A machine learning system that predicts which S&P 500 stocks will outperform the market each month, trained on four classical quantitative factors, and evaluated through a rigorous walk-forward simulation across 20 years of market data.

---

## Problem Statement

Given a universe of S&P 500 stocks, can a LightGBM model trained on classical factor scores predict monthly cross-sectional stock returns well enough to generate a long-short portfolio with a Sharpe ratio above 1.0 — and is that result statistically significant?

---

## Results

| Metric | Value |
|--------|-------|
| Mean Rank IC | 0.0342 |
| IC t-statistic | 3.79 (> 2.0 = significant) |
| IC > 0 months | 60.9% |
| Sharpe Ratio | 1.016 |
| Annual Return | 27.8% |
| Max Drawdown | -29.7% |
| Win Rate | 61.8% |
| 2008 Crisis Return | +24.9% (S&P 500: -38%) |
| 2020 COVID Return | +39.8% (S&P 500: -20%) |

---

## Project Structure

```
factor_model/
├── src/
│   ├── data.py         # download and clean S&P 500 price data
│   ├── factors.py      # compute 4 factors, build training table
│   ├── model.py        # walk-forward validation loop
│   └── evaluate.py     # rank IC, backtesting, performance metrics
├── notebooks/
│   ├── run.py          # single script: runs entire pipeline
│   ├── day3_model.py   # simple train/test split exploration
│   ├── day4_walkforward.py  # walk-forward exploration script
│   └── validate_data.py     # data quality checks
├── data/               # raw price CSVs (generated)
├── features/           # factor table CSV (generated)
└── outputs/            # predictions and results (generated)
```

---

## Setup

**1. Clone the repository and create a virtual environment**

```bash
git clone <repo-url>
cd factor_model
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install yfinance pandas numpy lightgbm scikit-learn scipy matplotlib
```

---

## Running the Pipeline

The entire pipeline runs from a single script:

```bash
python3 notebooks/run.py
```

On first run this loads pre-computed data from the `data/`, `features/`, and `outputs/` folders. To rebuild any stage from scratch, set the corresponding flag at the top of `run.py`:

```python
REBUILD_DATA    = False   # re-download 378 stocks (~15 min)
REBUILD_FACTORS = False   # recompute factor table (~2 min)
REBUILD_PREDS   = False   # re-run walk-forward loop (~20 min)
```

To rebuild everything from scratch:

```bash
# step by step
python3 -m src.data        # download prices → data/prices.csv
python3 -m src.factors     # compute factors → features/factors.csv
python3 notebooks/day4_walkforward.py   # walk-forward → outputs/predictions.csv
```

---

## The Four Factors

All factors are computed from daily price data and cross-sectionally normalized to percentile ranks [0, 1] each month.

| Factor | Definition | Academic Basis |
|--------|-----------|----------------|
| Momentum | 12-month trailing return, excluding last month | Jegadeesh & Titman (1993) |
| Reversal | Prior month return, inverted | Short-term mean reversion |
| Low Volatility | 60-day return std dev, inverted | Low-volatility anomaly |
| Size | Log price, inverted (proxy for market cap) | Fama & French (1993) |

---

## Methodology

### Walk-Forward Validation

Standard cross-validation cannot be applied to time-series data — randomly shuffling observations leaks future information into training. Walk-forward validation is used instead:

```
Round 1:  train 2001–2003  →  predict Feb 2004
Round 2:  train 2001–2004  →  predict Mar 2004
...
Round 238: train 2001–2023  →  predict Nov 2023
```

The model is retrained from scratch each month on all data strictly before the test month. The `<` operator on date filtering is critical — `<=` would include the test month in training (lookahead bias).

### Long-Short Portfolio

Every month, stocks are ranked by predicted return. The top 10 are bought (long) and the bottom 10 are sold short. The strategy holds for one month then rebalances. The long-short structure makes the strategy approximately market neutral — validated by strong performance during both the 2008 crisis and 2020 COVID crash.

### Evaluation

- **Rank IC** — Spearman correlation between predicted and actual rankings, computed per month and averaged. Measures whether the model ranked stocks correctly.
- **IC t-statistic** — tests whether mean IC is statistically above zero. Above 2.0 = significant at 95% confidence.
- **Sharpe Ratio** — annualized return divided by annualized volatility. Primary strategy performance metric.
- **Max Drawdown** — largest peak-to-trough loss. Measures worst-case risk.

---

## Model

**Algorithm:** LightGBM gradient boosting regressor

```python
lgb.LGBMRegressor(
    n_estimators      = 200,
    learning_rate     = 0.05,
    max_depth         = 4,
    num_leaves        = 15,
    min_child_samples = 5,
    random_state      = 42
)
```

LightGBM was chosen for its speed on tabular data, robustness to noisy financial signals, and native SHAP support for explainability.

---

## Data

- **Universe:** S&P 500 constituent stocks (current composition)
- **Period:** January 2000 — December 2023
- **Source:** Yahoo Finance via `yfinance`
- **Clean tickers:** 378 (119 dropped due to >20% missing data)
- **Factor table:** 103,572 rows (stock × month pairs)
- **Out-of-sample predictions:** 89,964 (238 months × ~378 stocks)

**Known limitation:** The universe uses the current S&P 500 composition, meaning only companies that survived to 2023 are included. This survivorship bias likely inflates historical returns. A production implementation would use a point-in-time constituent list.

---

## Limitations

- **Survivorship bias** — only current S&P 500 survivors included
- **No transaction costs** — monthly rebalancing costs ~0.2% per month in practice
- **Concentrated portfolio** — 10 stocks per side amplifies single-stock effects
- **Only 4 factors** — value and quality factors excluded due to data constraints
- **No regime detection** — model weights factors equally across all market conditions

---

## Extensions (Future Work)

- Add value (book-to-market) and quality (return on equity) factors from fundamental data
- Expand to 50+ stocks per side to reduce concentration risk
- Implement transaction cost model to measure net-of-cost performance
- Add SHAP analysis to measure factor importance across market regimes
- Use a point-in-time S&P 500 constituent list to eliminate survivorship bias
- Hyperparameter tuning with Optuna using time-aware cross-validation

---

## Course

Machine Learning — IS62

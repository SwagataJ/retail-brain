"""
Retail Brain — Sales Forecast Prediction

Loads a trained NBEATSx model and generates 30-day sales forecasts
(2026-01-01 to 2026-01-30) for all 5 product categories.

Outputs:
  - data/forecasts/forecast_results.csv (150 rows: 5 categories × 30 days)
  - Prints per-category summary table to stdout
"""

import os
import warnings

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FORECAST_DIR = os.path.join(DATA_DIR, "forecasts")
MODEL_DIR = os.path.join(ROOT, "models", "nbeats_model")

# ── Config ────────────────────────────────────────────────────────────────────
HORIZON = 30
FORECAST_START = pd.Timestamp("2026-01-01")
CATEGORIES = ["Beauty", "Clothing", "Electronics", "Groceries", "Home"]

FUTR_EXOG_COLS = [
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "active_promos",
]


def build_future_exog() -> pd.DataFrame:
    """Build future exogenous features for the 30-day forecast window."""
    future_dates = pd.date_range(FORECAST_START, periods=HORIZON, freq="D")

    rows = []
    for cat in CATEGORIES:
        for d in future_dates:
            dow = d.dayofweek
            month = d.month
            rows.append({
                "unique_id": cat,
                "ds": d,
                "day_of_week_sin": np.sin(2 * np.pi * dow / 7),
                "day_of_week_cos": np.cos(2 * np.pi * dow / 7),
                "month_sin": np.sin(2 * np.pi * month / 12),
                "month_cos": np.cos(2 * np.pi * month / 12),
                "is_weekend": float(dow >= 5),
                "active_promos": 0.0,  # no promos assumed for future
            })

    return pd.DataFrame(rows)


def add_exog_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add exogenous features to a future dataframe from make_future_dataframe()."""
    dow = df["ds"].dt.dayofweek
    month = df["ds"].dt.month
    df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_weekend"] = (dow >= 5).astype(float)
    df["active_promos"] = 0.0  # no promos assumed for future
    return df


def main():
    # Load model
    print(f"Loading model from {MODEL_DIR}/...")
    nf = NeuralForecast.load(path=MODEL_DIR)

    # Build future exogenous DataFrame using model's expected dates
    futr_df = nf.make_future_dataframe()
    futr_df = add_exog_features(futr_df)

    # Generate predictions
    print("Generating 30-day forecast...")
    preds = nf.predict(futr_df=futr_df)
    preds = preds.reset_index()

    # Find the prediction column
    pred_col = [c for c in preds.columns if c.startswith("NBEATSx")][0]

    # Clamp negative predictions to 0
    preds[pred_col] = preds[pred_col].clip(lower=0)

    # Prepare output
    results = preds[["unique_id", "ds", pred_col]].copy()
    results.columns = ["category", "date", "predicted_revenue"]
    results["predicted_revenue"] = results["predicted_revenue"].round(2)
    results = results.sort_values(["category", "date"]).reset_index(drop=True)

    # Save CSV
    os.makedirs(FORECAST_DIR, exist_ok=True)
    out_path = os.path.join(FORECAST_DIR, "forecast_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\nForecast saved to {out_path} ({len(results)} rows)")

    # Print summary table
    print("\n" + "=" * 65)
    print("30-DAY FORECAST SUMMARY (Jan 2026)")
    print("=" * 65)
    print(f"{'Category':<15} {'Total Rev':>12} {'Daily Avg':>12} {'Min Day':>10} {'Max Day':>10}")
    print("-" * 59)

    for cat in CATEGORIES:
        cat_data = results[results["category"] == cat]["predicted_revenue"]
        print(
            f"{cat:<15} "
            f"${cat_data.sum():>11,.2f} "
            f"${cat_data.mean():>11,.2f} "
            f"${cat_data.min():>9,.2f} "
            f"${cat_data.max():>9,.2f}"
        )

    total = results["predicted_revenue"].sum()
    daily_avg = results.groupby("date")["predicted_revenue"].sum().mean()
    print("-" * 59)
    print(f"{'ALL':<15} ${total:>11,.2f} ${daily_avg:>11,.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()

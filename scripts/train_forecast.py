"""
Retail Brain — Sales Forecasting Training Pipeline

Trains an NBEATSx (N-BEATS with exogenous variables) model on daily category
revenue. Uses neuralforecast for training and evaluation.

Steps:
  1. Load & aggregate transaction data → daily revenue by category
  2. Add temporal features (day-of-week, month cyclical) + promotion counts
  3. Save intermediate data/daily_sales.csv
  4. Train NBEATSx via NeuralForecast
  5. Evaluate on validation set (MAE, MAPE, RMSE per category + overall)
  6. Save evaluation plot and trained model
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx
from neuralforecast.losses.pytorch import MAE

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FORECAST_DIR = os.path.join(DATA_DIR, "forecasts")
MODEL_DIR = os.path.join(ROOT, "models", "nbeats_model")

os.makedirs(FORECAST_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
HORIZON = 30
INPUT_SIZE = 60
SEED = 42
MAX_STEPS = 1000
EARLY_STOP_PATIENCE = 50
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

DATE_START = pd.Timestamp("2024-01-01")
DATE_END = pd.Timestamp("2025-12-31")
TRAIN_END = pd.Timestamp("2025-09-30")
VAL_START = pd.Timestamp("2025-10-01")
VAL_END = pd.Timestamp("2025-12-31")

FUTR_EXOG_COLS = [
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "active_promos",
]


# ── Step 1: Load & Aggregate ─────────────────────────────────────────────────

def load_and_aggregate() -> pd.DataFrame:
    """Join transaction_items → transactions → products, aggregate daily revenue per category."""
    print("Loading CSVs...")
    items = pd.read_csv(os.path.join(DATA_DIR, "transaction_items.csv"))
    txns = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"), parse_dates=["transaction_date"])
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))

    # Join to get category and date on each line item
    items = items.merge(txns[["transaction_id", "transaction_date"]], on="transaction_id")
    items = items.merge(products[["product_id", "category"]], on="product_id")

    # Revenue per line item
    items["revenue"] = items["unit_price"] * items["quantity"]

    # Aggregate: daily revenue by category
    daily = (
        items
        .groupby(["category", "transaction_date"])["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"transaction_date": "ds", "revenue": "y"})
    )

    # Build complete grid: every category × every date
    categories = sorted(daily["category"].unique())
    all_dates = pd.date_range(DATE_START, DATE_END, freq="D")
    grid = pd.MultiIndex.from_product([categories, all_dates], names=["category", "ds"])
    grid_df = pd.DataFrame(index=grid).reset_index()

    daily = grid_df.merge(daily, on=["category", "ds"], how="left")
    daily["y"] = daily["y"].fillna(0.0)

    # NeuralForecast requires 'unique_id' column
    daily["unique_id"] = daily["category"]
    daily = daily.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    print(f"  Daily sales grid: {len(daily)} rows ({len(categories)} categories × {len(all_dates)} days)")
    return daily


# ── Step 2: Feature Engineering ───────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features and weekend flag."""
    dow = df["ds"].dt.dayofweek  # 0=Mon, 6=Sun
    month = df["ds"].dt.month

    df["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["is_weekend"] = (dow >= 5).astype(float)

    return df


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Count active promotions per category-day."""
    promos = pd.read_csv(os.path.join(DATA_DIR, "promotions.csv"), parse_dates=["start_date", "end_date"])
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))

    # Resolve product-level promos to their category
    prod_cat = products.set_index("product_id")["category"].to_dict()

    promo_records = []
    for _, row in promos.iterrows():
        cat = row["category"] if pd.notna(row["category"]) else prod_cat.get(row.get("product_id"))
        if cat is None:
            continue
        dates = pd.date_range(row["start_date"], row["end_date"], freq="D")
        for d in dates:
            promo_records.append({"category": cat, "ds": d})

    if promo_records:
        promo_df = pd.DataFrame(promo_records)
        promo_counts = promo_df.groupby(["category", "ds"]).size().reset_index(name="active_promos")
        df = df.merge(promo_counts, on=["category", "ds"], how="left")
        df["active_promos"] = df["active_promos"].fillna(0.0)
    else:
        df["active_promos"] = 0.0

    return df


# ── Step 3: Train & Evaluate ─────────────────────────────────────────────────

def evaluate(actual: pd.Series, predicted: pd.Series) -> dict:
    """Compute MAE, RMSE, MAPE."""
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    # MAPE only on non-zero actuals to avoid division by zero
    mask = actual > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def main():
    np.random.seed(SEED)

    # Load & prepare data
    daily = load_and_aggregate()
    daily = add_temporal_features(daily)
    daily = add_promo_features(daily)

    # Save intermediate CSV
    daily_csv_path = os.path.join(DATA_DIR, "daily_sales.csv")
    daily.to_csv(daily_csv_path, index=False)
    print(f"Saved {daily_csv_path}")

    # Prepare NeuralForecast-compatible DataFrame
    nf_cols = ["unique_id", "ds", "y"] + FUTR_EXOG_COLS
    nf_df = daily[nf_cols].copy()

    # Split: train up to TRAIN_END, val from VAL_START to VAL_END
    train_df = nf_df[nf_df["ds"] <= TRAIN_END].copy()
    val_df = nf_df[(nf_df["ds"] >= VAL_START) & (nf_df["ds"] <= VAL_END)].copy()

    print(f"\nTrain set: {len(train_df)} rows (up to {TRAIN_END.date()})")
    print(f"Validation set: {len(val_df)} rows ({VAL_START.date()} to {VAL_END.date()})")

    # Configure NBEATSx
    model = NBEATSx(
        h=HORIZON,
        input_size=INPUT_SIZE,
        stack_types=["identity", "trend", "seasonality"],
        n_blocks=[1, 1, 1],
        mlp_units=3 * [[512, 512]],
        futr_exog_list=FUTR_EXOG_COLS,
        scaler_type="standard",
        loss=MAE(),
        max_steps=MAX_STEPS,
        early_stop_patience_steps=EARLY_STOP_PATIENCE,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        random_seed=SEED,
        val_check_steps=50,
    )

    nf = NeuralForecast(models=[model], freq="D")

    # Train (using validation set for early stopping)
    print("\nTraining NBEATSx...")
    nf.fit(df=train_df, val_size=HORIZON)

    # Cross-validation on validation period
    # We use the full data up to VAL_END and predict the last HORIZON days
    full_df = nf_df[nf_df["ds"] <= VAL_END].copy()

    print("\nGenerating validation predictions...")
    preds = nf.predict(futr_df=val_df)
    preds = preds.reset_index()

    # The predict output has columns: unique_id, ds, NBEATSx
    pred_col = [c for c in preds.columns if c.startswith("NBEATSx")][0]

    # Merge predictions with actual values for evaluation
    eval_df = val_df[val_df["ds"].isin(preds["ds"].unique())].merge(
        preds[["unique_id", "ds", pred_col]], on=["unique_id", "ds"], how="inner"
    )
    eval_df[pred_col] = eval_df[pred_col].clip(lower=0)

    # Evaluate per category
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"{'Category':<15} {'MAE':>10} {'RMSE':>10} {'MAPE (%)':>10}")
    print("-" * 45)

    all_metrics = []
    categories = sorted(eval_df["unique_id"].unique())
    for cat in categories:
        mask = eval_df["unique_id"] == cat
        metrics = evaluate(eval_df.loc[mask, "y"], eval_df.loc[mask, pred_col])
        all_metrics.append(metrics)
        print(f"{cat:<15} {metrics['MAE']:>10.2f} {metrics['RMSE']:>10.2f} {metrics['MAPE']:>9.1f}%")

    overall = evaluate(eval_df["y"], eval_df[pred_col])
    print("-" * 45)
    print(f"{'OVERALL':<15} {overall['MAE']:>10.2f} {overall['RMSE']:>10.2f} {overall['MAPE']:>9.1f}%")
    print("=" * 70)

    # Generate evaluation plot
    fig, axes = plt.subplots(len(categories), 1, figsize=(14, 3 * len(categories)), sharex=True)
    if len(categories) == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        cat_actual = val_df[val_df["unique_id"] == cat].sort_values("ds")
        cat_pred = eval_df[eval_df["unique_id"] == cat].sort_values("ds")

        ax.plot(cat_actual["ds"], cat_actual["y"], label="Actual", color="steelblue", linewidth=1.5)
        ax.plot(cat_pred["ds"], cat_pred[pred_col], label="Predicted", color="coral",
                linewidth=1.5, linestyle="--")
        ax.set_ylabel("Revenue ($)")
        ax.set_title(f"{cat}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle("NBEATSx Validation: Actual vs Predicted (Oct-Dec 2025)", fontsize=14, y=1.01)
    plt.tight_layout()

    plot_path = os.path.join(FORECAST_DIR, "evaluation_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nEvaluation plot saved to {plot_path}")

    # Re-fit on ALL data (through Dec 2025) for production forecasting
    print("\nRe-training on full dataset for production model...")
    prod_model = NBEATSx(
        h=HORIZON,
        input_size=INPUT_SIZE,
        stack_types=["identity", "trend", "seasonality"],
        n_blocks=[1, 1, 1],
        mlp_units=3 * [[512, 512]],
        futr_exog_list=FUTR_EXOG_COLS,
        scaler_type="standard",
        loss=MAE(),
        max_steps=MAX_STEPS,
        early_stop_patience_steps=EARLY_STOP_PATIENCE,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        random_seed=SEED,
        val_check_steps=50,
    )
    nf_prod = NeuralForecast(models=[prod_model], freq="D")
    nf_prod.fit(df=nf_df, val_size=HORIZON)

    # Save production model
    nf_prod.save(path=MODEL_DIR, model_index=None, overwrite=True, save_dataset=True)
    print(f"Production model saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

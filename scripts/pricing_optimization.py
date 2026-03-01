"""
Retail Brain — Pricing Optimization Pipeline

Estimates price elasticity, analyzes discount sensitivity, optimizes discount
depth, identifies promotion timing windows, and assesses cannibalization risk
from transaction data.

Outputs (in data/pricing_optimization/):
  CSVs:  price_elasticity.csv, discount_sensitivity.csv, optimal_discounts.csv,
         promotion_timing.csv, cannibalization_risk.csv, margin_impact_summary.csv
  PNGs:  elasticity_by_category.png, discount_response_curves.png,
         optimal_discount_scatter.png, demand_seasonality_heatmap.png,
         cannibalization_network.png

Usage:
    source retail_brain_env/bin/activate
    python scripts/pricing_optimization.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ───────────────────────────────────────────────────────────────────
SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "pricing_optimization")

np.random.seed(SEED)


# ── Step 1: Load Data ────────────────────────────────────────────────────────

def load_data():
    """Load the four input CSVs."""
    print("Loading data...")
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    promotions = pd.read_csv(os.path.join(DATA_DIR, "promotions.csv"),
                             parse_dates=["start_date", "end_date"])
    items = pd.read_csv(os.path.join(DATA_DIR, "transaction_items.csv"))
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"),
                                parse_dates=["transaction_date"])

    print(f"  Products:          {len(products):,}")
    print(f"  Promotions:        {len(promotions):,}")
    print(f"  Transaction items: {len(items):,}")
    print(f"  Transactions:      {len(transactions):,}")
    return products, promotions, items, transactions


# ── Step 2: Price Elasticity ─────────────────────────────────────────────────

def compute_price_elasticity(products, promotions, items, transactions):
    """Estimate price elasticity per product using arc elasticity."""
    print("\nComputing price elasticity...")

    # Join items → transactions (for dates) → products (for category, prices)
    df = items.merge(
        transactions[["transaction_id", "transaction_date"]], on="transaction_id"
    )
    df = df.merge(
        products[["product_id", "category", "base_price", "cost_price"]],
        on="product_id",
    )
    df["date"] = df["transaction_date"].dt.date

    # Compute margin for each product
    product_info = products[["product_id", "category", "base_price", "cost_price"]].copy()
    product_info["margin_pct"] = (
        (product_info["base_price"] - product_info["cost_price"])
        / product_info["base_price"] * 100
    ).round(2)

    # --- Per-product arc elasticity ---
    # Bucket discounts: 0, and each distinct non-zero level
    df["discount_bucket"] = df["discount_pct"].round(0).astype(int)

    # Average qty per item at each discount bucket (more stable than daily avg)
    item_avg = df.groupby(["product_id", "discount_bucket"]).agg(
        avg_qty=("quantity", "mean"),
        n_obs=("item_id", "count"),
    ).reset_index()

    product_arc = {}  # raw per-product arc estimates
    product_disc_obs = {}  # observation count at discount level used

    for pid, grp in item_avg.groupby("product_id"):
        zero_row = grp[grp["discount_bucket"] == 0]
        nonzero = grp[grp["discount_bucket"] > 0]

        if len(zero_row) == 0 or len(nonzero) == 0:
            continue
        if zero_row["n_obs"].values[0] < 30:
            continue

        # Use highest discount bucket with at least 5 observations
        qualified = nonzero[nonzero["n_obs"] >= 5].sort_values(
            "discount_bucket", ascending=False
        )
        if len(qualified) == 0:
            continue

        q0 = zero_row["avg_qty"].values[0]
        max_disc_row = qualified.iloc[0]
        q1 = max_disc_row["avg_qty"]
        d = max_disc_row["discount_bucket"] / 100.0

        if q0 > 0 and d > 0:
            pct_dq = (q1 - q0) / ((q1 + q0) / 2)
            pct_dp = -d / (1 - d / 2)  # arc price change
            elasticity = pct_dq / pct_dp if abs(pct_dp) > 1e-6 else 0.0
            product_arc[pid] = elasticity
            product_disc_obs[pid] = int(max_disc_row["n_obs"])

    # --- Category-level OLS log-log regression ---
    # Use log(1 - discount/100) as price regressor to isolate discount effect
    # from cross-product base-price variation
    df["price_ratio"] = (1 - df["discount_pct"] / 100).clip(lower=0.01)
    df["log_price_ratio"] = np.log(df["price_ratio"])
    df["log_qty"] = np.log(df["quantity"].clip(lower=1))

    category_elasticity = {}
    for cat, cat_df in df.groupby("category"):
        if len(cat_df) < 50:
            continue
        x = cat_df["log_price_ratio"].values
        y = cat_df["log_qty"].values
        # OLS: log(qty) = a + b*log(price_ratio) → b is the elasticity
        x_mean = x.mean()
        y_mean = y.mean()
        ss_xy = ((x - x_mean) * (y - y_mean)).sum()
        ss_xx = ((x - x_mean) ** 2).sum()
        if ss_xx > 0:
            b = ss_xy / ss_xx
            category_elasticity[cat] = b

    print(f"  Category-level elasticity (OLS log-log):")
    for cat, e in sorted(category_elasticity.items()):
        print(f"    {cat:<15} {e:+.3f}")

    # --- Assemble final elasticity table ---
    # Blend per-product arc with category OLS: weight by discount observation count
    # Products with <30 discounted observations use pure category estimate;
    # products with 100+ observations can fully override.
    BLEND_MIN = 30
    BLEND_MAX = 100

    rows = []
    for _, p in product_info.iterrows():
        pid = p["product_id"]
        cat_e = category_elasticity.get(p["category"], -0.1)

        if pid in product_arc:
            n_obs = product_disc_obs[pid]
            w = min(max((n_obs - BLEND_MIN) / (BLEND_MAX - BLEND_MIN), 0.0), 1.0)
            e = w * product_arc[pid] + (1 - w) * cat_e
        else:
            e = cat_e

        # Cap |elasticity| at 5.0
        e = max(min(e, 5.0), -5.0)
        abs_e = abs(e)

        if abs_e > 1.5:
            label = "high"
        elif abs_e >= 1.0:
            label = "medium"
        else:
            label = "low"

        rows.append({
            "product_id": pid,
            "category": p["category"],
            "base_price": p["base_price"],
            "cost_price": p["cost_price"],
            "margin_pct": p["margin_pct"],
            "elasticity_final": round(e, 4),
            "elasticity_label": label,
        })

    elasticity_df = pd.DataFrame(rows)

    # Summary
    label_counts = elasticity_df["elasticity_label"].value_counts()
    print(f"\n  Elasticity estimates for {len(elasticity_df):,} products:")
    for label in ["high", "medium", "low"]:
        count = label_counts.get(label, 0)
        print(f"    {label:<8} {count:>5,}")

    cat_avg = elasticity_df.groupby("category")["elasticity_final"].mean()
    print(f"\n  Average elasticity by category:")
    for cat, avg_e in sorted(cat_avg.items(), key=lambda x: x[1]):
        print(f"    {cat:<15} {avg_e:+.3f}")

    return elasticity_df


# ── Step 3: Discount Sensitivity ─────────────────────────────────────────────

def analyze_discount_sensitivity(elasticity_df, items, transactions):
    """Measure how profit responds at each discount level per category."""
    print("\nAnalyzing discount sensitivity...")

    # Join items → transactions → product info from elasticity_df
    df = items.merge(
        transactions[["transaction_id", "transaction_date"]], on="transaction_id"
    )
    df = df.merge(
        elasticity_df[["product_id", "category", "cost_price"]],
        on="product_id",
    )

    # Compute per-item financials
    df["revenue"] = df["unit_price"] * df["quantity"]
    df["profit"] = df["revenue"] - df["cost_price"] * df["quantity"]
    df["discount_bucket"] = df["discount_pct"].round(0).astype(int)

    # Group by category × discount bucket — use per-item averages for fair comparison
    grouped = df.groupby(["category", "discount_bucket"]).agg(
        avg_qty=("quantity", "mean"),
        avg_revenue=("revenue", "mean"),
        avg_profit=("profit", "mean"),
        total_revenue=("revenue", "sum"),
        total_profit=("profit", "sum"),
        n_items=("item_id", "count"),
    ).reset_index()
    grouped["profit_margin"] = (
        grouped["total_profit"] / grouped["total_revenue"].clip(lower=1) * 100
    ).round(2)

    # Compute uplift indices relative to zero-discount baseline (per-item basis)
    rows = []
    for cat, cat_grp in grouped.groupby("category"):
        baseline = cat_grp[cat_grp["discount_bucket"] == 0]
        if len(baseline) == 0:
            continue
        base_avg_profit = baseline["avg_profit"].values[0]
        base_qty = baseline["avg_qty"].values[0]

        for _, row in cat_grp.iterrows():
            profit_index = (
                row["avg_profit"] / base_avg_profit * 100
                if base_avg_profit > 0 else 100
            )
            qty_uplift = (
                (row["avg_qty"] - base_qty) / base_qty * 100
                if base_qty > 0 else 0
            )
            rows.append({
                "category": cat,
                "discount_pct": row["discount_bucket"],
                "profit_index": round(profit_index, 1),
                "qty_uplift_pct": round(qty_uplift, 1),
            })

    sensitivity_df = pd.DataFrame(rows)

    # Identify profit-maximizing discount per category
    print(f"\n  Profit-maximizing discount per category:")
    for cat, cat_grp in sensitivity_df.groupby("category"):
        best = cat_grp.loc[cat_grp["profit_index"].idxmax()]
        print(f"    {cat:<15} {best['discount_pct']:>3}%  "
              f"(profit index={best['profit_index']:.1f}, "
              f"qty uplift={best['qty_uplift_pct']:+.1f}%)")

    print(f"\n  Discount sensitivity: {len(sensitivity_df)} rows")
    return sensitivity_df


# ── Step 4: Optimal Discount Depth ───────────────────────────────────────────

def optimize_discount_depth(elasticity_df, sensitivity_df):
    """Find optimal discount per product that maximizes expected profit."""
    print("\nOptimizing discount depth...")

    rows = []
    for _, p in elasticity_df.iterrows():
        base_price = p["base_price"]
        cost_price = p["cost_price"]
        elasticity = abs(p["elasticity_final"])
        margin = base_price - cost_price

        if margin <= 0 or base_price <= 0:
            rows.append({
                "product_id": p["product_id"],
                "category": p["category"],
                "optimal_discount_pct": 0.0,
                "profit_uplift_pct": 0.0,
                "revenue_uplift_pct": 0.0,
                "margin_at_optimal_pct": round(margin / base_price * 100, 2)
                    if base_price > 0 else 0.0,
                "recommendation": "avoid_discount",
            })
            continue

        baseline_qty = 1.0  # normalized

        # Profit function: (base_price*(1-d) - cost_price) * baseline_qty * (1 + |e|*d)
        def neg_profit(d):
            price_at_d = base_price * (1 - d)
            qty_at_d = baseline_qty * (1 + elasticity * d)
            profit = (price_at_d - cost_price) * qty_at_d
            return -profit

        # Baseline profit (d=0)
        profit_at_zero = margin * baseline_qty

        # Cap discount so margin never drops below 2% of revenue
        # base_price*(1-d) - cost_price >= 0.02 * base_price*(1-d)
        # 0.98 * base_price*(1-d) >= cost_price
        # (1-d) >= cost_price / (0.98 * base_price)
        min_price_ratio = cost_price / (0.98 * base_price)
        max_d = min(0.40, max(0, 1 - min_price_ratio))

        if max_d < 0.005:
            # No room for discount
            rows.append({
                "product_id": p["product_id"],
                "category": p["category"],
                "optimal_discount_pct": 0.0,
                "profit_uplift_pct": 0.0,
                "revenue_uplift_pct": 0.0,
                "margin_at_optimal_pct": round(margin / base_price * 100, 2),
                "recommendation": "hold_price",
            })
            continue

        result = minimize_scalar(neg_profit, bounds=(0, max_d), method="bounded")
        optimal_d = result.x

        # Compute metrics at optimal discount
        price_opt = base_price * (1 - optimal_d)
        qty_opt = baseline_qty * (1 + elasticity * optimal_d)
        profit_opt = (price_opt - cost_price) * qty_opt
        revenue_opt = price_opt * qty_opt
        revenue_base = base_price * baseline_qty

        profit_uplift = (
            (profit_opt - profit_at_zero) / profit_at_zero * 100
            if profit_at_zero > 0 else 0
        )
        revenue_uplift = (
            (revenue_opt - revenue_base) / revenue_base * 100
            if revenue_base > 0 else 0
        )
        margin_at_opt = (price_opt - cost_price) / price_opt * 100 if price_opt > 0 else 0

        # Classify recommendation
        if optimal_d >= 0.15 and profit_uplift > 5:
            recommendation = "strong_promote"
        elif optimal_d >= 0.05 and profit_uplift > 1:
            recommendation = "moderate_promote"
        elif optimal_d < 0.05:
            recommendation = "hold_price"
        else:
            recommendation = "avoid_discount"

        rows.append({
            "product_id": p["product_id"],
            "category": p["category"],
            "optimal_discount_pct": round(optimal_d * 100, 2),
            "profit_uplift_pct": round(profit_uplift, 2),
            "revenue_uplift_pct": round(revenue_uplift, 2),
            "margin_at_optimal_pct": round(margin_at_opt, 2),
            "recommendation": recommendation,
        })

    optimal_df = pd.DataFrame(rows)

    # Summary
    rec_counts = optimal_df["recommendation"].value_counts()
    print(f"\n  Optimization results for {len(optimal_df):,} products:")
    for rec in ["strong_promote", "moderate_promote", "hold_price", "avoid_discount"]:
        count = rec_counts.get(rec, 0)
        print(f"    {rec:<20} {count:>5,}")

    cat_avg = optimal_df.groupby("category").agg(
        avg_discount=("optimal_discount_pct", "mean"),
        avg_uplift=("profit_uplift_pct", "mean"),
    ).round(2)
    print(f"\n  Average optimal discount by category:")
    for cat, row in cat_avg.iterrows():
        print(f"    {cat:<15} {row['avg_discount']:>5.1f}%  "
              f"(avg profit uplift={row['avg_uplift']:+.1f}%)")

    return optimal_df


# ── Step 5: Promotion Timing ────────────────────────────────────────────────

def identify_promotion_timing(items, transactions, products):
    """Find best months for promotions per category based on demand seasonality."""
    print("\nIdentifying promotion timing windows...")

    # Join items → transactions → products
    df = items.merge(
        transactions[["transaction_id", "transaction_date"]], on="transaction_id"
    )
    df = df.merge(
        products[["product_id", "category"]], on="product_id"
    )
    df["month"] = df["transaction_date"].dt.month

    # Monthly demand per category
    monthly = df.groupby(["category", "month"]).agg(
        total_qty=("quantity", "sum"),
    ).reset_index()

    # Demand index: 100 = average month for that category
    cat_avg = monthly.groupby("category")["total_qty"].mean().rename("cat_avg_qty")
    monthly = monthly.merge(cat_avg, on="category")
    monthly["demand_index"] = (
        monthly["total_qty"] / monthly["cat_avg_qty"] * 100
    ).round(1)

    # Promotion score and recommendation
    def timing_recommendation(idx):
        if idx < 85:
            return "Strong promotion window"
        elif idx < 95:
            return "Good promotion window"
        elif idx <= 110:
            return "Neutral"
        else:
            return "Avoid — high organic demand"

    def promotion_score(idx):
        # Higher score = better opportunity for promotion
        if idx < 85:
            return 5
        elif idx < 95:
            return 4
        elif idx <= 110:
            return 3
        else:
            return 1

    monthly["timing_recommendation"] = monthly["demand_index"].apply(timing_recommendation)
    monthly["promotion_score"] = monthly["demand_index"].apply(promotion_score)

    timing_df = monthly[["category", "month", "demand_index",
                          "promotion_score", "timing_recommendation"]].copy()

    # Print summary
    print(f"\n  Promotion timing: {len(timing_df)} rows")
    print(f"\n  Strong promotion windows (demand index < 85):")
    strong = timing_df[timing_df["demand_index"] < 85].sort_values("demand_index")
    if len(strong) > 0:
        for _, row in strong.head(10).iterrows():
            print(f"    {row['category']:<15} Month {row['month']:>2}  "
                  f"(demand index={row['demand_index']:.1f})")
    else:
        print("    None found")

    print(f"\n  Avoid windows (demand index > 110):")
    avoid = timing_df[timing_df["demand_index"] > 110].sort_values(
        "demand_index", ascending=False
    )
    if len(avoid) > 0:
        for _, row in avoid.head(10).iterrows():
            print(f"    {row['category']:<15} Month {row['month']:>2}  "
                  f"(demand index={row['demand_index']:.1f})")
    else:
        print("    None found")

    return timing_df


# ── Step 6: Cannibalization Risk ─────────────────────────────────────────────

def assess_cannibalization_risk(items, transactions, products, promotions):
    """Measure whether promoting a product hurts same-category peers."""
    print("\nAssessing cannibalization risk...")

    # Join items → transactions → products
    df = items.merge(
        transactions[["transaction_id", "transaction_date"]], on="transaction_id"
    )
    df = df.merge(
        products[["product_id", "category", "subcategory"]], on="product_id"
    )

    rows = []

    for _, promo in promotions.iterrows():
        promo_start = promo["start_date"]
        promo_end = promo["end_date"]
        pre_start = promo_start - pd.Timedelta(days=14)
        pre_end = promo_start - pd.Timedelta(days=1)

        # Determine promoted products and category
        if pd.notna(promo.get("product_id")) and promo["product_id"] > 0:
            promoted_pid = int(promo["product_id"])
            promo_cat = products.loc[
                products["product_id"] == promoted_pid, "category"
            ].values
            if len(promo_cat) == 0:
                continue
            promo_cat = promo_cat[0]
            promoted_pids = {promoted_pid}
        elif pd.notna(promo.get("category")) and promo["category"]:
            promo_cat = promo["category"]
            # Category-level promo: all products in category are promoted
            promoted_pids = set(
                products.loc[products["category"] == promo_cat, "product_id"]
            )
        else:
            continue

        # Non-promoted products in same category
        cat_products = set(
            products.loc[products["category"] == promo_cat, "product_id"]
        )
        non_promoted = cat_products - promoted_pids
        if len(non_promoted) == 0:
            continue

        # Baseline period sales for non-promoted products
        baseline_sales = df[
            (df["product_id"].isin(non_promoted))
            & (df["transaction_date"] >= pre_start)
            & (df["transaction_date"] <= pre_end)
        ].groupby("product_id")["quantity"].sum()

        # Promo period sales for non-promoted products
        promo_sales = df[
            (df["product_id"].isin(non_promoted))
            & (df["transaction_date"] >= promo_start)
            & (df["transaction_date"] <= promo_end)
        ].groupby("product_id")["quantity"].sum()

        # Normalize by period length
        baseline_days = max((pre_end - pre_start).days + 1, 1)
        promo_days = max((promo_end - promo_start).days + 1, 1)

        for pid in non_promoted:
            b_qty = baseline_sales.get(pid, 0) / baseline_days
            p_qty = promo_sales.get(pid, 0) / promo_days

            if b_qty > 0:
                cannib_pct = (b_qty - p_qty) / b_qty * 100
            else:
                cannib_pct = 0.0

            subcat = products.loc[
                products["product_id"] == pid, "subcategory"
            ].values[0]
            rows.append({
                "category": promo_cat,
                "subcategory": subcat,
                "cannibalization_pct": cannib_pct,
            })

    if len(rows) == 0:
        print("  No cannibalization data available")
        return pd.DataFrame(columns=[
            "category", "subcategory", "avg_cannibalization_pct", "risk_label"
        ])

    raw = pd.DataFrame(rows)

    # Aggregate to category × subcategory level
    cannib_df = raw.groupby(["category", "subcategory"]).agg(
        avg_cannibalization_pct=("cannibalization_pct", "mean"),
    ).reset_index()
    cannib_df["avg_cannibalization_pct"] = cannib_df["avg_cannibalization_pct"].round(2)

    def risk_label(pct):
        if pct > 10:
            return "High"
        elif pct > 5:
            return "Medium"
        elif pct >= 0:
            return "Low"
        else:
            return "Halo effect"

    cannib_df["risk_label"] = cannib_df["avg_cannibalization_pct"].apply(risk_label)

    # Summary
    label_counts = cannib_df["risk_label"].value_counts()
    print(f"\n  Cannibalization risk: {len(cannib_df)} category/subcategory pairs")
    for label in ["High", "Medium", "Low", "Halo effect"]:
        count = label_counts.get(label, 0)
        print(f"    {label:<15} {count:>3}")

    return cannib_df


# ── Save Outputs ─────────────────────────────────────────────────────────────

def save_outputs(elasticity_df, sensitivity_df, optimal_df, timing_df, cannib_df):
    """Save all CSV outputs."""
    print("\nSaving outputs...")

    path = os.path.join(OUTPUT_DIR, "price_elasticity.csv")
    elasticity_df.to_csv(path, index=False)
    print(f"  Saved price_elasticity.csv: {len(elasticity_df):,} rows")

    path = os.path.join(OUTPUT_DIR, "discount_sensitivity.csv")
    sensitivity_df.to_csv(path, index=False)
    print(f"  Saved discount_sensitivity.csv: {len(sensitivity_df)} rows")

    path = os.path.join(OUTPUT_DIR, "optimal_discounts.csv")
    optimal_df.to_csv(path, index=False)
    print(f"  Saved optimal_discounts.csv: {len(optimal_df):,} rows")

    path = os.path.join(OUTPUT_DIR, "promotion_timing.csv")
    timing_df.to_csv(path, index=False)
    print(f"  Saved promotion_timing.csv: {len(timing_df)} rows")

    path = os.path.join(OUTPUT_DIR, "cannibalization_risk.csv")
    cannib_df.to_csv(path, index=False)
    print(f"  Saved cannibalization_risk.csv: {len(cannib_df)} rows")

    # Margin impact summary (one row per category)
    summary_rows = []
    for cat in sorted(elasticity_df["category"].unique()):
        cat_elast = elasticity_df[elasticity_df["category"] == cat]
        cat_opt = optimal_df[optimal_df["category"] == cat]

        avg_current_margin = cat_elast["margin_pct"].mean()
        avg_optimal_discount = cat_opt["optimal_discount_pct"].mean()
        avg_new_margin = cat_opt["margin_at_optimal_pct"].mean()
        avg_profit_uplift = cat_opt["profit_uplift_pct"].mean()

        rec_counts = cat_opt["recommendation"].value_counts()
        top_rec = rec_counts.index[0] if len(rec_counts) > 0 else "hold_price"

        summary_rows.append({
            "category": cat,
            "avg_current_margin": round(avg_current_margin, 2),
            "avg_optimal_discount": round(avg_optimal_discount, 2),
            "avg_new_margin": round(avg_new_margin, 2),
            "avg_profit_uplift": round(avg_profit_uplift, 2),
            "top_recommendation": top_rec,
        })

    summary_df = pd.DataFrame(summary_rows)
    path = os.path.join(OUTPUT_DIR, "margin_impact_summary.csv")
    summary_df.to_csv(path, index=False)
    print(f"  Saved margin_impact_summary.csv: {len(summary_df)} rows")

    # Print summary table
    print("\n  ┌─ Margin Impact Summary ──────────────────────────────────────────────┐")
    print(f"  │ {'Category':<15} {'Cur Margin':>10} {'Opt Disc':>9} "
          f"{'New Margin':>10} {'Profit ↑':>9} {'Recommendation':<20} │")
    print(f"  ├{'─' * 77}┤")
    for _, row in summary_df.iterrows():
        print(f"  │ {row['category']:<15} {row['avg_current_margin']:>9.1f}% "
              f"{row['avg_optimal_discount']:>8.1f}% "
              f"{row['avg_new_margin']:>9.1f}% "
              f"{row['avg_profit_uplift']:>8.1f}% "
              f"{row['top_recommendation']:<20} │")
    print(f"  └{'─' * 77}┘")

    return summary_df


# ── Visualizations ──────────────────────────────────────────────────────────

def plot_elasticity_by_category(elasticity_df):
    """Box plot of elasticity distribution per category."""
    categories = sorted(elasticity_df["category"].unique())
    data = [
        elasticity_df[elasticity_df["category"] == cat]["elasticity_final"].values
        for cat in categories
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, tick_labels=categories, patch_artist=True)
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors[:len(categories)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=-1.0, color="#95a5a6", linestyle="--", linewidth=0.8,
               label="|e| = 1.0 (unit elastic)")
    ax.axhline(y=-1.5, color="#e74c3c", linestyle="--", linewidth=0.8,
               label="|e| = 1.5 (high elastic)")
    ax.set_xlabel("Category")
    ax.set_ylabel("Price Elasticity")
    ax.set_title("Price Elasticity Distribution by Category")
    ax.legend(loc="lower left")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "elasticity_by_category.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_discount_response_curves(sensitivity_df):
    """Profit index vs discount % per category (line plot)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    for i, (cat, grp) in enumerate(sorted(sensitivity_df.groupby("category"))):
        grp = grp.sort_values("discount_pct")
        ax.plot(grp["discount_pct"], grp["profit_index"], "o-",
                color=colors[i % len(colors)], label=cat, linewidth=2, markersize=5)

    ax.axhline(y=100, color="#95a5a6", linestyle="--", linewidth=0.8,
               label="Baseline (no discount)")
    ax.set_xlabel("Discount (%)")
    ax.set_ylabel("Profit Index (100 = baseline)")
    ax.set_title("Profit Response to Discount Depth by Category")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "discount_response_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_optimal_discount_scatter(optimal_df):
    """Optimal discount vs profit uplift per product (scatter, colored by category)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Beauty": "#9b59b6", "Clothing": "#3498db", "Electronics": "#e74c3c",
              "Groceries": "#2ecc71", "Home": "#f39c12"}

    for cat, grp in sorted(optimal_df.groupby("category")):
        ax.scatter(grp["optimal_discount_pct"], grp["profit_uplift_pct"],
                   c=colors.get(cat, "#95a5a6"), label=cat, alpha=0.5, s=20)

    ax.set_xlabel("Optimal Discount (%)")
    ax.set_ylabel("Profit Uplift (%)")
    ax.set_title("Optimal Discount vs Profit Uplift by Product")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "optimal_discount_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_demand_seasonality_heatmap(timing_df):
    """Category × month demand index heatmap."""
    pivot = timing_df.pivot(index="category", columns="month",
                            values="demand_index").fillna(100)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=70, vmax=130)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([month_labels[m - 1] for m in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > 115 or val < 80 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="Demand Index (100 = avg)")
    ax.set_title("Demand Seasonality by Category — Promotion Timing Guide")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "demand_seasonality_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_cannibalization_network(cannib_df):
    """Cannibalization % by category/subcategory (grouped bar chart)."""
    if len(cannib_df) == 0:
        print("  Skipped cannibalization_network.png (no data)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    categories = sorted(cannib_df["category"].unique())
    colors = {"Beauty": "#9b59b6", "Clothing": "#3498db", "Electronics": "#e74c3c",
              "Groceries": "#2ecc71", "Home": "#f39c12"}

    # Sort by category then cannibalization %
    plot_df = cannib_df.sort_values(["category", "avg_cannibalization_pct"],
                                     ascending=[True, False])
    labels = [f"{row['category']}\n{row['subcategory']}" for _, row in plot_df.iterrows()]
    x = np.arange(len(labels))
    bar_colors = [colors.get(row["category"], "#95a5a6") for _, row in plot_df.iterrows()]

    ax.bar(x, plot_df["avg_cannibalization_pct"].values, color=bar_colors, alpha=0.8)

    # Risk thresholds
    ax.axhline(y=10, color="#e74c3c", linestyle="--", linewidth=0.8, label="High risk (>10%)")
    ax.axhline(y=5, color="#f39c12", linestyle="--", linewidth=0.8, label="Medium risk (>5%)")
    ax.axhline(y=0, color="#95a5a6", linestyle="-", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Avg Cannibalization (%)")
    ax.set_title("Cannibalization Risk by Category / Subcategory")
    ax.legend(loc="upper right")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "cannibalization_network.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load data
    products, promotions, items, transactions = load_data()

    # Step 2: Price elasticity
    elasticity_df = compute_price_elasticity(products, promotions, items, transactions)

    # Step 3: Discount sensitivity
    sensitivity_df = analyze_discount_sensitivity(elasticity_df, items, transactions)

    # Step 4: Optimal discount depth
    optimal_df = optimize_discount_depth(elasticity_df, sensitivity_df)

    # Step 5: Promotion timing
    timing_df = identify_promotion_timing(items, transactions, products)

    # Step 6: Cannibalization risk
    cannib_df = assess_cannibalization_risk(items, transactions, products, promotions)

    # Save CSVs
    save_outputs(elasticity_df, sensitivity_df, optimal_df, timing_df, cannib_df)

    # Save visualizations
    print("\nGenerating visualizations...")
    plot_elasticity_by_category(elasticity_df)
    plot_discount_response_curves(sensitivity_df)
    plot_optimal_discount_scatter(optimal_df)
    plot_demand_seasonality_heatmap(timing_df)
    plot_cannibalization_network(cannib_df)

    print("\nDone! All outputs saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()

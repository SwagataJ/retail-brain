"""
Retail Brain — Customer Intelligence Pipeline

Generates customer segmentation, churn predictions, category affinity analysis,
and re-engagement recommendations from transaction data.

Outputs (in data/customer_intelligence/):
  CSVs:  rfm_scores.csv, segment_summary.csv, category_affinity.csv,
         churn_predictions.csv, recommendations.csv
  PNGs:  segment_distribution.png, rfm_heatmap.png,
         churn_feature_importance.png, category_affinity_heatmap.png

Usage:
    source retail_brain_env/bin/activate
    python scripts/customer_intelligence.py
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ───────────────────────────────────────────────────────────────────
SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "customer_intelligence")
REFERENCE_DATE = pd.Timestamp("2026-01-01")
H1_START = pd.Timestamp("2025-01-01")
H1_END = pd.Timestamp("2025-06-30")
H2_START = pd.Timestamp("2025-07-01")
H2_END = pd.Timestamp("2025-12-31")

np.random.seed(SEED)


# ── Step 1: Load Data ────────────────────────────────────────────────────────

def load_data():
    """Load the four input CSVs."""
    print("Loading data...")
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"),
                            parse_dates=["signup_date"])
    transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"),
                               parse_dates=["transaction_date"])
    items = pd.read_csv(os.path.join(DATA_DIR, "transaction_items.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))

    print(f"  Customers:         {len(customers):,}")
    print(f"  Transactions:      {len(transactions):,}")
    print(f"  Transaction items: {len(items):,}")
    print(f"  Products:          {len(products):,}")
    return customers, transactions, items, products


# ── Step 2: RFM Scores ──────────────────────────────────────────────────────

def compute_rfm(customers, transactions):
    """Compute Recency, Frequency, Monetary, ABV and quintile scores."""
    print("\nComputing RFM scores...")

    rfm = transactions.groupby("customer_id").agg(
        last_purchase=("transaction_date", "max"),
        frequency=("transaction_id", "count"),
        monetary=("total_amount", "sum"),
    ).reset_index()

    rfm["recency"] = (REFERENCE_DATE - rfm["last_purchase"]).dt.days
    rfm["abv"] = rfm["monetary"] / rfm["frequency"]

    # Quintile scores (1=best for recency, 5=best for frequency/monetary)
    rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5,
                             labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5,
                             labels=[1, 2, 3, 4, 5]).astype(int)

    # Merge back to full customer list (customers with 0 transactions get defaults)
    result = customers[["customer_id"]].merge(rfm, on="customer_id", how="left")
    result["recency"] = result["recency"].fillna(
        (REFERENCE_DATE - customers.set_index("customer_id").loc[
            result.loc[result["recency"].isna(), "customer_id"], "signup_date"
        ].values).days if result["recency"].isna().any() else 0
    )

    # For customers with no transactions, fill defaults
    no_txn_mask = result["frequency"].isna()
    if no_txn_mask.any():
        result.loc[no_txn_mask, "frequency"] = 0
        result.loc[no_txn_mask, "monetary"] = 0.0
        result.loc[no_txn_mask, "abv"] = 0.0
        result.loc[no_txn_mask, "r_score"] = 1
        result.loc[no_txn_mask, "f_score"] = 1
        result.loc[no_txn_mask, "m_score"] = 1
        # Recency for no-txn customers: days since signup
        no_txn_ids = result.loc[no_txn_mask, "customer_id"]
        signup_lookup = customers.set_index("customer_id")["signup_date"]
        for cid in no_txn_ids:
            result.loc[result["customer_id"] == cid, "recency"] = \
                (REFERENCE_DATE - signup_lookup[cid]).days

    result.drop(columns=["last_purchase"], inplace=True, errors="ignore")
    print(f"  RFM computed for {len(result):,} customers")
    return result


# ── Step 3: Behavioral Segments ──────────────────────────────────────────────

def assign_segments(rfm, transactions, customers):
    """Assign mutually exclusive behavioral segments in priority order."""
    print("\nAssigning behavioral segments...")

    abv_75th = rfm.loc[rfm["abv"] > 0, "abv"].quantile(0.75)

    # H1 vs H2 frequency per customer
    h1_freq = transactions[
        transactions["transaction_date"].between(H1_START, H1_END)
    ].groupby("customer_id")["transaction_id"].count().rename("h1_freq")

    h2_freq = transactions[
        transactions["transaction_date"].between(H2_START, H2_END)
    ].groupby("customer_id")["transaction_id"].count().rename("h2_freq")

    rfm = rfm.merge(h1_freq, on="customer_id", how="left")
    rfm = rfm.merge(h2_freq, on="customer_id", how="left")
    rfm["h1_freq"] = rfm["h1_freq"].fillna(0).astype(int)
    rfm["h2_freq"] = rfm["h2_freq"].fillna(0).astype(int)

    # Tenure in days
    signup_lookup = customers.set_index("customer_id")["signup_date"]
    rfm = rfm.merge(
        customers[["customer_id", "signup_date"]], on="customer_id", how="left"
    )
    rfm["tenure_days"] = (REFERENCE_DATE - rfm["signup_date"]).dt.days

    # Assign segments in priority order
    conditions = [
        # 1. High-Value Lapsed: top 25% ABV + 180+ days inactive
        (rfm["abv"] >= abv_75th) & (rfm["recency"] >= 180),
        # 2. At-Risk: H1→H2 frequency declined 50%+
        (rfm["h1_freq"] > 0) & (rfm["h2_freq"] <= rfm["h1_freq"] * 0.5),
        # 3. New: first purchase within 30 days of reference date
        (rfm["recency"] <= 30) & (rfm["tenure_days"] <= 30),
        # 4. Loyal: 5+ purchases
        rfm["frequency"] >= 5,
        # 5. Repeat: 2-5 purchases, active within 90 days
        (rfm["frequency"].between(2, 5)) & (rfm["recency"] <= 90),
    ]
    choices = ["High-Value Lapsed", "At-Risk", "New", "Loyal", "Repeat"]

    rfm["segment"] = np.select(conditions, choices, default="Inactive")

    # Print segment distribution
    seg_counts = rfm["segment"].value_counts()
    print("  Segment distribution:")
    for seg, count in seg_counts.items():
        print(f"    {seg:<22} {count:>7,}  ({count / len(rfm) * 100:.1f}%)")

    return rfm


# ── Step 4: Churn Prediction ────────────────────────────────────────────────

def predict_churn(rfm):
    """Train Random Forest churn model and score all customers."""
    print("\nTraining churn prediction model...")

    # Feature engineering
    rfm["freq_trend"] = rfm["h2_freq"] - rfm["h1_freq"]

    # Average days between purchases (for customers with 2+ purchases)
    rfm["avg_days_between"] = np.where(
        rfm["frequency"] >= 2,
        rfm["tenure_days"] / rfm["frequency"],
        rfm["tenure_days"],
    )

    feature_cols = [
        "r_score", "f_score", "m_score",
        "tenure_days", "h1_freq", "h2_freq",
        "freq_trend", "avg_days_between",
    ]

    # Label: churned if recency >= 180 days
    rfm["churn_label"] = (rfm["recency"] >= 180).astype(int)

    # Filter to customers with at least 1 transaction for meaningful prediction
    model_df = rfm[rfm["frequency"] > 0].copy()
    X = model_df[feature_cols]
    y = model_df["churn_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print("\n  Classification Report (test set):")
    report = classification_report(y_test, y_pred)
    for line in report.strip().split("\n"):
        print(f"    {line}")
    print(f"\n  ROC AUC: {auc:.4f}")

    # Score all customers
    all_proba = clf.predict_proba(rfm[feature_cols].fillna(0))[:, 1]
    rfm["churn_probability"] = all_proba
    rfm["churn_risk"] = pd.cut(
        rfm["churn_probability"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    print(f"\n  Risk tier distribution:")
    risk_counts = rfm["churn_risk"].value_counts().sort_index()
    for tier, count in risk_counts.items():
        print(f"    {tier:<8} {count:>7,}  ({count / len(rfm) * 100:.1f}%)")

    return rfm, clf, feature_cols


# ── Step 5: Category Affinity ───────────────────────────────────────────────

def compute_category_affinity(rfm, transactions, items, products):
    """Compute share-of-wallet per segment x category."""
    print("\nComputing category affinity...")

    # Join items → transactions → products to get segment + category + spend
    txn_items = items.merge(
        transactions[["transaction_id", "customer_id"]], on="transaction_id"
    )
    txn_items = txn_items.merge(
        products[["product_id", "category"]], on="product_id"
    )
    txn_items["line_total"] = txn_items["unit_price"] * txn_items["quantity"]

    # Add segment
    txn_items = txn_items.merge(
        rfm[["customer_id", "segment"]], on="customer_id"
    )

    # Share of wallet: spend per segment x category / total spend per segment
    seg_cat_spend = txn_items.groupby(["segment", "category"])["line_total"].sum()
    seg_total = txn_items.groupby("segment")["line_total"].sum()

    affinity = (seg_cat_spend / seg_total).reset_index()
    affinity.columns = ["segment", "category", "share_of_wallet"]
    affinity["share_of_wallet"] = (affinity["share_of_wallet"] * 100).round(2)

    # Pivot for display
    pivot = affinity.pivot(index="segment", columns="category",
                           values="share_of_wallet").fillna(0)
    print("\n  Category Affinity (% share of wallet):")
    print(pivot.to_string(float_format=lambda x: f"{x:.1f}%").replace(
        "\n", "\n  "
    ))

    return affinity, pivot


# ── Step 6: Recommendations ─────────────────────────────────────────────────

def generate_recommendations(affinity_pivot):
    """Generate per-segment timing and action recommendations."""
    print("\nGenerating re-engagement recommendations...")

    recs = {
        "New": {
            "timing": "Within 7 days of first purchase",
            "action_template": "Welcome offer: 10-15% discount on {top_cat}; "
                               "onboarding email series",
        },
        "Repeat": {
            "timing": "Every 21-30 days",
            "action_template": "Loyalty enrollment; bundle offers on "
                               "{top_cat} + {second_cat}",
        },
        "Loyal": {
            "timing": "Bi-weekly touchpoint",
            "action_template": "VIP early access to {top_cat}; "
                               "referral program; exclusive previews",
        },
        "High-Value Lapsed": {
            "timing": "Immediate (within 7 days)",
            "action_template": "20-25% win-back discount on {top_cat}; "
                               "personal outreach",
        },
        "At-Risk": {
            "timing": "Within 14 days",
            "action_template": "15% retention offer on {top_cat}; "
                               "feedback survey; personalized {second_cat} picks",
        },
        "Inactive": {
            "timing": "Quarterly batch",
            "action_template": "Low-cost seasonal email featuring {top_cat}; "
                               "re-activation coupon",
        },
    }

    rows = []
    for segment, cfg in recs.items():
        if segment in affinity_pivot.index:
            sorted_cats = affinity_pivot.loc[segment].sort_values(ascending=False)
            top_cat = sorted_cats.index[0]
            second_cat = sorted_cats.index[1] if len(sorted_cats) > 1 else top_cat
        else:
            top_cat, second_cat = "top category", "second category"

        action = cfg["action_template"].format(
            top_cat=top_cat, second_cat=second_cat
        )
        rows.append({
            "segment": segment,
            "recommended_timing": cfg["timing"],
            "recommended_action": action,
            "top_affinity_category": top_cat,
        })

    recs_df = pd.DataFrame(rows)

    print("\n  Recommendations:")
    for _, row in recs_df.iterrows():
        print(f"    {row['segment']:<22} | {row['recommended_timing']:<30} | "
              f"{row['recommended_action'][:60]}...")

    return recs_df


# ── Visualizations ──────────────────────────────────────────────────────────

def plot_segment_distribution(rfm):
    """Pie chart of segment sizes."""
    counts = rfm["segment"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12", "#95a5a6"]
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=colors[:len(counts)], startangle=140)
    ax.set_title("Customer Segment Distribution")
    path = os.path.join(OUTPUT_DIR, "segment_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_rfm_heatmap(rfm):
    """Heatmap of average RFM scores by segment."""
    seg_rfm = rfm.groupby("segment")[["r_score", "f_score", "m_score"]].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(seg_rfm.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(seg_rfm.columns)))
    ax.set_xticklabels(["Recency", "Frequency", "Monetary"])
    ax.set_yticks(range(len(seg_rfm.index)))
    ax.set_yticklabels(seg_rfm.index)
    for i in range(len(seg_rfm.index)):
        for j in range(len(seg_rfm.columns)):
            ax.text(j, i, f"{seg_rfm.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, label="Avg Score")
    ax.set_title("Average RFM Scores by Segment")
    path = os.path.join(OUTPUT_DIR, "rfm_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_churn_feature_importance(clf, feature_cols):
    """Bar chart of Random Forest feature importances."""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(feature_cols)),
            importances[indices],
            color="#3498db")
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title("Churn Model — Feature Importances")
    ax.invert_yaxis()
    path = os.path.join(OUTPUT_DIR, "churn_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_category_affinity_heatmap(affinity_pivot):
    """Heatmap of segment x category affinity."""
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(affinity_pivot.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(affinity_pivot.columns)))
    ax.set_xticklabels(affinity_pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(affinity_pivot.index)))
    ax.set_yticklabels(affinity_pivot.index)
    for i in range(len(affinity_pivot.index)):
        for j in range(len(affinity_pivot.columns)):
            ax.text(j, i, f"{affinity_pivot.values[i, j]:.1f}%",
                    ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Share of Wallet (%)")
    ax.set_title("Category Affinity by Segment (% Share of Wallet)")
    path = os.path.join(OUTPUT_DIR, "category_affinity_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Save Outputs ─────────────────────────────────────────────────────────────

def save_outputs(rfm, affinity, recs_df):
    """Save all CSV outputs."""
    print("\nSaving outputs...")

    # 1. RFM scores (per-customer)
    rfm_out = rfm[[
        "customer_id", "recency", "frequency", "monetary", "abv",
        "r_score", "f_score", "m_score", "segment", "churn_probability",
    ]].copy()
    rfm_out["monetary"] = rfm_out["monetary"].round(2)
    rfm_out["abv"] = rfm_out["abv"].round(2)
    rfm_out["churn_probability"] = rfm_out["churn_probability"].round(4)
    path = os.path.join(OUTPUT_DIR, "rfm_scores.csv")
    rfm_out.to_csv(path, index=False)
    print(f"  Saved rfm_scores.csv: {len(rfm_out):,} rows")

    # 2. Segment summary
    seg_summary = rfm.groupby("segment").agg(
        customer_count=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        avg_abv=("abv", "mean"),
        avg_churn_probability=("churn_probability", "mean"),
    ).round(2).reset_index()
    seg_summary["pct_of_total"] = (
        seg_summary["customer_count"] / seg_summary["customer_count"].sum() * 100
    ).round(1)
    path = os.path.join(OUTPUT_DIR, "segment_summary.csv")
    seg_summary.to_csv(path, index=False)
    print(f"  Saved segment_summary.csv: {len(seg_summary)} rows")

    # Print segment summary table
    print("\n  ┌─ Segment Summary ────────────────────────────────────────────┐")
    print(f"  │ {'Segment':<22} {'Count':>7} {'%':>6} {'Avg Rec':>8} "
          f"{'Avg Freq':>9} {'Avg $':>9} {'Churn%':>7} │")
    print(f"  ├{'─' * 72}┤")
    for _, row in seg_summary.iterrows():
        print(f"  │ {row['segment']:<22} {row['customer_count']:>7,} "
              f"{row['pct_of_total']:>5.1f}% {row['avg_recency']:>7.0f}d "
              f"{row['avg_frequency']:>8.1f} ${row['avg_monetary']:>8,.0f} "
              f"{row['avg_churn_probability'] * 100:>6.1f}% │")
    print(f"  └{'─' * 72}┘")

    # 3. Category affinity
    path = os.path.join(OUTPUT_DIR, "category_affinity.csv")
    affinity.to_csv(path, index=False)
    print(f"  Saved category_affinity.csv: {len(affinity)} rows")

    # 4. Churn predictions (per-customer)
    churn_out = rfm[[
        "customer_id", "churn_probability", "churn_risk", "segment",
    ]].copy()
    churn_out["churn_probability"] = churn_out["churn_probability"].round(4)
    path = os.path.join(OUTPUT_DIR, "churn_predictions.csv")
    churn_out.to_csv(path, index=False)
    print(f"  Saved churn_predictions.csv: {len(churn_out):,} rows")

    # 5. Recommendations
    path = os.path.join(OUTPUT_DIR, "recommendations.csv")
    recs_df.to_csv(path, index=False)
    print(f"  Saved recommendations.csv: {len(recs_df)} rows")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load data
    customers, transactions, items, products = load_data()

    # Step 2: RFM scores
    rfm = compute_rfm(customers, transactions)

    # Step 3: Behavioral segments
    rfm = assign_segments(rfm, transactions, customers)

    # Step 4: Churn prediction
    rfm, clf, feature_cols = predict_churn(rfm)

    # Step 5: Category affinity
    affinity, affinity_pivot = compute_category_affinity(
        rfm, transactions, items, products
    )

    # Step 6: Recommendations
    recs_df = generate_recommendations(affinity_pivot)

    # Save CSVs
    save_outputs(rfm, affinity, recs_df)

    # Save visualizations
    print("\nGenerating visualizations...")
    plot_segment_distribution(rfm)
    plot_rfm_heatmap(rfm)
    plot_churn_feature_importance(clf, feature_cols)
    plot_category_affinity_heatmap(affinity_pivot)

    print("\nDone! All outputs saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""
Retail Brain — Synthetic Data Generator

Generates 6 CSV files with realistic patterns for ML model development:
- products.csv       (~2,000 products)
- customers.csv      (~100,000 customers)
- stores.csv         (~100 stores)
- promotions.csv     (~50 promotions)
- transactions.csv   (~500,000 transactions over 2 years)
- transaction_items.csv (~1,200,000 line items)

Embedded patterns:
- Seasonality (Nov-Dec spike, Jan-Feb dip)
- 4 customer segments (high_value, regular, bargain_hunter, churned)
- Price elasticity (Groceries inelastic, Electronics elastic)
- Upward trend in online store sales
- Higher weekend sales
"""

import os
import numpy as np
import pandas as pd
from faker import Faker

# ── Config ───────────────────────────────────────────────────────────────────
SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

NUM_PRODUCTS = 2_000
NUM_CUSTOMERS = 100_000
NUM_STORES = 100
NUM_PROMOTIONS = 50
NUM_TRANSACTIONS = 500_000
AVG_ITEMS_PER_TXN = 2.4  # targets ~1.2M line items

DATE_START = pd.Timestamp("2024-01-01")
DATE_END = pd.Timestamp("2025-12-31")

np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# ── Helpers ──────────────────────────────────────────────────────────────────

CATEGORY_CONFIG = {
    "Electronics": {
        "subcategories": ["Laptops", "Smartphones", "Headphones", "Tablets", "Cameras"],
        "brands": ["TechNova", "PixelPro", "VoltEdge", "ClearView", "ByteWave"],
        "price_range": (50, 1500),
        "margin": (0.10, 0.25),
        "elasticity": "high",
    },
    "Clothing": {
        "subcategories": ["T-Shirts", "Jeans", "Dresses", "Sneakers", "Jackets"],
        "brands": ["UrbanThread", "FitForm", "StreetPulse", "CozyKnit", "LuxWear"],
        "price_range": (15, 200),
        "margin": (0.40, 0.60),
        "elasticity": "medium",
    },
    "Groceries": {
        "subcategories": ["Snacks", "Beverages", "Dairy", "Bakery", "Frozen"],
        "brands": ["FreshFarm", "NatureBite", "DailyHarvest", "PureBasics", "GreenLeaf"],
        "price_range": (2, 30),
        "margin": (0.15, 0.30),
        "elasticity": "low",
    },
    "Home": {
        "subcategories": ["Furniture", "Kitchenware", "Bedding", "Lighting", "Decor"],
        "brands": ["NestCraft", "HomeEase", "CozyDen", "BrightSpace", "TimberLux"],
        "price_range": (10, 500),
        "margin": (0.30, 0.50),
        "elasticity": "medium",
    },
    "Beauty": {
        "subcategories": ["Skincare", "Haircare", "Makeup", "Fragrance", "Shampoo"],
        "brands": ["GlowUp", "VelvetSkin", "PureRadiance", "BloomBeauty", "SilkEssence"],
        "price_range": (5, 120),
        "margin": (0.50, 0.70),
        "elasticity": "medium",
    },
}

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
    "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
    "Indianapolis", "Seattle", "Denver", "Boston", "Nashville",
]

STORE_NAMES_PREFIX = [
    "MegaMart", "ShopEasy", "ValueHub", "PrimeMall", "QuickStop",
    "CityCenter", "DailyNeeds", "TrendZone", "SmartBuy", "FreshMarket",
]

SEGMENT_DISTRIBUTION = {
    "high_value": 0.10,
    "regular": 0.45,
    "bargain_hunter": 0.25,
    "churned": 0.20,
}


# ── Generators ───────────────────────────────────────────────────────────────

def generate_products() -> pd.DataFrame:
    print("Generating products...")
    rows = []
    pid = 1
    products_per_cat = NUM_PRODUCTS // len(CATEGORY_CONFIG)

    for category, cfg in CATEGORY_CONFIG.items():
        for _ in range(products_per_cat):
            subcat = np.random.choice(cfg["subcategories"])
            brand = np.random.choice(cfg["brands"])
            base_price = round(np.random.uniform(*cfg["price_range"]), 2)
            margin = np.random.uniform(*cfg["margin"])
            cost_price = round(base_price * (1 - margin), 2)
            name = f"{brand} {subcat} {fake.lexify(text='??').upper()}-{pid}"
            rows.append({
                "product_id": pid,
                "product_name": name,
                "category": category,
                "subcategory": subcat,
                "base_price": base_price,
                "cost_price": cost_price,
                "brand": brand,
            })
            pid += 1

    return pd.DataFrame(rows)


def generate_customers() -> pd.DataFrame:
    print("Generating customers...")
    segments = np.random.choice(
        list(SEGMENT_DISTRIBUTION.keys()),
        size=NUM_CUSTOMERS,
        p=list(SEGMENT_DISTRIBUTION.values()),
    )

    signup_dates = []
    for seg in segments:
        if seg == "churned":
            # churned customers signed up early
            d = fake.date_between(start_date=DATE_START, end_date=DATE_START + pd.Timedelta(days=180))
        else:
            d = fake.date_between(start_date=DATE_START, end_date=DATE_END)
        signup_dates.append(d)

    rows = []
    for i in range(NUM_CUSTOMERS):
        rows.append({
            "customer_id": i + 1,
            "name": fake.name(),
            "email": fake.unique.email(),
            "age": np.random.randint(18, 76),
            "gender": np.random.choice(["M", "F", "Other"], p=[0.45, 0.45, 0.10]),
            "city": np.random.choice(CITIES),
            "signup_date": signup_dates[i],
            "segment_label": segments[i],
        })

    return pd.DataFrame(rows)


def generate_stores() -> pd.DataFrame:
    print("Generating stores...")
    rows = []
    store_types = ["mall", "standalone", "online"]
    for i in range(NUM_STORES):
        prefix = np.random.choice(STORE_NAMES_PREFIX)
        city = np.random.choice(CITIES)
        stype = np.random.choice(store_types, p=[0.30, 0.40, 0.30])
        rows.append({
            "store_id": i + 1,
            "store_name": f"{prefix} {city} #{i + 1}",
            "city": city,
            "store_type": stype,
        })
    return pd.DataFrame(rows)


def generate_promotions(products: pd.DataFrame) -> pd.DataFrame:
    print("Generating promotions...")
    rows = []
    categories = list(CATEGORY_CONFIG.keys())
    promo_types = ["seasonal", "clearance", "flash_sale"]

    for i in range(NUM_PROMOTIONS):
        # ~60% category-level, ~40% product-level
        if np.random.random() < 0.6:
            cat = np.random.choice(categories)
            prod_id = None
        else:
            prod_id = int(np.random.choice(products["product_id"]))
            cat = None

        start = fake.date_between(start_date=DATE_START, end_date=DATE_END - pd.Timedelta(days=14))
        duration = np.random.randint(3, 21)
        end = start + pd.Timedelta(days=duration)
        discount = np.random.choice([5, 10, 15, 20, 25, 30, 35, 40])

        rows.append({
            "promo_id": i + 1,
            "product_id": prod_id,
            "category": cat,
            "start_date": start,
            "end_date": end,
            "discount_pct": discount,
            "promo_type": np.random.choice(promo_types, p=[0.50, 0.25, 0.25]),
        })

    return pd.DataFrame(rows)


def _seasonality_multiplier(date: pd.Timestamp) -> float:
    """Monthly seasonality: spike Nov-Dec, dip Jan-Feb."""
    month = date.month
    multipliers = {
        1: 0.70, 2: 0.75, 3: 0.90, 4: 0.95,
        5: 1.00, 6: 1.05, 7: 1.05, 8: 1.00,
        9: 0.95, 10: 1.05, 11: 1.35, 12: 1.50,
    }
    return multipliers[month]


def _weekend_multiplier(date: pd.Timestamp) -> float:
    """Higher sales on weekends (Sat=5, Sun=6)."""
    return 1.30 if date.dayofweek >= 5 else 1.00


def _online_trend(date: pd.Timestamp) -> float:
    """Gradual upward trend for online stores over 2 years."""
    days_since_start = (date - DATE_START).days
    total_days = (DATE_END - DATE_START).days
    # goes from 1.0 to 1.6 linearly
    return 1.0 + 0.6 * (days_since_start / total_days)


def generate_transactions_and_items(
    customers: pd.DataFrame,
    stores: pd.DataFrame,
    products: pd.DataFrame,
    promotions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Generating transactions and line items...")

    # Pre-compute useful lookups
    customer_segments = customers.set_index("customer_id")["segment_label"].to_dict()
    customer_signup = customers.set_index("customer_id")["signup_date"].to_dict()
    store_types = stores.set_index("store_id")["store_type"].to_dict()
    product_categories = products.set_index("product_id")["category"].to_dict()
    product_prices = products.set_index("product_id")["base_price"].to_dict()
    product_ids = products["product_id"].values

    online_store_ids = stores[stores["store_type"] == "online"]["store_id"].values
    physical_store_ids = stores[stores["store_type"] != "online"]["store_id"].values

    # Build active promo lookup: date -> list of (product_id, category, discount_pct)
    # For efficiency, precompute per-day promo sets is too expensive; we'll check on the fly.
    promo_list = promotions.to_dict("records")

    # Segment-based transaction frequency (avg transactions per customer over 2 years)
    segment_txn_count = {
        "high_value": 12,
        "regular": 5,
        "bargain_hunter": 4,
        "churned": 2,
    }

    # Pre-generate all dates in range
    all_dates = pd.date_range(DATE_START, DATE_END)
    total_days = len(all_dates)

    # Generate date-level weights for sampling
    date_weights = np.array([
        _seasonality_multiplier(d) * _weekend_multiplier(d)
        for d in all_dates
    ])
    date_weights /= date_weights.sum()

    # Allocate transactions per customer based on segment
    print("  Allocating transactions per customer...")
    customer_ids = customers["customer_id"].values
    segments = customers["segment_label"].values

    txn_counts = np.array([
        max(1, int(np.random.poisson(segment_txn_count[seg])))
        for seg in segments
    ])

    # Scale to hit target total
    scale = NUM_TRANSACTIONS / txn_counts.sum()
    txn_counts = np.maximum(1, np.round(txn_counts * scale).astype(int))

    total_txns = txn_counts.sum()
    print(f"  Total transactions to generate: {total_txns}")

    # Build transactions
    txn_rows = []
    item_rows = []
    txn_id = 1
    item_id = 1

    # Category elasticity: how much discount affects quantity
    elasticity_qty_boost = {"high": 1.8, "medium": 1.3, "low": 1.05}

    for idx in range(len(customer_ids)):
        cid = int(customer_ids[idx])
        seg = segments[idx]
        n_txns = int(txn_counts[idx])
        signup = customer_signup[cid]

        # Churned customers only buy in first 6 months after signup
        if seg == "churned":
            cutoff = pd.Timestamp(signup) + pd.Timedelta(days=180)
            valid_mask = all_dates <= cutoff
            if not valid_mask.any():
                continue
            cust_dates = all_dates[valid_mask]
            cust_weights = date_weights[valid_mask]
            cust_weights = cust_weights / cust_weights.sum()
        else:
            signup_ts = pd.Timestamp(signup)
            valid_mask = all_dates >= signup_ts
            cust_dates = all_dates[valid_mask]
            cust_weights = date_weights[valid_mask]
            cust_weights = cust_weights / cust_weights.sum()

        if len(cust_dates) == 0:
            continue

        # Sample transaction dates
        txn_date_indices = np.random.choice(len(cust_dates), size=n_txns, p=cust_weights)

        for di in txn_date_indices:
            txn_date = cust_dates[di]

            # Store selection: online stores get a trend boost
            if np.random.random() < 0.3 * _online_trend(txn_date):
                sid = int(np.random.choice(online_store_ids))
            else:
                sid = int(np.random.choice(physical_store_ids))

            # Number of items in this transaction
            n_items = max(1, int(np.random.poisson(AVG_ITEMS_PER_TXN - 1)) + 1)

            # High-value customers buy more items
            if seg == "high_value":
                n_items = max(1, n_items + np.random.randint(0, 3))

            total_amount = 0.0
            chosen_products = np.random.choice(product_ids, size=n_items, replace=False) \
                if n_items <= len(product_ids) else np.random.choice(product_ids, size=n_items)

            for pid in chosen_products:
                pid = int(pid)
                base_p = product_prices[pid]
                cat = product_categories[pid]
                elasticity = CATEGORY_CONFIG[cat]["elasticity"]

                # Check if any promo applies
                discount_pct = 0
                for promo in promo_list:
                    ps = pd.Timestamp(promo["start_date"])
                    pe = pd.Timestamp(promo["end_date"])
                    if ps <= txn_date <= pe:
                        if promo["product_id"] == pid or promo["category"] == cat:
                            discount_pct = max(discount_pct, promo["discount_pct"])

                # Bargain hunters strongly prefer discounted items; if no discount, sometimes skip
                if seg == "bargain_hunter" and discount_pct == 0:
                    if np.random.random() < 0.5:
                        # Apply a small self-found discount (coupon behavior)
                        discount_pct = np.random.choice([5, 10, 15])

                unit_price = round(base_p * (1 - discount_pct / 100), 2)

                # Quantity: elastic products see higher qty when discounted
                base_qty = np.random.randint(1, 6)
                if discount_pct > 0:
                    boost = elasticity_qty_boost[elasticity]
                    base_qty = max(1, min(5, int(base_qty * (1 + (discount_pct / 100) * (boost - 1)))))
                quantity = min(base_qty, 5)

                line_total = round(unit_price * quantity, 2)
                total_amount += line_total

                item_rows.append({
                    "item_id": item_id,
                    "transaction_id": txn_id,
                    "product_id": pid,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "discount_pct": discount_pct,
                })
                item_id += 1

            txn_rows.append({
                "transaction_id": txn_id,
                "customer_id": cid,
                "store_id": sid,
                "transaction_date": txn_date.date(),
                "total_amount": round(total_amount, 2),
            })
            txn_id += 1

        # Progress logging every 10k customers
        if (idx + 1) % 10_000 == 0:
            print(f"  Processed {idx + 1}/{len(customer_ids)} customers, "
                  f"{txn_id - 1} transactions, {item_id - 1} items so far...")

    print(f"  Final: {txn_id - 1} transactions, {item_id - 1} line items")
    return pd.DataFrame(txn_rows), pd.DataFrame(item_rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    products = generate_products()
    customers = generate_customers()
    stores = generate_stores()
    promotions = generate_promotions(products)
    transactions, items = generate_transactions_and_items(
        customers, stores, products, promotions
    )

    # Save all CSVs
    files = {
        "products.csv": products,
        "customers.csv": customers,
        "stores.csv": stores,
        "promotions.csv": promotions,
        "transactions.csv": transactions,
        "transaction_items.csv": items,
    }

    for fname, df in files.items():
        path = os.path.join(OUTPUT_DIR, fname)
        df.to_csv(path, index=False)
        print(f"Saved {fname}: {len(df):,} rows")

    print("\nDone! All CSVs saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()

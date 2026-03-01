"""
Retail Brain AI Copilot — interactive CLI for querying forecasting,
customer intelligence, and pricing optimization outputs via natural language.

Uses AWS Bedrock (Claude) with the converse API for NLU, response generation,
and tool use. Pre-computed CSV outputs are loaded at startup; small files are
embedded in the system prompt and large files are aggregated into summaries.
Claude can call data-lookup tools to drill into row-level detail.

Usage:
    python scripts/ai_copilot.py [--debug]
"""

import argparse
import io
import json
import os
import re
import sys
import traceback
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MAX_TOKENS = 4096
TEMPERATURE = 0.3
MAX_HISTORY = 20          # conversation turns retained
MAX_CONVERSATION_CHARS = 400_000  # ~100K tokens — leaves room for system prompt

CHART_DIR = os.path.join(DATA_DIR, "copilot_charts")
EXPORT_DIR = os.path.join(DATA_DIR, "copilot_exports")
HISTORY_FILE = os.path.join(DATA_DIR, "copilot_history.json")

COLORS = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
          "#1abc9c", "#e67e22", "#34495e", "#d35400", "#27ae60"]

DEBUG = False

# CSV paths keyed by logical name ─ (subdirectory, filename)
CSV_MANIFEST = {
    # Historical sales
    "daily_sales":            (".", "daily_sales.csv"),
    "transactions":           (".", "transactions.csv"),
    "transaction_items":      (".", "transaction_items.csv"),
    "products":               (".", "products.csv"),
    # Forecasts
    "forecast_results":       ("forecasts", "forecast_results.csv"),
    # Customer Intelligence
    "rfm_scores":             ("customer_intelligence", "rfm_scores.csv"),
    "segment_summary":        ("customer_intelligence", "segment_summary.csv"),
    "category_affinity":      ("customer_intelligence", "category_affinity.csv"),
    "churn_predictions":      ("customer_intelligence", "churn_predictions.csv"),
    "recommendations":        ("customer_intelligence", "recommendations.csv"),
    # Pricing Optimization
    "price_elasticity":       ("pricing_optimization", "price_elasticity.csv"),
    "discount_sensitivity":   ("pricing_optimization", "discount_sensitivity.csv"),
    "optimal_discounts":      ("pricing_optimization", "optimal_discounts.csv"),
    "promotion_timing":       ("pricing_optimization", "promotion_timing.csv"),
    "cannibalization_risk":   ("pricing_optimization", "cannibalization_risk.csv"),
    "margin_impact_summary":  ("pricing_optimization", "margin_impact_summary.csv"),
}

# Small CSVs embedded verbatim in the system prompt
SMALL_CSVS = [
    "forecast_results", "segment_summary", "category_affinity",
    "recommendations", "margin_impact_summary", "discount_sensitivity",
    "promotion_timing", "cannibalization_risk",
]

# Large CSVs that get pre-aggregated
LARGE_CSVS = ["daily_sales", "transactions", "transaction_items", "products",
              "rfm_scores", "churn_predictions", "price_elasticity",
              "optimal_discounts"]

# ---------------------------------------------------------------------------
# Conversation persistence
# ---------------------------------------------------------------------------

def save_history(conversation):
    """Save conversation history to disk."""
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(conversation, f)
    except Exception as exc:
        if DEBUG:
            print(f"  Warning: could not save history: {exc}")


def load_history():
    """Load conversation history from disk. Returns list or empty list."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        # Validate structure
        if not isinstance(history, list):
            return []
        return history
    except Exception as exc:
        if DEBUG:
            print(f"  Warning: could not load history: {exc}")
        return []

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv_safe(path):
    """Load a CSV file, returning None on failure."""
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"  Warning: could not load {path}: {exc}")
        return None


def load_all_data():
    """Load all 12 CSVs from the data directories. Returns dict[name → DataFrame]."""
    data = {}
    for name, (subdir, fname) in CSV_MANIFEST.items():
        path = os.path.join(DATA_DIR, subdir, fname)
        df = load_csv_safe(path)
        if df is not None:
            data[name] = df
    return data

# ---------------------------------------------------------------------------
# Aggregations for large files
# ---------------------------------------------------------------------------

def _df_to_csv_string(df):
    """Convert a DataFrame to a compact CSV string."""
    return df.to_csv(index=False)


def compute_aggregations(data):
    """Pre-compute summary tables from large CSVs. Returns dict[name → csv_string]."""
    aggs = {}

    # Monthly revenue by category from daily sales
    if "daily_sales" in data:
        df = data["daily_sales"].copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df["month"] = df["ds"].dt.to_period("M").astype(str)
        monthly = df.groupby(["category", "month"]).agg(
            revenue=("y", "sum"),
            days=("y", "count"),
        ).reset_index()
        monthly["revenue"] = monthly["revenue"].round(2)
        aggs["monthly_sales_by_category"] = _df_to_csv_string(monthly)

    # RFM per-segment percentiles
    if "rfm_scores" in data:
        df = data["rfm_scores"]
        cols = ["recency", "frequency", "monetary"]
        if "churn_probability" in df.columns:
            cols.append("churn_probability")
        agg = df.groupby("segment")[cols].describe(
            percentiles=[0.25, 0.5, 0.75]
        )
        # Flatten multi-level columns
        rows = []
        for seg in agg.index:
            row = {"segment": seg}
            for col in cols:
                for stat in ["25%", "50%", "75%"]:
                    row[f"{col}_{stat}"] = round(agg.loc[seg, (col, stat)], 2)
            rows.append(row)
        aggs["rfm_percentiles"] = _df_to_csv_string(pd.DataFrame(rows))

    # Churn risk distribution per segment
    if "churn_predictions" in data:
        df = data["churn_predictions"]
        ct = df.groupby(["segment", "churn_risk"]).size().reset_index(name="count")
        totals = df.groupby("segment").size().reset_index(name="total")
        ct = ct.merge(totals, on="segment")
        ct["pct"] = (ct["count"] / ct["total"] * 100).round(1)
        aggs["churn_distribution"] = _df_to_csv_string(
            ct[["segment", "churn_risk", "count", "pct"]]
        )

    # Price elasticity summary per category
    if "price_elasticity" in data:
        df = data["price_elasticity"]
        summary = df.groupby("category").agg(
            mean_elasticity=("elasticity_final", "mean"),
            median_elasticity=("elasticity_final", "median"),
            product_count=("product_id", "count"),
        ).reset_index()
        label_counts = df.groupby(["category", "elasticity_label"]).size().unstack(
            fill_value=0
        ).reset_index()
        summary = summary.merge(label_counts, on="category", how="left")
        summary = summary.round(4)
        aggs["elasticity_summary"] = _df_to_csv_string(summary)

    # Discount optimization summary per category
    if "optimal_discounts" in data:
        df = data["optimal_discounts"]
        summary = df.groupby("category").agg(
            avg_optimal_discount=("optimal_discount_pct", "mean"),
            avg_profit_uplift=("profit_uplift_pct", "mean"),
            avg_revenue_uplift=("revenue_uplift_pct", "mean"),
        ).reset_index()
        rec_counts = df.groupby(["category", "recommendation"]).size().unstack(
            fill_value=0
        ).reset_index()
        summary = summary.merge(rec_counts, on="category", how="left")
        summary = summary.round(4)
        aggs["discount_optimization_summary"] = _df_to_csv_string(summary)

    return aggs

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt(data, aggregations):
    """Assemble the system prompt with embedded data and aggregations."""
    sections = []

    sections.append(
        "You are Retail Brain Copilot, an AI assistant for business users.\n"
        "Answer questions using ONLY the data provided below. If the data is "
        "insufficient to answer a question, say so clearly.\n"
        "Use business-friendly language. Format currency with $ and commas. "
        "Format percentages to one decimal place. Cite the dataset name when "
        "referencing data."
    )

    # Embedded small CSVs
    sections.append("## Available Data\n")
    for name in SMALL_CSVS:
        if name in data:
            csv_str = _df_to_csv_string(data[name])
            sections.append(f"### {name}\n```csv\n{csv_str}```\n")
        else:
            sections.append(f"### {name}\nNot available.\n")

    # Pre-computed aggregations
    sections.append("## Aggregated Summaries (from large datasets)\n")
    for name, csv_str in aggregations.items():
        sections.append(f"### {name}\n```csv\n{csv_str}```\n")

    # Tool use instructions
    sections.append(
        "## Data Lookup Tools\n"
        "For questions about specific customers, products, or daily sales, use "
        "the provided tools to look up row-level data from the full datasets "
        "(daily_sales: ~3,600 rows, rfm_scores: ~100K rows, churn_predictions: "
        "~100K rows, price_elasticity: ~2K rows, optimal_discounts: ~2K rows). "
        "The aggregated summaries above give you the big picture; the tools let "
        "you drill into specifics. For daily sales charts, use search_daily_sales "
        "to fetch the data, then emit a [CHART:...] directive.\n"
    )

    # Chart rendering instructions
    sections.append(
        "## Charts\n"
        "To include a chart in your response, emit a directive on its own line:\n"
        "[CHART:{\"type\":\"bar\",\"title\":\"Title\",\"data\":{\"labels\":[...],\"values\":[...]}}]\n\n"
        "Supported chart types:\n"
        "- bar: {\"labels\": [...], \"values\": [...]} or grouped: {\"labels\": [...], \"series\": [{\"name\": \"...\", \"values\": [...]}]}\n"
        "- line: {\"labels\": [...], \"values\": [...]} or multi-series: {\"labels\": [...], \"series\": [{\"name\": \"...\", \"values\": [...]}]}\n"
        "- pie: {\"labels\": [...], \"values\": [...]}\n"
        "- scatter: {\"x\": [...], \"y\": [...], \"x_label\": \"...\", \"y_label\": \"...\"}\n"
        "- heatmap: {\"x_labels\": [...], \"y_labels\": [...], \"values\": [[...]], \"x_label\": \"...\", \"y_label\": \"...\"}\n\n"
        "Include a chart when a visual would help the user understand the data better. "
        "Always include the textual answer as well."
    )

    return "\n\n".join(sections)

# ---------------------------------------------------------------------------
# Tool definitions (Bedrock converse format)
# ---------------------------------------------------------------------------

def build_tool_definitions():
    """Return Bedrock converse toolConfig with tool specs."""
    tools = [
        {
            "toolSpec": {
                "name": "lookup_customer",
                "description": (
                    "Look up a specific customer by their ID. Returns the "
                    "customer's RFM scores, segment, churn probability, "
                    "churn risk tier, first and last transaction dates, "
                    "and total transaction count."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "customer_id": {
                                "type": "integer",
                                "description": "The customer ID to look up",
                            }
                        },
                        "required": ["customer_id"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "lookup_product",
                "description": (
                    "Look up a specific product by its ID. Returns the "
                    "product's category, prices, margin, price elasticity, "
                    "optimal discount, and pricing recommendation."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "product_id": {
                                "type": "integer",
                                "description": "The product ID to look up",
                            }
                        },
                        "required": ["product_id"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "search_customers",
                "description": (
                    "Search and filter customers by segment, churn risk, or "
                    "churn probability range. Returns top N matching customers "
                    "sorted by the specified field."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "segment": {
                                "type": "string",
                                "description": "Filter by segment name (e.g. Loyal, Inactive, New)",
                            },
                            "churn_risk": {
                                "type": "string",
                                "description": "Filter by churn risk tier: Low, Medium, or High",
                            },
                            "min_churn_probability": {
                                "type": "number",
                                "description": "Minimum churn probability (0-1)",
                            },
                            "max_churn_probability": {
                                "type": "number",
                                "description": "Maximum churn probability (0-1)",
                            },
                            "sort_by": {
                                "type": "string",
                                "description": "Column to sort by (default: churn_probability)",
                            },
                            "ascending": {
                                "type": "boolean",
                                "description": "Sort ascending (default: false)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max rows to return (default: 20)",
                            },
                        },
                        "required": [],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "search_products",
                "description": (
                    "Search and filter products by category, elasticity label, "
                    "or pricing recommendation. Returns top N matching products."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Filter by category (e.g. Electronics, Clothing)",
                            },
                            "elasticity_label": {
                                "type": "string",
                                "description": "Filter by elasticity label: low, medium, or high",
                            },
                            "recommendation": {
                                "type": "string",
                                "description": "Filter by pricing recommendation (e.g. hold_price, discount)",
                            },
                            "sort_by": {
                                "type": "string",
                                "description": "Column to sort by (default: elasticity_final)",
                            },
                            "ascending": {
                                "type": "boolean",
                                "description": "Sort ascending (default: true)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max rows to return (default: 20)",
                            },
                        },
                        "required": [],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "search_daily_sales",
                "description": (
                    "Look up daily sales revenue from the daily_sales dataset "
                    "(~3,600 rows). Filter by category and/or date range. "
                    "Returns daily rows with columns: category, ds (date), "
                    "y (revenue). Use this to answer questions about "
                    "historical sales trends, daily patterns, or to get data "
                    "for plotting sales over time."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Filter by category (e.g. Electronics, Clothing, Beauty, Home, Sports)",
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date inclusive (YYYY-MM-DD)",
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date inclusive (YYYY-MM-DD)",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max rows to return (default: 90)",
                            },
                        },
                        "required": [],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "lookup_customer_transactions",
                "description": (
                    "Look up all transactions for a specific customer. "
                    "Returns spending by category (department) with total "
                    "amount, transaction count, and items bought. Can also "
                    "return monthly spending breakdown. Use this to answer "
                    "questions about a customer's purchase history across "
                    "departments/categories."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "customer_id": {
                                "type": "integer",
                                "description": "The customer ID to look up",
                            },
                            "group_by": {
                                "type": "string",
                                "description": "Grouping: 'category' for per-department totals, 'month' for monthly by category (default: category)",
                            },
                        },
                        "required": ["customer_id"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "top_customers_by_revenue",
                "description": (
                    "Find the top N customers by total revenue and return "
                    "their spending breakdown by category (department). "
                    "Use this to answer questions like 'which customers "
                    "drove the most revenue' or 'top spenders by department'."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Number of top customers to return (default: 10)",
                            },
                            "category": {
                                "type": "string",
                                "description": "Optional: only rank by revenue in this category",
                            },
                        },
                        "required": [],
                    }
                },
            }
        },
    ]
    return {"tools": tools}

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def execute_tool(tool_name, tool_input, data):
    """Execute a data-lookup tool and return the result as a JSON string."""

    if tool_name == "lookup_customer":
        cid = tool_input["customer_id"]
        result = {}
        if "rfm_scores" in data:
            match = data["rfm_scores"][data["rfm_scores"]["customer_id"] == cid]
            if not match.empty:
                result["rfm"] = match.iloc[0].to_dict()
        if "churn_predictions" in data:
            match = data["churn_predictions"][
                data["churn_predictions"]["customer_id"] == cid
            ]
            if not match.empty:
                result["churn"] = match.iloc[0].to_dict()
        if not result:
            return json.dumps({"error": f"Customer {cid} not found"})
        # Merge into flat dict
        merged = {}
        if "rfm" in result:
            merged.update(result["rfm"])
        if "churn" in result:
            for k, v in result["churn"].items():
                if k not in merged:
                    merged[k] = v
        # Add actual transaction dates from transactions data
        if "transactions" in data:
            txns = data["transactions"][data["transactions"]["customer_id"] == cid]
            if not txns.empty:
                dates = pd.to_datetime(txns["transaction_date"])
                merged["last_transaction_date"] = str(dates.max().date())
                merged["first_transaction_date"] = str(dates.min().date())
                merged["total_transactions"] = len(txns)
        return json.dumps(merged, default=str)

    elif tool_name == "lookup_product":
        pid = tool_input["product_id"]
        result = {}
        if "price_elasticity" in data:
            match = data["price_elasticity"][
                data["price_elasticity"]["product_id"] == pid
            ]
            if not match.empty:
                result.update(match.iloc[0].to_dict())
        if "optimal_discounts" in data:
            match = data["optimal_discounts"][
                data["optimal_discounts"]["product_id"] == pid
            ]
            if not match.empty:
                for k, v in match.iloc[0].to_dict().items():
                    if k not in result:
                        result[k] = v
        if not result:
            return json.dumps({"error": f"Product {pid} not found"})
        return json.dumps(result, default=str)

    elif tool_name == "search_customers":
        if "rfm_scores" not in data:
            return json.dumps({"error": "rfm_scores data not available"})
        df = data["rfm_scores"].copy()
        # Merge churn data if available
        if "churn_predictions" in data:
            churn = data["churn_predictions"]
            extra_cols = [c for c in churn.columns if c not in df.columns]
            if extra_cols:
                df = df.merge(
                    churn[["customer_id"] + extra_cols],
                    on="customer_id", how="left",
                )
        # Apply filters
        if "segment" in tool_input:
            df = df[df["segment"].str.lower() == tool_input["segment"].lower()]
        if "churn_risk" in tool_input and "churn_risk" in df.columns:
            df = df[df["churn_risk"].str.lower() == tool_input["churn_risk"].lower()]
        if "min_churn_probability" in tool_input and "churn_probability" in df.columns:
            df = df[df["churn_probability"] >= tool_input["min_churn_probability"]]
        if "max_churn_probability" in tool_input and "churn_probability" in df.columns:
            df = df[df["churn_probability"] <= tool_input["max_churn_probability"]]
        # Sort
        sort_col = tool_input.get("sort_by", "churn_probability")
        if sort_col not in df.columns:
            sort_col = "churn_probability" if "churn_probability" in df.columns else df.columns[0]
        ascending = tool_input.get("ascending", False)
        df = df.sort_values(sort_col, ascending=ascending)
        # Limit
        limit = tool_input.get("limit", 20)
        df = df.head(limit)
        return df.to_json(orient="records", default_handler=str)

    elif tool_name == "search_products":
        if "price_elasticity" not in data:
            return json.dumps({"error": "price_elasticity data not available"})
        df = data["price_elasticity"].copy()
        # Merge optimal discounts if available
        if "optimal_discounts" in data:
            od = data["optimal_discounts"]
            extra_cols = [c for c in od.columns if c not in df.columns]
            if extra_cols:
                df = df.merge(
                    od[["product_id"] + extra_cols],
                    on="product_id", how="left",
                )
        # Apply filters
        if "category" in tool_input:
            df = df[df["category"].str.lower() == tool_input["category"].lower()]
        if "elasticity_label" in tool_input and "elasticity_label" in df.columns:
            df = df[
                df["elasticity_label"].str.lower()
                == tool_input["elasticity_label"].lower()
            ]
        if "recommendation" in tool_input and "recommendation" in df.columns:
            df = df[
                df["recommendation"].str.lower()
                == tool_input["recommendation"].lower()
            ]
        # Sort
        sort_col = tool_input.get("sort_by", "elasticity_final")
        if sort_col not in df.columns:
            sort_col = "elasticity_final" if "elasticity_final" in df.columns else df.columns[0]
        ascending = tool_input.get("ascending", True)
        df = df.sort_values(sort_col, ascending=ascending)
        # Limit
        limit = tool_input.get("limit", 20)
        df = df.head(limit)
        return df.to_json(orient="records", default_handler=str)

    elif tool_name == "search_daily_sales":
        if "daily_sales" not in data:
            return json.dumps({"error": "daily_sales data not available"})
        df = data["daily_sales"][["category", "ds", "y"]].copy()
        # Apply filters
        if "category" in tool_input:
            df = df[df["category"].str.lower() == tool_input["category"].lower()]
        if "start_date" in tool_input:
            df = df[df["ds"] >= tool_input["start_date"]]
        if "end_date" in tool_input:
            df = df[df["ds"] <= tool_input["end_date"]]
        df = df.sort_values("ds")
        limit = tool_input.get("limit", 90)
        df = df.head(limit)
        return df.to_json(orient="records", default_handler=str)

    elif tool_name == "lookup_customer_transactions":
        for required in ("transactions", "transaction_items", "products"):
            if required not in data:
                return json.dumps({"error": f"{required} data not available"})
        cid = tool_input["customer_id"]
        # Filter transactions for this customer
        txns = data["transactions"][data["transactions"]["customer_id"] == cid]
        if txns.empty:
            return json.dumps({"error": f"Customer {cid} not found in transactions"})
        # Join with items and products to get categories
        items = data["transaction_items"].merge(
            txns[["transaction_id", "transaction_date"]], on="transaction_id"
        )
        items = items.merge(
            data["products"][["product_id", "category"]], on="product_id"
        )
        items["line_total"] = items["quantity"] * items["unit_price"] * (1 - items["discount_pct"])

        group_by = tool_input.get("group_by", "category")
        if group_by == "month":
            items["month"] = pd.to_datetime(items["transaction_date"]).dt.to_period("M").astype(str)
            summary = items.groupby(["category", "month"]).agg(
                total_spent=("line_total", "sum"),
                items_bought=("quantity", "sum"),
            ).reset_index().round(2)
            summary = summary.sort_values(["month", "category"])
        else:
            summary = items.groupby("category").agg(
                total_spent=("line_total", "sum"),
                items_bought=("quantity", "sum"),
                transactions=("transaction_id", "nunique"),
            ).reset_index().round(2)
            summary = summary.sort_values("total_spent", ascending=False)
        return summary.to_json(orient="records", default_handler=str)

    elif tool_name == "top_customers_by_revenue":
        for required in ("transactions", "transaction_items", "products"):
            if required not in data:
                return json.dumps({"error": f"{required} data not available"})
        # Join transactions → items → products
        items = data["transaction_items"].merge(
            data["transactions"][["transaction_id", "customer_id"]],
            on="transaction_id",
        )
        items = items.merge(
            data["products"][["product_id", "category"]], on="product_id"
        )
        items["line_total"] = items["quantity"] * items["unit_price"] * (1 - items["discount_pct"])
        # Optional category filter
        if "category" in tool_input:
            items = items[items["category"].str.lower() == tool_input["category"].lower()]
        # Find top N customers by total revenue
        limit = tool_input.get("limit", 10)
        top_ids = (
            items.groupby("customer_id")["line_total"]
            .sum().nlargest(limit).index.tolist()
        )
        # Get per-category breakdown for those customers
        top_items = items[items["customer_id"].isin(top_ids)]
        summary = top_items.groupby(["customer_id", "category"]).agg(
            total_spent=("line_total", "sum"),
            items_bought=("quantity", "sum"),
        ).reset_index().round(2)
        # Add total per customer for context
        totals = summary.groupby("customer_id")["total_spent"].sum().reset_index()
        totals.columns = ["customer_id", "grand_total"]
        summary = summary.merge(totals, on="customer_id")
        summary = summary.sort_values(["grand_total", "category"], ascending=[False, True])
        return summary.to_json(orient="records", default_handler=str)

    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------

CHART_PATTERN = re.compile(r"\[CHART:(\{.*?\})\]", re.DOTALL)


def extract_chart_directives(response):
    """Extract [CHART:{...}] directives from response text.

    Returns (clean_text, list_of_spec_dicts).
    """
    specs = []
    for match in CHART_PATTERN.finditer(response):
        try:
            spec = json.loads(match.group(1))
            specs.append(spec)
        except json.JSONDecodeError:
            pass
    clean = CHART_PATTERN.sub("", response).strip()
    return clean, specs


def render_chart(spec, index, save=True):
    """Render a single chart spec.

    If *save* is True (default), saves the chart as a PNG and returns the
    filepath.  If *save* is False, returns the matplotlib Figure object
    directly (caller is responsible for displaying/closing it).
    Returns None on failure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save:
        os.makedirs(CHART_DIR, exist_ok=True)
    chart_type = spec.get("type", "bar")
    title = spec.get("title", "Chart")
    chart_data = spec.get("data", {})

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            labels = chart_data.get("labels", [])
            if "series" in chart_data:
                # Grouped bar
                import numpy as np
                series_list = chart_data["series"]
                n_groups = len(labels)
                n_series = len(series_list)
                width = 0.8 / n_series
                x = np.arange(n_groups)
                for i, s in enumerate(series_list):
                    ax.bar(x + i * width, s["values"], width,
                           label=s["name"], color=COLORS[i % len(COLORS)])
                ax.set_xticks(x + width * (n_series - 1) / 2)
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.legend()
            else:
                values = chart_data.get("values", [])
                bars = ax.bar(labels, values,
                              color=COLORS[:len(labels)])
                ax.set_xticklabels(labels, rotation=45, ha="right")

        elif chart_type == "line":
            labels = chart_data.get("labels", [])
            if "series" in chart_data:
                for i, s in enumerate(chart_data["series"]):
                    ax.plot(labels, s["values"], label=s["name"],
                            color=COLORS[i % len(COLORS)], marker="o",
                            markersize=4)
                ax.legend()
            else:
                values = chart_data.get("values", [])
                ax.plot(labels, values, color=COLORS[0], marker="o",
                        markersize=4)
            plt.xticks(rotation=45, ha="right")

        elif chart_type == "pie":
            labels = chart_data.get("labels", [])
            values = chart_data.get("values", [])
            ax.pie(values, labels=labels, autopct="%1.1f%%",
                   colors=COLORS[:len(labels)], startangle=140)
            ax.axis("equal")

        elif chart_type == "scatter":
            x_vals = chart_data.get("x", [])
            y_vals = chart_data.get("y", [])
            ax.scatter(x_vals, y_vals, color=COLORS[0], alpha=0.6)
            ax.set_xlabel(chart_data.get("x_label", "X"))
            ax.set_ylabel(chart_data.get("y_label", "Y"))

        elif chart_type == "heatmap":
            import numpy as np
            values = np.array(chart_data.get("values", [[]]))
            im = ax.imshow(values, cmap="YlOrRd", aspect="auto")
            ax.set_xticks(range(len(chart_data.get("x_labels", []))))
            ax.set_xticklabels(chart_data.get("x_labels", []),
                               rotation=45, ha="right")
            ax.set_yticks(range(len(chart_data.get("y_labels", []))))
            ax.set_yticklabels(chart_data.get("y_labels", []))
            ax.set_xlabel(chart_data.get("x_label", ""))
            ax.set_ylabel(chart_data.get("y_label", ""))
            fig.colorbar(im, ax=ax)

        else:
            plt.close(fig)
            return None

        ax.set_title(title)
        fig.tight_layout()

        if not save:
            return fig

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CHART_DIR, f"chart_{index}_{ts}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    except Exception as exc:
        plt.close("all")
        if DEBUG:
            traceback.print_exc()
        print(f"  Warning: chart rendering failed: {exc}")
        return None


def render_all_charts(specs, save=True):
    """Render all chart specs.

    If *save* is True, returns list of saved file paths.
    If *save* is False, returns list of matplotlib Figure objects.
    """
    results = []
    for i, spec in enumerate(specs):
        result = render_chart(spec, i, save=save)
        if result:
            results.append(result)
    return results

# ---------------------------------------------------------------------------
# Bedrock client and API
# ---------------------------------------------------------------------------

def create_bedrock_client():
    """Create a boto3 Bedrock Runtime client."""
    import boto3
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def _conversation_chars(conversation):
    """Estimate total character count of a conversation."""
    total = 0
    for msg in conversation:
        for block in msg.get("content", []):
            if "text" in block:
                total += len(block["text"])
            elif "json" in block:
                total += len(json.dumps(block["json"]))
            elif "toolUse" in block:
                total += len(json.dumps(block["toolUse"].get("input", {})))
            elif "toolResult" in block:
                for c in block["toolResult"].get("content", []):
                    if "text" in c:
                        total += len(c["text"])
                    elif "json" in c:
                        total += len(json.dumps(c["json"]))
    return total


def _trim_conversation(conversation):
    """Trim conversation by turn count and total size."""
    while len(conversation) > MAX_HISTORY:
        conversation.pop(0)
    while conversation and _conversation_chars(conversation) > MAX_CONVERSATION_CHARS:
        conversation.pop(0)
    # Ensure conversation starts with a user turn (Bedrock requirement)
    while conversation and conversation[0]["role"] != "user":
        conversation.pop(0)


def call_claude(client, system_prompt, tools, conversation, user_message, data):
    """Send a message to Claude via Bedrock converse API.

    Handles the tool-use loop: if Claude responds with toolUse blocks, execute
    them locally, feed results back, and repeat until Claude returns a final
    text response.
    """
    # Append user message
    conversation.append({"role": "user", "content": [{"text": user_message}]})

    _trim_conversation(conversation)

    while True:
        response = client.converse(
            modelId=MODEL_ID,
            system=[{"text": system_prompt}],
            messages=list(conversation),
            inferenceConfig={
                "maxTokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            },
            toolConfig=tools,
        )

        output = response["output"]["message"]
        conversation.append(output)

        # Check for tool use
        tool_uses = [
            block for block in output["content"] if "toolUse" in block
        ]
        if not tool_uses:
            # Final text response
            text_parts = [
                block["text"] for block in output["content"] if "text" in block
            ]
            return "\n".join(text_parts)

        # Execute each tool and build results
        tool_results = []
        for block in tool_uses:
            tu = block["toolUse"]
            tool_name = tu["name"]
            tool_input = tu["input"]
            tool_id = tu["toolUseId"]

            if DEBUG:
                print(f"  [Tool call] {tool_name}({json.dumps(tool_input)})")

            result_str = execute_tool(tool_name, tool_input, data)

            if DEBUG:
                preview = result_str[:200] + "..." if len(result_str) > 200 else result_str
                print(f"  [Tool result] {preview}")

            # Bedrock requires toolResult json to be an object, not an array
            parsed = None
            try:
                parsed = json.loads(result_str)
            except json.JSONDecodeError:
                pass

            if parsed is not None:
                if isinstance(parsed, list):
                    parsed = {"results": parsed}
                content = [{"json": parsed}]
            else:
                content = [{"text": result_str}]

            tool_results.append({
                "toolResult": {
                    "toolUseId": tool_id,
                    "content": content,
                }
            })

        # Send tool results back
        conversation.append({"role": "user", "content": tool_results})

        # Trim again after tool loop
        _trim_conversation(conversation)

# ---------------------------------------------------------------------------
# REPL commands
# ---------------------------------------------------------------------------

def print_banner():
    print("\n" + "=" * 60)
    print("  Retail Brain AI Copilot")
    print("  Ask questions about forecasting, customers, and pricing")
    print("=" * 60)
    print("  Type /help for commands, /quit to exit\n")


def print_help():
    print("\nCommands:")
    print("  /help   — Show this help message")
    print("  /quit   — Exit the copilot (also: /exit)")
    print("  /data   — Show loaded datasets and row counts")
    print("  /clear  — Reset conversation history")
    print("  /export — Save conversation to file")
    print()
    print("Example queries:")
    print('  "What is the forecasted revenue for Electronics next week?"')
    print('  "Which customer segment has the highest churn risk?"')
    print('  "When should we run promotions for Clothing?"')
    print('  "What is the churn probability for customer 42?"')
    print('  "Show me the top 10 highest-risk customers in the Loyal segment"')
    print('  "Show me a chart of forecasted revenue by category"')
    print()


def print_data_summary(data):
    print("\nLoaded datasets:")
    for name, df in sorted(data.items()):
        subdir, fname = CSV_MANIFEST[name]
        print(f"  {name:30s}  {len(df):>8,} rows  ({subdir}/{fname})")
    print()


def handle_command(cmd, data, conversation):
    """Handle slash commands. Returns response string or None if not a command."""
    cmd_lower = cmd.strip().lower()

    if cmd_lower in ("/quit", "/exit"):
        save_history(conversation)
        print("\nGoodbye!")
        sys.exit(0)

    if cmd_lower == "/help":
        print_help()
        return ""

    if cmd_lower == "/data":
        print_data_summary(data)
        return ""

    if cmd_lower == "/clear":
        conversation.clear()
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        print("  Conversation history cleared.")
        return ""

    if cmd_lower == "/export":
        os.makedirs(EXPORT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(EXPORT_DIR, f"conversation_{ts}.txt")
        with open(path, "w") as f:
            for msg in conversation:
                role = msg["role"]
                for block in msg["content"]:
                    if "text" in block:
                        f.write(f"[{role}] {block['text']}\n\n")
                    elif "toolUse" in block:
                        tu = block["toolUse"]
                        f.write(f"[tool call] {tu['name']}({json.dumps(tu['input'])})\n\n")
                    elif "toolResult" in block:
                        f.write("[tool result] ...\n\n")
        print(f"  Conversation exported to {path}")
        return ""

    return None  # Not a command

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global DEBUG

    parser = argparse.ArgumentParser(description="Retail Brain AI Copilot")
    parser.add_argument("--debug", action="store_true", help="Show debug info")
    args = parser.parse_args()
    DEBUG = args.debug

    print_banner()

    # --- Check boto3 ---
    try:
        import boto3  # noqa: F401
    except ImportError:
        print("Error: boto3 is required. Install it with: pip install boto3")
        sys.exit(1)

    # --- Load data ---
    print("Loading data...")
    data = load_all_data()
    if not data:
        print("Error: no data files found. Run the pipelines first.")
        sys.exit(1)
    print(f"  Loaded {len(data)} datasets")

    # --- Compute aggregations ---
    print("Computing aggregations...")
    aggregations = compute_aggregations(data)
    print(f"  Computed {len(aggregations)} summary tables")

    # --- Build system prompt and tools ---
    system_prompt = build_system_prompt(data, aggregations)
    tools = build_tool_definitions()
    if DEBUG:
        print(f"  System prompt: {len(system_prompt)} chars")

    # --- Create Bedrock client ---
    print("Connecting to AWS Bedrock...")
    try:
        client = create_bedrock_client()
    except Exception as exc:
        print(f"Error: could not create Bedrock client: {exc}")
        if DEBUG:
            traceback.print_exc()
        sys.exit(1)

    print("Ready!\n")

    # --- REPL ---
    conversation = load_history()
    if conversation:
        print(f"  Restored {len(conversation)} messages from previous session")
        print("  (Use /clear to start fresh)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            save_history(conversation)
            print("\nGoodbye!")
            break
        except KeyboardInterrupt:
            print()
            continue

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            result = handle_command(user_input, data, conversation)
            if result is not None:
                continue

        # Call Claude
        print("Thinking...")
        try:
            response = call_claude(
                client, system_prompt, tools, conversation, user_input, data
            )
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            continue
        except Exception as exc:
            print(f"\n  Error: {exc}")
            if DEBUG:
                traceback.print_exc()
            continue

        # Save history after each successful exchange
        save_history(conversation)

        # Extract and render charts
        clean_text, chart_specs = extract_chart_directives(response)

        print(f"\nCopilot: {clean_text}\n")

        if chart_specs:
            paths = render_all_charts(chart_specs)
            for p in paths:
                print(f"  Chart saved: {p}")
            if paths:
                print()


if __name__ == "__main__":
    main()

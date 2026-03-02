# Scripts

Python pipelines that power Retail Brain. Run them in order.

## Files

| Script | Purpose | Output |
|--------|---------|--------|
| `generate_data.py` | Generates synthetic retail dataset (~2M rows across 6 CSVs) with embedded seasonality, weekend effects, and customer behavior patterns | `data/*.csv` |
| `train_forecast.py` | Trains an NBEATSx model on daily category-level sales with exogenous features (day-of-week, month, weekend, promo count). Evaluates on a 30-day holdout | `models/nbeats_model/`, `lightning_logs/` |
| `predict_forecast.py` | Loads the trained model and generates 30-day forward forecasts per category | `data/forecasts/` |
| `customer_intelligence.py` | End-to-end customer analytics: RFM scoring, K-Means segmentation, churn prediction (Random Forest), category affinity, and re-engagement recommendations | `data/customer_intelligence/` |
| `pricing_optimization.py` | Price elasticity estimation (arc + OLS), discount sensitivity analysis, optimal discount depth (scipy), promotion timing windows, and cannibalization risk assessment | `data/pricing_optimization/` |
| `ai_copilot.py` | Interactive CLI for querying all pipeline outputs via natural language. Uses AWS Bedrock (Claude) with 8 data-lookup tools for drill-down queries (customers, products, daily sales, stores, transactions) and chart rendering | `data/copilot_charts/`, `data/copilot_exports/` |
| `ai_copilot_app.py` | Streamlit web interface for the AI Copilot. Reuses all backend logic from `ai_copilot.py` and renders charts inline | — |

## Usage

```bash
source retail_brain_env/bin/activate

python scripts/generate_data.py          # Step 1: Create data
python scripts/train_forecast.py         # Step 2: Train forecast model
python scripts/predict_forecast.py       # Step 3: Generate forecasts
python scripts/customer_intelligence.py  # Step 4: Customer intelligence
python scripts/pricing_optimization.py   # Step 5: Pricing optimization
python scripts/ai_copilot.py             # Step 6: AI Copilot CLI (requires AWS credentials)
streamlit run scripts/ai_copilot_app.py  # Step 6 alt: AI Copilot web UI
```

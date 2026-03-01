# Pricing Optimization

Output from `scripts/pricing_optimization.py` — price elasticity estimation, discount optimization, promotion timing, and cannibalization analysis.

## CSV Outputs

| File | Description |
|------|-------------|
| `price_elasticity.csv` | Per-product price elasticity estimates with category, margin, and sensitivity label |
| `discount_sensitivity.csv` | Profit index and quantity uplift at each discount level per category |
| `optimal_discounts.csv` | Per-product optimal discount depth with profit/revenue uplift and recommendation |
| `promotion_timing.csv` | Monthly demand index and promotion window scoring per category |
| `cannibalization_risk.csv` | Category/subcategory cannibalization risk assessment |
| `margin_impact_summary.csv` | Per-category margin impact summary with top recommendation |

## Visualizations

| File | Description |
|------|-------------|
| `elasticity_by_category.png` | Box plot of price elasticity distribution per category |
| `discount_response_curves.png` | Profit index vs discount depth per category (line plot) |
| `optimal_discount_scatter.png` | Optimal discount vs profit uplift per product (scatter, colored by category) |
| `demand_seasonality_heatmap.png` | Category × month demand index heatmap for promotion timing |
| `cannibalization_network.png` | Cannibalization % by category/subcategory (grouped bar chart) |

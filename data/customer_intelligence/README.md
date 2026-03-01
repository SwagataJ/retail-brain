# Customer Intelligence

Output from `scripts/customer_intelligence.py` — segmentation, churn prediction, and re-engagement analysis.

## CSV Outputs

| File | Description |
|------|-------------|
| `rfm_scores.csv` | Per-customer RFM scores, segment assignment, and churn probability |
| `segment_summary.csv` | Aggregate metrics (count, avg recency/frequency/monetary, churn %) per segment |
| `category_affinity.csv` | Share-of-wallet percentages per segment x category |
| `churn_predictions.csv` | Per-customer churn probability and risk tier (Low/Medium/High) |
| `recommendations.csv` | Re-engagement timing and actions per segment |

## Visualizations

| File | Description |
|------|-------------|
| `cluster_selection.png` | Elbow (inertia) + silhouette dual plot for optimal k selection |
| `segment_distribution.png` | Pie chart of customer segment sizes |
| `rfm_heatmap.png` | Average RFM scores by segment |
| `churn_feature_importance.png` | Random Forest feature importances (10 behavioral features) |
| `category_affinity_heatmap.png` | Segment x category spend heatmap |

# Customer Segmentation with KMeans

This project segments mall customers using KMeans clustering on annual income and spending score. It includes exploratory analysis, model selection visuals, clustering, and evaluation metrics.

## Dataset
`Mall_Customers.csv` includes the following columns:
- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

## Methodology
- Exploratory analysis of feature distributions
- Feature scaling with `StandardScaler`
- Elbow method and silhouette score for cluster selection
- KMeans clustering (`k=5`)
- Cluster profiling (mean income/spending and count)

## Outputs
All outputs are saved in `reports/`:
- `eda_income_distribution.png`
- `eda_spending_distribution.png`
- `eda_income_vs_spending.png`
- `elbow_method.png`
- `silhouette_by_k.png`
- `davies_bouldin_by_k.png`
- `calinski_harabasz_by_k.png`
- `customer_segments.png`
- `cluster_profile.csv`
- `metrics.csv`
- `metrics_by_k.csv`
- `segmented_customers.csv`

## How to Run
```bash
python kmeans_customer_segmentation.py
```

## Results Snapshot
- Evaluation metrics are saved to `reports/metrics.csv`
- Cluster profiles are saved to `reports/cluster_profile.csv`

## Interpreting the Metrics
- **Silhouette Score**: ranges from -1 to 1. Higher is better.
- **Davies-Bouldin Index**: lower is better.
- **Calinski-Harabasz Index**: higher is better.

To justify `k=5`, compare metrics across `k` in `reports/metrics_by_k.csv` alongside the elbow and silhouette plots.

## Best K Summary
Use these heuristics together:
- Pick a `k` at the elbow where WCSS reduction slows.
- Prefer higher silhouette scores.
- Prefer lower Davies-Bouldin scores.
- Prefer higher Calinski-Harabasz scores.

If all three agree around `k=5`, it is a strong, defensible choice.

## Notes for Analysts
This project demonstrates:
- Proper preprocessing and scaling
- Model selection with multiple diagnostics
- Clear, reproducible outputs
- Interpretable cluster summaries for stakeholders

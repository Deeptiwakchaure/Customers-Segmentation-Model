import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv("Mall_Customers.csv")
print("First 5 rows of the dataset:")
print(df.head())

# Selecting features
feature_cols = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[feature_cols]

# Basic EDA visuals
plt.figure(figsize=(8, 4))
sns.histplot(df["Annual Income (k$)"], bins=20, kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_income_distribution.png"))
plt.close()

plt.figure(figsize=(8, 4))
sns.histplot(df["Spending Score (1-100)"], bins=20, kde=True)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_spending_distribution.png"))
plt.close()

plt.figure(figsize=(6, 5))
sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    data=df,
    s=60
)
plt.title("Income vs Spending (Raw)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_income_vs_spending.png"))
plt.close()

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal clusters
wcss = []
silhouette_scores = []
metrics_by_k = []
cluster_range = range(2, 11)
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    if i in cluster_range:
        sil = silhouette_score(X_scaled, kmeans.labels_)
        db = davies_bouldin_score(X_scaled, kmeans.labels_)
        ch = calinski_harabasz_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(sil)
        metrics_by_k.append(
            {"k": i, "silhouette": round(sil, 4), "davies_bouldin": round(db, 4), "calinski_harabasz": round(ch, 4)}
        )

plt.figure()
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method (WCSS)")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "elbow_method.png"))
plt.close()

plt.figure()
plt.plot(list(cluster_range), silhouette_scores, marker="o")
plt.title("Silhouette Score by K")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "silhouette_by_k.png"))
plt.close()

metrics_by_k_df = pd.DataFrame(metrics_by_k)
metrics_by_k_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_by_k.csv"), index=False)

plt.figure()
plt.plot(metrics_by_k_df["k"], metrics_by_k_df["davies_bouldin"], marker="o")
plt.title("Davies-Bouldin Index by K")
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Index")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "davies_bouldin_by_k.png"))
plt.close()

plt.figure()
plt.plot(metrics_by_k_df["k"], metrics_by_k_df["calinski_harabasz"], marker="o")
plt.title("Calinski-Harabasz Index by K")
plt.xlabel("Number of Clusters")
plt.ylabel("Calinski-Harabasz Index")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "calinski_harabasz_by_k.png"))
plt.close()

# Applying KMeans with 5 clusters
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
df['Cluster'] = y_kmeans

# Visualizing the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='Set2',
    data=df,
    s=100
)
plt.title("Customer Segments (KMeans)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "customer_segments.png"))
plt.close()

# Cluster profiling
cluster_profile = (
    df.groupby("Cluster")[feature_cols]
    .mean()
    .round(2)
    .reset_index()
)
cluster_counts = df["Cluster"].value_counts().sort_index().reset_index()
cluster_counts.columns = ["Cluster", "Count"]
cluster_profile = cluster_profile.merge(cluster_counts, on="Cluster", how="left")
cluster_profile.to_csv(os.path.join(OUTPUT_DIR, "cluster_profile.csv"), index=False)

# Saving clustered data
df.to_csv(os.path.join(OUTPUT_DIR, "segmented_customers.csv"), index=False)
print("Clustering complete. Outputs saved to reports/.")

# === Evaluation Metrics ===
sil_score = silhouette_score(X_scaled, y_kmeans)
db_index = davies_bouldin_score(X_scaled, y_kmeans)
ch_score = calinski_harabasz_score(X_scaled, y_kmeans)

print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {db_index:.4f}")
print(f"Calinski-Harabasz Index: {ch_score:.4f}")

metrics_df = pd.DataFrame(
    {
        "metric": ["silhouette", "davies_bouldin", "calinski_harabasz"],
        "value": [round(sil_score, 4), round(db_index, 4), round(ch_score, 4)],
    }
)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

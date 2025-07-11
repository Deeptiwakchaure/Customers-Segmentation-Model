import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load dataset
df = pd.read_csv('Mall_Customers.csv')
print("First 5 rows of the dataset:")
print(df.head())

# Selecting features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method to find optimal clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig('elbow_method.png')
plt.close()

# Applying KMeans with 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
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
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.savefig('customer_segments.png')
plt.close()

# Saving clustered data
df.to_csv('segmented_customers.csv', index=False)
print("Clustering complete. Outputs saved.")

# === Evaluation Metrics ===
sil_score = silhouette_score(X, y_kmeans)
db_index = davies_bouldin_score(X, y_kmeans)
ch_score = calinski_harabasz_score(X, y_kmeans)

print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {db_index:.4f}")
print(f"Calinski-Harabasz Index: {ch_score:.4f}")

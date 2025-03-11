import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


merged_csv_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_3channels_resting_lanzhou_2015\merged_EEG_3channels_metadata.csv"
output_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_3channels_resting_lanzhou_2015\output"
os.makedirs(output_folder, exist_ok=True)


df_merged = pd.read_csv(merged_csv_path, delimiter=';', decimal='.')
print("Merged Data (first 5 rows):")
print(df_merged.head())


variables_of_interest = [
    "Mean_Alpha_Power", "Mean_Beta_Power", "Mean_Theta_Power", "Mean_Delta_Power",
    "PHQ-9", "CTQ-SF", "LES", "SSRS", "GAD-7", "PSQI"
]

variables = [var for var in variables_of_interest if var in df_merged.columns]
print("\nVariables for clustering:", variables)

df_cluster = df_merged[variables].dropna().copy()
print("\nData used for clustering (first 5 rows):")
print(df_cluster.head())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("\nPCA explained variance ratio:", pca.explained_variance_ratio_)


sil_scores = []
k_range = range(2, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)
    print(f"k = {k}: silhouette score = {sil:.3f}")

optimal_k = k_range[np.argmax(sil_scores)]
print("\nOptimal number of clusters (by silhouette score):", optimal_k)


kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans_final.fit_predict(X_scaled)
df_cluster["Cluster"] = clusters


plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", edgecolor="k", s=50)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Scatter Plot with K-Means Clusters")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.tight_layout()
pca_plot_path = os.path.join(output_folder, "pca_clusters.png")
plt.savefig(pca_plot_path, dpi=150)
plt.show()
print("PCA clustering plot saved to:", pca_plot_path)


cluster_summary = df_cluster.groupby("Cluster").mean()
print("\nCluster Summary:")
print(cluster_summary)

cluster_summary_csv = os.path.join(output_folder, "cluster_summary.csv")
cluster_summary.to_csv(cluster_summary_csv, sep=';')
print("Cluster summary saved to:", cluster_summary_csv)

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


eeg_features_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_128channels_resting_lanzhou_2015\output.xlsx"

meta_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_128channels_resting_lanzhou_2015\subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx"


output_dir = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_128channels_resting_lanzhou_2015"
os.makedirs(output_dir, exist_ok=True)


df_eeg = pd.read_excel(eeg_features_path)
print("EEG Features (first 5 rows):")
print(df_eeg.head())



df_eeg['Subject'] = df_eeg['File'].str.split('_').str[0]
df_eeg['Subject'] = df_eeg['Subject'].apply(lambda x: x[1:] if x.startswith('0') else x)
print("\nEEG Features with extracted Subject IDs (first 5 rows):")
print(df_eeg[['File', 'Subject']].head())


df_meta = pd.read_excel(meta_path)
print("\nMetadata (first 5 rows):")
print(df_meta.head())

print("\nMetadata columns:")
print(df_meta.columns)


df_eeg['Subject'] = df_eeg['Subject'].astype(str)
df_meta['subject id'] = df_meta['subject id'].astype(str)


df_merged = pd.merge(df_eeg, df_meta, left_on='Subject', right_on='subject id', how='left')

df_merged['type'] = df_merged['type'].fillna("NoMetadata")
print("\nMerged Data (first 5 rows):")
print(df_merged.head())

# Save the merged data (should have all 53 rows from EEG features)
merged_output_path = os.path.join(output_dir, "merged_EEG_128channels.xlsx")
df_merged.to_excel(merged_output_path, index=False)
print("\nMerged data saved to:", merged_output_path)


alpha_cols = [col for col in df_merged.columns if col.startswith("Ch") and col.endswith("_Alpha_Power")]
print("\nAlpha Power columns:")
print(alpha_cols)


df_merged['Avg_Alpha_Power'] = df_merged[alpha_cols].mean(axis=1)


X = df_merged[alpha_cols].values
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_merged['PCA1'] = X_pca[:, 0]
df_merged['PCA2'] = X_pca[:, 1]


plt.figure(figsize=(8, 6))
for grp in df_merged['type'].unique():
    idx = df_merged['type'] == grp
    plt.scatter(df_merged.loc[idx, 'PCA1'], df_merged.loc[idx, 'PCA2'], label=grp)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of Alpha Power Features")
plt.legend()
pca_plot_path = os.path.join(output_dir, "pca_alpha_power.png")
plt.tight_layout()
plt.savefig(pca_plot_path)
print("\nPCA plot saved to:", pca_plot_path)
plt.show()


kmeans = KMeans(n_clusters=2, random_state=42)
df_merged['Cluster'] = kmeans.fit_predict(X_pca)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_merged, x='PCA1', y='PCA2', hue='Cluster', style='type', palette='deep')
plt.title("K-means Clustering on PCA of Alpha Power Features")
kmeans_plot_path = os.path.join(output_dir, "kmeans_pca_alpha_power.png")
plt.tight_layout()
plt.savefig(kmeans_plot_path)
print("K-means clustering plot saved to:", kmeans_plot_path)
plt.show()


metadata_vars = ['age', 'PHQ-9', 'CTQ-SF', 'LES', 'SSRS', 'GAD-7', 'PSQI']
for var in metadata_vars:
    if var in df_merged.columns:
        df_merged[var] = pd.to_numeric(df_merged[var], errors='coerce')

corr_vars = ['Avg_Alpha_Power'] + metadata_vars
corr_matrix = df_merged[corr_vars].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix: Avg Alpha Power & Metadata")
corr_heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
plt.tight_layout()
plt.savefig(corr_heatmap_path)
print("Correlation heatmap saved to:", corr_heatmap_path)
plt.show()


final_output_path = os.path.join(output_dir, "final_merged_EEG_128channels_analysis.xlsx")
df_merged.to_excel(final_output_path, index=False)
print("Final merged data with analysis saved to:", final_output_path)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.formula.api as smf

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


data_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_128channels_resting_lanzhou_2015\output\final_merged_EEG_128channels_analysis.xlsx"
output_dir = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_128channels_resting_lanzhou_2015\output"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(data_path)
print("Columns in Excel:")
print(df.columns)


group_map = {"HC": 0, "Control": 0, "MDD": 1, "Depression": 1}
df["Group_binary"] = df["Group"].map(group_map)
df = df.dropna(subset=["Group_binary"])


bands = ["Avg_Alpha_Power"]
group_depr = df[df["Group"].isin(["MDD", "Depression"])]
group_ctrl = df[df["Group"].isin(["HC", "Control"])]

t_test_results = []
print("\nGroup-Level t-tests:")
for band in bands:
    t_stat, p_val = stats.ttest_ind(group_depr[band], group_ctrl[band], equal_var=False, nan_policy='omit')
    depr_mean = group_depr[band].mean()
    ctrl_mean = group_ctrl[band].mean()
    result_str = (f"Band: {band}\n"
                  f"  Depression mean: {depr_mean:.6f}, Control mean: {ctrl_mean:.6f}\n"
                  f"  t-statistic: {t_stat:.4f}, p-value: {p_val:.4f}\n")
    print(result_str)
    t_test_results.append({
        'Band': band,
        'Depression_mean': depr_mean,
        'Control_mean': ctrl_mean,
        't_statistic': t_stat,
        'p_value': p_val
    })
ttest_csv_path = os.path.join(output_dir, "ttest_results_128_eeg_scale.csv")
pd.DataFrame(t_test_results).to_csv(ttest_csv_path, index=False, sep=';')
print(f"T-test results saved to: {ttest_csv_path}")

# -------------------------------

formula = ("Avg_Alpha_Power ~ Q('PHQ-9') + age + Q('CTQ-SF') + Q('LES') + "
           "Q('SSRS') + Q('GAD-7') + Q('PSQI') + C(gender)")
model = smf.ols(formula, data=df).fit()
regression_summary = model.summary().as_text()
regression_path = os.path.join(output_dir, "regression_summary_Avg_Alpha_Power.txt")
with open(regression_path, "w") as f:
    f.write(regression_summary)
print(f"\nRegression summary saved to: {regression_path}")


corr_vars = ["Avg_Alpha_Power", "age", "PHQ-9", "CTQ-SF", "LES", "SSRS", "GAD-7", "PSQI", "PCA1", "PCA2"]
df_corr = df[corr_vars].copy()
corr_matrix = df_corr.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
plt.savefig(heatmap_path)
plt.show()
print(f"Correlation heatmap saved to: {heatmap_path}")


features = ["Avg_Alpha_Power", "PCA1", "PCA2"]
X = df[features].values
y = df["Group_binary"].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(LogisticRegression(), X_scaled, y, cv=cv)


cm = confusion_matrix(y, y_pred_cv)
print("\nConfusion Matrix:")
print(cm)
report = classification_report(y, y_pred_cv, digits=3)
print("\nClassification Report:")
print(report)
report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("Confusion Matrix:\n" + np.array2string(cm) + "\n\n" + report)
print(f"Classification report saved to: {report_path}")


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_probs = np.zeros(len(y))
for train_idx, test_idx in skf.split(X_scaled, y):
    model_fold = LogisticRegression()
    model_fold.fit(X_scaled[train_idx], y[train_idx])
    probs = model_fold.predict_proba(X_scaled[test_idx])[:, 1]
    y_probs[test_idx] = probs

fpr, tpr, _ = roc_curve(y, y_probs)
roc_auc = auc(fpr, tpr)
print(f"\nROC AUC: {roc_auc:.4f}")

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()
roc_path = os.path.join(output_dir, "logistic_roc_curve.png")
plt.savefig(roc_path)
plt.show()
print(f"ROC curve saved to: {roc_path}")


features_for_cluster = ["Avg_Alpha_Power", "PCA1", "PCA2"]
X_cluster = df[features_for_cluster].values
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)


kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_cluster_scaled)
df["Cluster"] = clusters


plt.figure(figsize=(8, 6))
plt.scatter(df["PCA1"], df["PCA2"], c=clusters, cmap="viridis", edgecolor="k", s=100)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("KMeans Clustering on EEG Features (PCA Map)")


pca_for_plot = PCA(n_components=2)
X_cluster_pca = pca_for_plot.fit_transform(X_cluster_scaled)
centers_pca = pca_for_plot.transform(kmeans.cluster_centers_)
plt

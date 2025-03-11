import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt


data_excel = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\EEG_bandpower_features.xlsx"
output_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\interpretability"
os.makedirs(output_folder, exist_ok=True)


df = pd.read_excel(data_excel)
print("Data loaded. Columns:", df.columns.tolist())


if 'Group' not in df.columns:
    raise KeyError("Missing 'Group' column. Please merge EEG features with metadata that contains group labels.")


if df['Group'].dtype == 'object':
    df['Group'] = df['Group'].map({'MDD': 1, 'HC': 0})


feature_cols = ["delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"]


df = df.dropna(subset=feature_cols + ['Group'])


X = df[feature_cols].values
y = df['Group'].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


clf = LogisticRegression(solver='liblinear', random_state=42)
clf.fit(X_scaled, y)


explainer = shap.LinearExplainer(clf, X_scaled, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_scaled)


plt.figure()
shap.summary_plot(shap_values, X_scaled, feature_names=feature_cols, show=False)
summary_plot_path = os.path.join(output_folder, "shap_summary.png")
plt.savefig(summary_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print("SHAP summary plot saved to:", summary_plot_path)


plt.figure()
shap.force_plot(explainer.expected_value, shap_values[0], X_scaled[0], feature_names=feature_cols, matplotlib=True, show=False)
force_plot_path = os.path.join(output_folder, "shap_force_plot_instance0.png")
plt.savefig(force_plot_path, dpi=150, bbox_inches='tight')
plt.close()
print("SHAP force plot for instance 0 saved to:", force_plot_path)


shap_df = pd.DataFrame(shap_values, columns=feature_cols)
shap_df.insert(0, "Subject", df["Subject"])
output_shap_excel = os.path.join(output_folder, "shap_values.xlsx")
shap_df.to_excel(output_shap_excel, index=False)
print("SHAP values saved to Excel file:", output_shap_excel)

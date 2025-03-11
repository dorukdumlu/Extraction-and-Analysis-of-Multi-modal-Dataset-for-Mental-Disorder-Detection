import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import logit
from sklearn.metrics import confusion_matrix, classification_report


merged_csv_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_3channels_resting_lanzhou_2015\merged_EEG_3channels_metadata.csv"
output_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_3channels_resting_lanzhou_2015\output"
# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


df_merged = pd.read_csv(merged_csv_path, delimiter=';', decimal='.')
print("Data loaded. Columns:\n", df_merged.columns)


df_mdd = df_merged[df_merged["type"] == "MDD"]
df_hc  = df_merged[df_merged["type"] == "HC"]

bands = ["Mean_Alpha_Power", "Mean_Beta_Power", "Mean_Theta_Power", "Mean_Delta_Power"]

ttest_results = []
for band in bands:
    mdd_vals = df_mdd[band].dropna()
    hc_vals  = df_hc[band].dropna()

    t_stat, p_val = ttest_ind(mdd_vals, hc_vals, equal_var=False, nan_policy='omit')
    ttest_results.append([band, t_stat, p_val])


df_ttest = pd.DataFrame(ttest_results, columns=["Band", "t_stat", "p_value"])
print("\n=== T-Test Results ===")
print(df_ttest)

ttest_csv_path = os.path.join(output_folder, "ttest_results_3channels.csv")
df_ttest.to_csv(ttest_csv_path, index=False, sep=';')
print(f"T-test results saved to: {ttest_csv_path}")


corr_vars = [
    "Mean_Alpha_Power", "Mean_Beta_Power", "Mean_Theta_Power", "Mean_Delta_Power",
    "PHQ-9", "CTQ-SF", "LES", "SSRS", "GAD-7", "PSQI"
]

corr_vars = [v for v in corr_vars if v in df_merged.columns]

df_for_corr = df_merged[corr_vars].dropna()
corr_matrix = df_for_corr.corr()

print("\n=== Correlation Matrix ===")
print(corr_matrix)

corr_csv_path = os.path.join(output_folder, "correlation_matrix_3channels.csv")
corr_matrix.to_csv(corr_csv_path, index=True, sep=';')
print(f"Correlation matrix saved to: {corr_csv_path}")

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0)
plt.title("Correlation: EEG(3ch) & Clinical Variables")
plt.tight_layout()
heatmap_path = os.path.join(output_folder, "correlation_heatmap_3channels.png")
plt.savefig(heatmap_path, dpi=150)
plt.show()
print(f"Correlation heatmap saved to: {heatmap_path}")

# Create a binary column
df_merged["Group_binary"] = df_merged["type"].map({"MDD": 1, "HC": 0})

predictors = ["Mean_Alpha_Power", "Mean_Beta_Power", "Mean_Theta_Power", "Mean_Delta_Power"]



df_logit = df_merged.dropna(subset=["Group_binary"] + predictors).copy()


formula = "Group_binary ~ " + " + ".join([f"Q('{p}')" for p in predictors])
print("\n=== Logistic Regression ===")
print("Using formula:", formula)

try:
    # Fit model
    model = logit(formula, data=df_logit).fit(disp=False)
    summary_str = model.summary().as_text()
    print(summary_str)


    logit_summary_path = os.path.join(output_folder, "logit_summary_3channels.txt")
    with open(logit_summary_path, "w", encoding="utf-8") as f:
        f.write(summary_str)
    print(f"Logistic regression summary saved to: {logit_summary_path}")


    df_logit["pred_prob"] = model.predict(df_logit[predictors])
    df_logit["pred_class"] = (df_logit["pred_prob"] >= 0.5).astype(int)


    cm = confusion_matrix(df_logit["Group_binary"], df_logit["pred_class"])
    cr = classification_report(df_logit["Group_binary"], df_logit["pred_class"], target_names=["HC","MDD"])

    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)


    logit_eval_path = os.path.join(output_folder, "logit_evaluation_3channels.txt")
    with open(logit_eval_path, "w", encoding="utf-8") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(cr + "\n")

    print(f"Logistic regression evaluation saved to: {logit_eval_path}")

except Exception as e:
    print("Logistic regression failed:", e)

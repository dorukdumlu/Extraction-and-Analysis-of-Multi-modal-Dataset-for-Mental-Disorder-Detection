
import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA


data_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\EEG\resting state.xlsx"
output_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\EEG\resting state further analysis"
if not os.path.exists(output_path):
    os.makedirs(output_path)


subject_col = "subject"
filename_col = "filename"
group_col = "Group"
power_cols = ["delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"]


phq_col = "PHQ9"
gad_col = "GAD7"
psqi_col = "PSQI"
age_col = "Age"

df = pd.read_excel(data_path)
print("Columns in the dataset:")
print(df.columns.tolist())


rename_map = {
    "Patient Health Questionnaire-9 (PHQ-9)": "PHQ9",
    "Generalized Anxiety Disorder, GAD-7": "GAD7",
    "Pittsburgh Sleep Quality Index,PSQI": "PSQI"
    # add more renames if needed
}
df.rename(columns=rename_map, inplace=True)


all_cols_to_numeric = power_cols + [phq_col, gad_col, psqi_col, age_col]
for c in all_cols_to_numeric:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows with no group info
df = df.dropna(subset=[group_col])


desc_stats = df[power_cols].describe().T
desc_stats["median"] = df[power_cols].median()
desc_stats["range"] = df[power_cols].max() - df[power_cols].min()
desc_stats_file = os.path.join(output_path, "descriptive_stats_power.csv")
desc_stats.to_csv(desc_stats_file, sep=";")
print("Descriptive statistics for power measures saved to:", desc_stats_file)


group_output = []
for col in power_cols:
    mdd_vals = df.loc[df[group_col] == "MDD", col].dropna()
    hc_vals  = df.loc[df[group_col] == "HC",  col].dropna()

    if len(mdd_vals) < 2 or len(hc_vals) < 2:
        msg = f"{col}: Not enough data for group comparison."
        group_output.append(msg)
        print(msg)
        continue

    # Independent t-test
    t_stat, p_val = stats.ttest_ind(mdd_vals, hc_vals, equal_var=False)

    # Mann-Whitney U
    u_stat, p_val_u = stats.mannwhitneyu(mdd_vals, hc_vals, alternative="two-sided")

    result_str = (
        f"\nMeasure: {col}\n"
        f"  MDD mean={mdd_vals.mean():.4f}, HC mean={hc_vals.mean():.4f}\n"
        f"  t-test: t={t_stat:.3f}, p={p_val:.4f}\n"
        f"  Mann-Whitney U: U={u_stat:.3f}, p={p_val_u:.4f}\n"
    )
    group_output.append(result_str)
    print(result_str)

group_comp_file = os.path.join(output_path, "group_comparisons_power.txt")
with open(group_comp_file, "w") as f:
    for line in group_output:
        f.write(line + "\n")
print("Group comparisons for power measures saved to:", group_comp_file)


corr_output = []
clinical_cols = [phq_col, gad_col, psqi_col]
for power_col in power_cols:
    for clin_col in clinical_cols:
        if clin_col not in df.columns:
            continue
        valid_df = df[[power_col, clin_col]].dropna()
        if len(valid_df) < 2:
            msg = f"Not enough data to correlate {power_col} with {clin_col}."
            corr_output.append(msg)
            print(msg)
            continue
        r_val, p_val = stats.pearsonr(valid_df[power_col], valid_df[clin_col])
        msg = (f"Correlation between {power_col} and {clin_col}: "
               f"r={r_val:.3f}, p={p_val:.4f}")
        corr_output.append(msg)
        print(msg)

corr_file = os.path.join(output_path, "clinical_correlations_power.txt")
with open(corr_file, "w") as f:
    for line in corr_output:
        f.write(line + "\n")
print("Correlations with clinical measures saved to:", corr_file)


df["Group_Code"] = df[group_col].map({"MDD": 1, "HC": 0})
log_df = df[power_cols + ["Group_Code"]].dropna()
logistic_output = ""
if len(log_df["Group_Code"].unique()) == 2:
    X_log = log_df[power_cols]
    y_log = log_df["Group_Code"]

    if len(X_log) > 1:
        logreg = LogisticRegression(solver="liblinear")
        logreg.fit(X_log, y_log)
        y_pred = logreg.predict(X_log)
        acc_score = accuracy_score(y_log, y_pred)
        cmatrix = confusion_matrix(y_log, y_pred)
        report = classification_report(y_log, y_pred)
        logistic_output = (
            f"Logistic Regression Accuracy: {acc_score:.3f}\n"
            f"Confusion Matrix:\n{cmatrix}\n"
            f"Classification Report:\n{report}"
        )
        print(logistic_output)
    else:
        logistic_output = "Not enough rows for logistic regression after dropping NaNs."
else:
    logistic_output = "Group column does not have exactly two categories (MDD vs. HC)."
log_file = os.path.join(output_path, "logistic_regression_power.txt")
with open(log_file, "w") as f:
    f.write(logistic_output)
print("Logistic regression results saved to:", log_file)


pca_df = df[power_cols].dropna()
if len(pca_df) > 1:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pca_df)
    var_ratio = pca.explained_variance_ratio_
    pca_summary = f"Explained Variance Ratio (PC1, PC2): {var_ratio}"
    print(pca_summary)
    pca_result_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    pca_csv = os.path.join(output_path, "PCA_power.csv")
    pca_result_df.to_csv(pca_csv, sep=";", index=False)
    with open(os.path.join(output_path, "pca_summary_power.txt"), "w") as f:
        f.write(pca_summary)
    print("PCA results saved to:", pca_csv)
else:
    print("Not enough data for PCA on power measures.")

print("\nPower analysis complete. All results have been saved to your chosen output folder.")

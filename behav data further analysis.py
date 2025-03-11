import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA


file_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\Behavioral_Data\behavioral_data_updated.xlsx"
output_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\Behavioral_Data\output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


subject_col   = "subject"
group_col     = "Group"          # Expected values: "MDD" and "HC"
condition_col = "CellNumber"     # Factor for repeated measures

p_rt_col = "PWaitResp.RT"
p_acc_col = "PWaitResp.ACC"
performance_cols = [p_rt_col, p_acc_col]

df = pd.read_excel(file_path)
print("Columns in the dataset:")
print(df.columns.tolist())


df[performance_cols] = df[performance_cols].apply(pd.to_numeric, errors="coerce")


print("\n--- Descriptive Statistics for Key Performance Measures ---")
desc_stats = df[performance_cols].describe().T
desc_stats["median"] = df[performance_cols].median()
desc_stats["range"]  = df[performance_cols].max() - df[performance_cols].min()
print(desc_stats)

desc_stats_file = os.path.join(output_folder, "descriptive_stats_behavioral.csv")
desc_stats.to_csv(desc_stats_file, sep=";")
print(f"Descriptive stats saved to: {desc_stats_file}")


group_comparison_output = []
for col in performance_cols:
    mdd_vals = df.loc[df[group_col] == "MDD", col].dropna()
    hc_vals  = df.loc[df[group_col] == "HC", col].dropna()

    if len(mdd_vals) == 0 or len(hc_vals) == 0:
        group_comparison_output.append(f"{col}: Not enough data for MDD/HC comparison.")
        continue

    t_stat, p_val = stats.ttest_ind(mdd_vals, hc_vals, equal_var=False)
    u_stat, p_val_u = stats.mannwhitneyu(mdd_vals, hc_vals, alternative="two-sided")
    result_str = (
        f"\nMeasure: {col}\n"
        f"  t-test: t = {t_stat:.3f}, p = {p_val:.4f}\n"
        f"  Mann-Whitney U: U = {u_stat:.3f}, p = {p_val_u:.4f}"
    )
    group_comparison_output.append(result_str)
    print(result_str)

group_comp_file = os.path.join(output_folder, "group_comparisons.txt")
with open(group_comp_file, "w") as f:
    for line in group_comparison_output:
        f.write(line + "\n")
print(f"\nGroup comparisons saved to: {group_comp_file}")


print("\n--- Repeated Measures ANOVA (Example on PWaitResp.RT) ---")

aggregated_df = df.groupby([subject_col, condition_col], as_index=False)[p_rt_col].mean()


unique_conditions = aggregated_df[condition_col].nunique()


subject_counts = aggregated_df.groupby(subject_col).size()
balanced_subjects = subject_counts[subject_counts == unique_conditions].index
aggregated_balanced = aggregated_df[aggregated_df[subject_col].isin(balanced_subjects)]

if aggregated_balanced.empty:
    anova_results_str = "No subjects have balanced data across all conditions."
    print(anova_results_str)
else:
    try:
        aov = AnovaRM(
            data=aggregated_balanced,
            depvar=p_rt_col,
            subject=subject_col,
            within=[condition_col]
        ).fit()
        anova_summary = aov.summary().as_text()
        anova_results_str = "Repeated Measures ANOVA results:\n" + anova_summary
        print(anova_results_str)
    except Exception as e:
        anova_results_str = f"Error in repeated measures ANOVA: {e}"
        print(anova_results_str)

anova_file = os.path.join(output_folder, "repeated_measures_anova.txt")
with open(anova_file, "w") as f:
    f.write(anova_results_str)
print(f"\nRepeated measures ANOVA results saved to: {anova_file}")


"""
print("\n--- Correlations with Clinical Measures ---")
correlation_output = []
for col in performance_cols:
    valid = df[[col, phq_col, gad_col]].dropna()
    if len(valid) < 2:
        correlation_output.append(f"Not enough data for correlation with {col}")
        continue
    r_phq, p_phq = stats.pearsonr(valid[col], valid[phq_col])
    r_gad, p_gad = stats.pearsonr(valid[col], valid[gad_col])
    result_str = (
        f"\nCorrelation for {col}:\n"
        f"  vs. {phq_col}: r = {r_phq:.3f}, p = {p_phq:.4f}\n"
        f"  vs. {gad_col}: r = {r_gad:.3f}, p = {p_gad:.4f}"
    )
    correlation_output.append(result_str)
    print(result_str)
corr_file = os.path.join(output_folder, "clinical_correlations.txt")
with open(corr_file, "w") as f:
    for line in correlation_output:
        f.write(line + "\n")
print(f"\nCorrelation results saved to: {corr_file}")
"""


"""
print("\n--- Linear Regression: Predict PHQ9 ---")
regression_output = ""
reg_df = df[[phq_col, p_rt_col, p_acc_col]].dropna()
if len(reg_df) > 1:
    X = reg_df[[p_rt_col, p_acc_col]]
    X = sm.add_constant(X)
    y = reg_df[phq_col]
    lin_model = sm.OLS(y, X).fit()
    regression_output = lin_model.summary().as_text()
    print(regression_output)
else:
    regression_output = "Not enough data to run linear regression."
reg_file = os.path.join(output_folder, "linear_regression_behavioral.txt")
with open(reg_file, "w") as f:
    f.write(regression_output)
print(f"\nLinear regression results saved to: {reg_file}")
"""

print("\n--- Logistic Regression (MDD vs. HC) ---")
df["Group_Code"] = df[group_col].map({"MDD": 1, "HC": 0})
log_df = df[[p_rt_col, p_acc_col, "Group_Code"]].dropna()
logistic_output = ""
if len(log_df["Group_Code"].unique()) == 2:
    X_log = log_df[[p_rt_col, p_acc_col]]
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
log_file = os.path.join(output_folder, "logistic_regression_behavioral.txt")
with open(log_file, "w") as f:
    f.write(logistic_output)
print(f"\nLogistic regression results saved to: {log_file}")


print("\n--- PCA on Key Performance Measures ---")
pca_cols = [p_rt_col]
pca_df = df[pca_cols].dropna()
pca_output = ""
if len(pca_df) > 1:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pca_df)
    var_ratio = pca.explained_variance_ratio_
    pca_output = f"Explained Variance Ratio: {var_ratio}"
    print(pca_output)
    pca_result_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    pca_csv = os.path.join(output_folder, "PCA_behavioral.csv")
    pca_result_df.to_csv(pca_csv, sep=";", index=False)
    print(f"PCA results saved to: {pca_csv}")
else:
    pca_output = "Not enough rows for PCA after dropping NaNs."
pca_file = os.path.join(output_folder, "pca_summary.txt")
with open(pca_file, "w") as f:
    f.write(pca_output)

print("\nBehavioral analysis complete. All results have been saved to the output folder.")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")


file_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\128channels_resting\output\final_merged_EEG_128channels_analysis.xlsx"
df = pd.read_excel(file_path)


if "subject" in df.columns:
    df["subject"] = df["subject"].astype(str).str.strip()


print("Columns in dataset:")
print(df.columns.tolist())


df = df.rename(columns={
    "PHQ-9": "PHQ_9",
    "CTQ-SF": "CTQ_SF",
    "LES": "LES",
    "SSRS": "SSRS",
    "GAD-7": "GAD_7",
    "PSQI": "PSQI",
    "education（years）": "education_years"
})


alpha_cols = ["Ch1_Alpha_Power", "Avg_Alpha_Power", "PCA1", "PCA2"]
clinical_cols = ["PHQ_9", "GAD_7", "PSQI", "age", "education_years"]
use_cols = [col for col in alpha_cols + clinical_cols if col in df.columns]
df_subset = df[use_cols].dropna()


pearson_corr = df_subset.corr(method="pearson")
print("\nPearson Correlation Matrix:")
print(pearson_corr)
# Save to a text file
with open("correlation_results.txt", "w") as f:
    f.write("Pearson Correlation Matrix:\n")
    f.write(pearson_corr.to_string())


plt.figure(figsize=(8, 6))
sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", center=0)
plt.title("Pearson Correlation: EEG Alpha & Clinical Metrics")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()


spearman_corr = df_subset.corr(method="spearman")
print("\nSpearman Correlation Matrix:")
print(spearman_corr)
with open("spearman_correlation_results.txt", "w") as f:
    f.write("Spearman Correlation Matrix:\n")
    f.write(spearman_corr.to_string())

plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", center=0)
plt.title("Spearman Correlation: EEG Alpha & Clinical Metrics")
plt.tight_layout()
plt.savefig("spearman_correlation_heatmap.png")
plt.close()


if "Avg_Alpha_Power" in df_subset.columns and "PHQ_9" in df_subset.columns:
    plt.figure(figsize=(6, 4))
    sns.regplot(x="Avg_Alpha_Power", y="PHQ_9", data=df_subset, ci=95, scatter_kws={'s': 40})
    plt.xlabel("Average Alpha Power")
    plt.ylabel("PHQ-9 Score")
    plt.title("Avg_Alpha_Power vs PHQ_9")
    plt.tight_layout()
    plt.savefig("scatter_AvgAlpha_PHQ9.png")
    plt.close()
else:
    print("Either 'Avg_Alpha_Power' or 'PHQ_9' not found in dataset.")


ols_cols = ["Avg_Alpha_Power", "PHQ_9", "CTQ_SF", "LES", "SSRS", "GAD_7", "PSQI", "age", "gender"]
for col in ols_cols:
    if col not in df.columns:
        print(f"Warning: {col} not found in dataset for OLS predicting Avg_Alpha_Power")


ols_formula = "Avg_Alpha_Power ~ C(gender) + PHQ_9 + age + CTQ_SF + LES + SSRS + GAD_7 + PSQI"
ols_model = smf.ols(formula=ols_formula, data=df).fit()
ols_summary = ols_model.summary().as_text()
print("\n=== OLS Regression: Predicting Avg_Alpha_Power ===")
print(ols_summary)
with open("ols_regression_avg_alpha.txt", "w") as f:
    f.write(ols_summary)


ols_formula2 = "PHQ_9 ~ C(gender) + Ch1_Alpha_Power + Avg_Alpha_Power + age"
ols_model2 = smf.ols(formula=ols_formula2, data=df).fit()
ols_summary2 = ols_model2.summary().as_text()
print("\n=== OLS Regression: Predicting PHQ_9 ===")
print(ols_summary2)
with open("ols_regression_PHQ9.txt", "w") as f:
    f.write(ols_summary2)


if "Group" in df.columns:
    df["group_binary"] = df["Group"].map({"HC": 0, "MDD": 1})
else:
    print("Warning: 'Group' column not found for logistic regression.")

logit_formula = "group_binary ~ C(gender) + Ch1_Alpha_Power + Avg_Alpha_Power + age"
logit_model = smf.logit(formula=logit_formula, data=df).fit(disp=False)
logit_summary = logit_model.summary().as_text()
print("\n=== Logistic Regression Results ===")
print(logit_summary)
with open("logistic_regression_results.txt", "w") as f:
    f.write(logit_summary)


df["pred_prob"] = logit_model.predict(df)
df["pred_class"] = (df["pred_prob"] >= 0.5).astype(int)
if "group_binary" in df.columns:
    cm = confusion_matrix(df["group_binary"], df["pred_class"])
    cr = classification_report(df["group_binary"], df["pred_class"], target_names=["HC", "MDD"])
    print("\nConfusion Matrix (Logistic Regression):")
    print(cm)
    print("\nClassification Report (Logistic Regression):")
    print(cr)
    with open("logistic_classification_report.txt", "w") as f:
        f.write("Confusion Matrix:\n" + np.array2string(cm) + "\n\n")
        f.write("Classification Report:\n" + cr)
else:
    print("group_binary column not available for classification report.")

print("All correlation matrices, regression outputs, and figures have been saved to files.")

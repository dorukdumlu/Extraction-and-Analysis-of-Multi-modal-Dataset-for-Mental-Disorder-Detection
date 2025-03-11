import os
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


excel_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\audio\dot_probe_audio\dot probe output.xlsx"


output_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\audio\dot_probe_audio\output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


summary_file = os.path.join(output_folder, "analysis_summary.txt")


df = pd.read_excel(excel_path)
summary_text = "=== Data Preview ===\n" + df.head().to_string() + "\n\n"


df["Group_Code"] = df["Group"].map({"MDD": 1, "HC": 0})


col_condA = "ERP_CondA_Mean"
col_condB = "ERP_CondB_Mean"
col_psd    = "Overall_PSD_Mean"
col_alpha  = "Alpha_Power"
col_phq    = "Patient Health Questionnaire-9 (PHQ-9)"
col_gad    = "Generalized Anxiety Disorder, GAD-7"
col_age    = "Age"

df_mdd = df[df["Group"] == "MDD"].copy()
df_hc  = df[df["Group"] == "HC"].copy()

def cohen_d(x, y):

    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1)*np.std(x, ddof=1)**2 + (ny - 1)*np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

summary_text += "=== Group Comparisons: MDD vs. HC ===\n"
features_to_test = [col_condA, col_condB, col_psd, col_alpha]

for feat in features_to_test:
    x_mdd = df_mdd[feat].dropna()
    x_hc  = df_hc[feat].dropna()
    t_stat, p_val = stats.ttest_ind(x_mdd, x_hc, equal_var=False)
    d_val = cohen_d(x_mdd, x_hc)
    summary_text += f"{feat}: t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={d_val:.3f}\n"
summary_text += "\n"


summary_text += "=== Within-Subject Comparison: ConditionA vs. ConditionB ===\n"
# Overall subjects
t_stat, p_val = stats.ttest_rel(df[col_condA], df[col_condB])
summary_text += f"All Subjects: t={t_stat:.3f}, p={p_val:.4f}\n"


if len(df_mdd) > 0:
    t_stat_mdd, p_val_mdd = stats.ttest_rel(df_mdd[col_condA], df_mdd[col_condB])
    summary_text += f"MDD only: t={t_stat_mdd:.3f}, p={p_val_mdd:.4f}\n"

# HC only
if len(df_hc) > 0:
    t_stat_hc, p_val_hc = stats.ttest_rel(df_hc[col_condA], df_hc[col_condB])
    summary_text += f"HC only: t={t_stat_hc:.3f}, p={p_val_hc:.4f}\n"
summary_text += "\n"


summary_text += "=== Correlations with Clinical Measures (Pearson) ===\n"
measures_to_correlate = [col_condA, col_condB, col_alpha]
clinical_vars = [col_phq, col_gad]

for feat in measures_to_correlate:
    for clin in clinical_vars:
        valid_rows = df[[feat, clin]].dropna()
        if len(valid_rows) < 2:
            continue
        r_val, p_val = stats.pearsonr(valid_rows[feat], valid_rows[clin])
        summary_text += f"{feat} vs. {clin}: r={r_val:.3f}, p={p_val:.4f}\n"
summary_text += "\n"


summary_text += "=== Multiple Linear Regression: Predict PHQ-9 ===\n"
df_reg = df.dropna(subset=[col_phq, col_condA, col_alpha, col_age, "Group_Code"]).copy()
X = df_reg[[col_condA, col_alpha, col_age, "Group_Code"]]
y = df_reg[col_phq]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
reg_summary = model.summary().as_text()
summary_text += reg_summary + "\n\n"


summary_text += "=== Logistic Regression: Classify MDD vs. HC ===\n"
df_clf = df.dropna(subset=[col_condA, col_alpha, col_age, "Group_Code"]).copy()
X_clf = df_clf[[col_condA, col_alpha, col_age]]
y_clf = df_clf["Group_Code"]
logreg = LogisticRegression(solver="liblinear")
logreg.fit(X_clf, y_clf)
y_pred = logreg.predict(X_clf)
acc = accuracy_score(y_clf, y_pred)
cm = confusion_matrix(y_clf, y_pred)
clf_report = classification_report(y_clf, y_pred, target_names=["HC", "MDD"])
summary_text += f"Accuracy: {acc:.3f}\n"
summary_text += "Confusion Matrix:\n" + np.array2string(cm) + "\n"
summary_text += "Classification Report:\n" + clf_report + "\n"


with open(summary_file, "w") as f:
    f.write(summary_text)
print(f"Analysis summary saved to: {summary_file}")


cm_df = pd.DataFrame(cm, index=["HC", "MDD"], columns=["Predicted HC", "Predicted MDD"])
cm_csv = os.path.join(output_folder, "confusion_matrix.csv")
cm_df.to_csv(cm_csv)
print(f"Confusion matrix saved to: {cm_csv}")

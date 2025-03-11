import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import statsmodels.api as sm


def read_csv_fixed(path):

    try:
        df = pd.read_csv(path, sep=";", engine="python", encoding="utf-8")
    except Exception as e:
        print("Error reading CSV:", e)
        return None


    if df.shape[1] == 1:

        df = df.iloc[:, 0].str.split(";", expand=True)

        df.columns = df.iloc[0].str.strip()
        df = df[1:].reset_index(drop=True)
    df.columns = [col.strip() for col in df.columns]
    return df


def convert_numeric_columns(df):

    for col in df.columns:

        if df[col].dtype == object:

            df[col] = pd.to_numeric(df[col].str.replace(",", "").str.strip(), errors="coerce")
    return df


def compute_average_band_power(df, band_keyword):

    cols = [col for col in df.columns if band_keyword in col]
    if not cols:
        print(f"No columns found for {band_keyword}.")
        return pd.Series(np.nan, index=df.index)

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[cols].mean(axis=1)


def perform_ttests(df):

    groups = df["Group"].dropna().unique()
    if len(groups) != 2:
        print("T-tests require exactly two groups; found:", groups)
        return {}
    results = {}
    for band in ["Delta_Power", "Theta_Power", "Beta_Power"]:
        col_name = "Avg_" + band
        group1 = df[df["Group"] == groups[0]][col_name].dropna()
        group2 = df[df["Group"] == groups[1]][col_name].dropna()
        if len(group1) == 0 or len(group2) == 0:
            print(f"Not enough data for t-test on {col_name}.")
            continue
        t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
        results[col_name] = {"t_stat": t_stat, "p_value": p_val}
    return results


def perform_correlation_analysis(df):

    clinical_vars = ["PHQ-9", "CTQ-SF", "LES", "SSRS", "GAD-7", "PSQI", "age", "education?years?"]
    clinical_vars = [var for var in clinical_vars if var in df.columns]
    band_vars = ["Avg_Delta_Power", "Avg_Theta_Power", "Avg_Beta_Power"]
    vars_to_use = band_vars + clinical_vars
    return df[vars_to_use].corr(method="pearson")


def perform_logistic_regression(df):

    df_lr = df.copy()
    if "Group" not in df_lr.columns:
        print("No 'Group' column found. Skipping logistic regression.")
        return None, None, None
    unique_groups = df_lr["Group"].dropna().unique()
    if len(unique_groups) != 2:
        print("Logistic regression requires exactly two groups; found:", unique_groups)
        return None, None, None
    mapping = {unique_groups[0]: 0, unique_groups[1]: 1}
    df_lr["Group_binary"] = df_lr["Group"].map(mapping)
    predictors = ["Avg_Delta_Power", "Avg_Theta_Power", "Avg_Beta_Power", "age"]
    if "education?years?" in df_lr.columns:
        predictors.append("education?years?")
    if "gender" in df_lr.columns:
        df_lr["gender_encoded"] = df_lr["gender"].map({"M": 1, "F": 0})
        predictors.append("gender_encoded")
    df_lr = df_lr.dropna(subset=predictors + ["Group_binary"])
    if df_lr.empty:
        print("No data available after dropping NaNs for logistic regression.")
        return None, None, None
    X = df_lr[predictors]
    y = df_lr["Group_binary"]
    X_sm = sm.add_constant(X)
    try:
        model = sm.Logit(y, X_sm).fit(disp=False)
    except Exception as e:
        print("Logistic regression failed:", e)
        return None, None, None
    summary_str = model.summary().as_text()
    preds = model.predict(X_sm)
    pred_class = (preds >= 0.5).astype(int)
    cm = confusion_matrix(y, pred_class)
    cr = classification_report(y, pred_class, target_names=[str(unique_groups[0]), str(unique_groups[1])])
    return summary_str, cm, cr


def perform_roc_analysis(df):

    df_lr = df.copy()
    if "Group" not in df_lr.columns:
        return None, None, None
    unique_groups = df_lr["Group"].dropna().unique()
    if len(unique_groups) != 2:
        return None, None, None
    mapping = {unique_groups[0]: 0, unique_groups[1]: 1}
    df_lr["Group_binary"] = df_lr["Group"].map(mapping)
    predictors = ["Avg_Delta_Power", "Avg_Theta_Power", "Avg_Beta_Power", "age"]
    if "education?years?" in df_lr.columns:
        predictors.append("education?years?")
    if "gender" in df_lr.columns:
        df_lr["gender_encoded"] = df_lr["gender"].map({"M": 1, "F": 0})
        predictors.append("gender_encoded")
    df_lr = df_lr.dropna(subset=predictors + ["Group_binary"])
    if df_lr.empty:
        return None, None, None
    X = df_lr[predictors]
    y = df_lr["Group_binary"]
    X_sm = sm.add_constant(X)
    try:
        model = sm.Logit(y, X_sm).fit(disp=False)
    except Exception as e:
        print("ROC analysis failed due to logistic regression error:", e)
        return None, None, None
    preds = model.predict(X_sm)
    try:
        fpr, tpr, thresholds = roc_curve(y, preds)
        roc_auc = auc(fpr, tpr)
    except Exception as e:
        print("ROC computation failed:", e)
        return None, None, None
    return fpr, tpr, roc_auc



def main():

    merged_csv_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\128channels_resting\output\beta theta delta.csv"
    output_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\128channels_resting\output\analysis_output"
    os.makedirs(output_folder, exist_ok=True)


    df = read_csv_fixed(merged_csv_path)
    if df is None:
        print("Failed to load data.")
        return
    print("Data loaded. Columns:", df.columns.tolist())
    df = convert_numeric_columns(df)


    df_avg = pd.DataFrame({
        "Avg_Delta_Power": compute_average_band_power(df, "Delta_Power"),
        "Avg_Theta_Power": compute_average_band_power(df, "Theta_Power"),
        "Avg_Beta_Power": compute_average_band_power(df, "Beta_Power")
    })
    df = pd.concat([df.reset_index(drop=True), df_avg], axis=1)

    updated_csv_path = os.path.join(output_folder, "updated_data_with_avg_band_power.csv")
    df.to_csv(updated_csv_path, index=False, sep=";")
    print("Updated CSV saved to:", updated_csv_path)


    if "Group" in df.columns:
        groups = df["Group"].dropna().unique()
        if len(groups) == 2:
            ttest_results = perform_ttests(df)
            if ttest_results:
                ttest_df = pd.DataFrame(ttest_results).T
                ttest_csv_path = os.path.join(output_folder, "ttest_results.csv")
                ttest_df.to_csv(ttest_csv_path, sep=";")
                print("T-test results saved to:", ttest_csv_path)
        else:
            print("T-tests require exactly two groups; found:", groups)
    else:
        print("No 'Group' column found, skipping t-tests...")


    corr_df = perform_correlation_analysis(df)
    corr_csv_path = os.path.join(output_folder, "correlation_matrix.csv")
    corr_df.to_csv(corr_csv_path, sep=";")
    print("Correlation matrix saved to:", corr_csv_path)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", center=0)
    plt.title("Pearson Correlation Matrix")
    heatmap_path = os.path.join(output_folder, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print("Correlation heatmap saved to:", heatmap_path)


    logit_summary, cm, cr = perform_logistic_regression(df)
    if logit_summary is not None:
        logit_summary_path = os.path.join(output_folder, "logistic_regression_summary.txt")
        with open(logit_summary_path, "w", encoding="utf-8") as f:
            f.write(logit_summary)
        classification_report_path = os.path.join(output_folder, "classification_report.txt")
        with open(classification_report_path, "w", encoding="utf-8") as f:
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm))
            f.write("\n\nClassification Report:\n")
            f.write(cr)
        print("Logistic regression summary and classification report saved.")
    else:
        print("Logistic regression was not performed due to missing or insufficient group data.")


    roc_results = perform_roc_analysis(df)
    if roc_results is not None:
        fpr, tpr, roc_auc = roc_results
        if roc_auc is not None:
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label="ROC curve (AUC = {:.2f})".format(roc_auc))
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            roc_curve_path = os.path.join(output_folder, "roc_curve.png")
            plt.savefig(roc_curve_path, dpi=300)
            plt.close()
            print("ROC curve saved to:", roc_curve_path)
        else:
            print("ROC analysis returned None for AUC. Skipping ROC plot.")
    else:
        print("ROC analysis was not performed due to logistic regression issues.")

    if "Avg_Alpha_Power" in df.columns and "PHQ-9" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.regplot(x="Avg_Alpha_Power", y="PHQ-9", data=df, ci=95, scatter_kws={'s': 40})
        plt.xlabel("Average Alpha Power")
        plt.ylabel("PHQ-9 Score")
        plt.title("Avg_Alpha_Power vs PHQ-9")
        scatter_path = os.path.join(output_folder, "scatter_AvgAlpha_PHQ9.png")
        plt.savefig(scatter_path, dpi=300)
        plt.close()
        print("Scatter plot saved to:", scatter_path)
    else:
        print("Either 'Avg_Alpha_Power' or 'PHQ-9' not found; skipping scatter plot.")

    print("All analyses completed. Check the output folder:", output_folder)


if __name__ == "__main__":
    main()

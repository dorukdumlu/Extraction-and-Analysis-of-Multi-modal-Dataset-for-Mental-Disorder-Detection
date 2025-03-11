import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm


def read_with_encodings(file_path, encodings, sep=";", quotechar='"', skiprows=0, header=0):
    """
    Attempt to read a CSV file using multiple encodings until one works.
    """
    for enc in encodings:
        try:
            df = pd.read_csv(
                file_path,
                encoding=enc,
                sep=sep,
                quotechar=quotechar,
                skiprows=skiprows,
                header=header,
                engine="python"
            )
            if df.shape[1] > 1:
                print(f"Successfully read '{os.path.basename(file_path)}' with encoding='{enc}'")
                return df
        except Exception as e:
            print(f"Failed with encoding='{enc}': {e}")
    raise ValueError(f"Could not read '{file_path}' with any provided encodings.")


def analyze_rt_columns(df):
    """
    Clean column names, identify RT columns, convert them to numeric,
    and return a summary DataFrame with mean, std, min, max, and count.
    """
    # Remove stray quotes and extra whitespace
    df.columns = df.columns.str.replace('"', '', regex=False).str.strip()
    print("Cleaned Columns:")
    print(df.columns.tolist())

    # Identify potential RT columns (e.g., columns that contain ".RT")
    rt_cols = [col for col in df.columns if ".RT" in col]
    print("RT columns detected:", rt_cols)

    summary_data = {}
    for col in rt_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        summary_data[col] = {
            "mean": df[col].mean(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
            "count": df[col].count()
        }
        print(f"Summary for {col}: {summary_data[col]}")
    rt_summary_df = pd.DataFrame(summary_data).T
    return rt_summary_df, rt_cols


def detect_outliers(df, rt_column, threshold=3):
    """
    Compute z-scores for a given RT column and return rows where the absolute z-score exceeds the threshold.
    """
    # Ensure the column is numeric
    df[rt_column] = pd.to_numeric(df[rt_column], errors="coerce")
    # Compute z-scores (drop missing values for calculation, then join back)
    z_scores = zscore(df[rt_column].dropna())
    # Create a DataFrame mapping index to zscore
    z_df = pd.DataFrame({"zscore": z_scores}, index=df[rt_column].dropna().index)
    # Merge z-scores back into original DataFrame
    df_with_z = df.join(z_df, how="left")
    outliers = df_with_z[np.abs(df_with_z["zscore"]) > threshold]
    print(f"Detected {outliers.shape[0]} outliers in column {rt_column} (|z| > {threshold}).")
    return outliers


def run_regression(df, predictor, outcome):
    """
    Run a linear regression with predictor variable predicting outcome.
    Returns the summary text as a string.
    """
    # Ensure both columns are numeric
    df[predictor] = pd.to_numeric(df[predictor], errors="coerce")
    df[outcome] = pd.to_numeric(df[outcome], errors="coerce")
    # Drop rows with missing values in predictor or outcome
    reg_df = df[[predictor, outcome]].dropna()
    if reg_df.empty:
        return "Not enough data for regression."
    X = reg_df[predictor]
    X = sm.add_constant(X)
    y = reg_df[outcome]
    model = sm.OLS(y, X).fit()
    summary_text = model.summary().as_text()
    print("Regression Summary:")
    print(summary_text)
    return summary_text


if __name__ == "__main__":
    # Set the path to your combined CSV file
    file_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\Behavioral_Data\combined_data.csv"
    # Adjust skiprows/header if necessary (e.g., skip an extra row if needed)
    SKIPROWS = 0
    HEADER_ROW = 0

    encodings_to_try = ["utf-8", "utf-16", "cp1252", "latin-1"]
    try:
        df = read_with_encodings(file_path, encodings_to_try, sep=";", quotechar='"', skiprows=SKIPROWS,
                                 header=HEADER_ROW)
    except Exception as e:
        print("Failed to read the file:", e)
        exit()

    # Perform RT analysis
    rt_summary_df, rt_cols = analyze_rt_columns(df)

    # For further analysis, choose one RT column (if available)
    chosen_rt = None
    for col in rt_cols:
        if "PWaitResp.RT" in col:
            chosen_rt = col
            break
    if chosen_rt is None and rt_cols:
        chosen_rt = rt_cols[0]


    outliers_df = None
    if chosen_rt is not None:
        outliers_df = detect_outliers(df, chosen_rt, threshold=3)


    regression_results = ""
    if "Age" in df.columns and chosen_rt is not None:
        regression_results = run_regression(df, predictor="Age", outcome=chosen_rt)
    else:
        regression_results = "Required columns for regression (e.g., Age and chosen RT) not found."


    output_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\Behavioral_Data"
    os.makedirs(output_folder, exist_ok=True)
    output_excel = os.path.join(output_folder, "combined_data_analysis.xlsx")


    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        # Original combined data sheet
        df.to_excel(writer, sheet_name="Combined Data", index=False)
        # RT summary sheet
        rt_summary_df.to_excel(writer, sheet_name="RT Summary")

        if outliers_df is not None and not outliers_df.empty:
            outliers_df.to_excel(writer, sheet_name="Outliers", index=False)
        else:
            pd.DataFrame({"Message": ["No outliers detected or insufficient data."]}).to_excel(writer,
                                                                                               sheet_name="Outliers",
                                                                                               index=False)

        reg_df = pd.DataFrame({"Regression Summary": regression_results.split("\n")})
        reg_df.to_excel(writer, sheet_name="Regression Results", index=False)

    print("Saved complete analysis to", output_excel)

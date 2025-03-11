import os
import pandas as pd
import statsmodels.formula.api as smf

# ------------------------------------------------------------------------
# 1. Configuration: File paths and column names
# ------------------------------------------------------------------------
file_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\Behavioral_Data\behavioral_data_updated.xlsx"
output_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\Behavioral_Data\output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Column names based on your file header
subject_col = "subject"         # Subject identifier
condition_col = "CellNumber"      # Trial condition indicator
rt_col = "PWaitResp.RT"           # Reaction time column

# ------------------------------------------------------------------------
# 2. Load Data and Convert Key Columns
# ------------------------------------------------------------------------
df = pd.read_excel(file_path)
print("Columns in the dataset:")
print(df.columns.tolist())

# Convert the reaction time column to numeric (coerce non-numeric values to NaN)
df[rt_col] = pd.to_numeric(df[rt_col], errors="coerce")

# ------------------------------------------------------------------------
# 3. Aggregate Data by Subject and Condition
# ------------------------------------------------------------------------
# Compute the mean reaction time for each subject-condition pair
aggregated_df = df.groupby([subject_col, condition_col], as_index=False)[rt_col].mean()
print("Aggregated data (first 5 rows):")
print(aggregated_df.head())


num_conditions = aggregated_df[condition_col].nunique()
print(f"Number of unique conditions: {num_conditions}")


subject_counts = aggregated_df.groupby(subject_col).size()
balanced_subjects = subject_counts[subject_counts == num_conditions].index
aggregated_balanced = aggregated_df[aggregated_df[subject_col].isin(balanced_subjects)]
print(f"Number of subjects with balanced data: {len(balanced_subjects)}")


if aggregated_balanced.empty:
    print("No subjects have balanced data across all conditions.")
    model_summary_text = "No subjects have balanced data across all conditions."
else:

    model = smf.mixedlm(f"{rt_col} ~ C({condition_col})", aggregated_balanced, groups=aggregated_balanced[subject_col])
    mixed_model = model.fit()
    model_summary_text = mixed_model.summary().as_text()
    print(model_summary_text)



output_file = os.path.join(output_folder, "mixed_effects_model_behavioral.txt")
with open(output_file, "w") as f:
    f.write(model_summary_text)
print(f"Mixed effects model results saved to: {output_file}")

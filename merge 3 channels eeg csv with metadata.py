import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


eeg_csv_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\EEG_3channels_resting_lanzhou_2015\merged_EEG_3channels_metadata.csv"
df_eeg = pd.read_csv(eeg_csv_path, delimiter=';', decimal='.')
print("EEG Features (first 5 rows):")
print(df_eeg.head())
print("Data type for Mean_Alpha_Power:", df_eeg["Mean_Alpha_Power"].dtype)



def extract_subject(filename):

    m = re.search(r"(\d+)", filename)
    if m:
        num_str = m.group(1)
        subj_id = str(int(num_str))  # removes any leading zeros
        return subj_id
    else:
        return None


df_eeg['Subject'] = df_eeg['Filename'].apply(extract_subject)
print("\nEEG Data with extracted Subject IDs:")
print(df_eeg[['Filename', 'Subject']].head())


meta_excel_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\EEG_3channels_resting_lanzhou_2015\subjects_information_EEG_3channels_resting_lanzhou_2015.xlsx"
df_meta = pd.read_excel(meta_excel_path, sheet_name=0)  # Adjust sheet_name if necessary
print("\nMetadata (first 5 rows):")
print(df_meta.head())


print("\nMetadata columns:")
print(df_meta.columns)


if "subject id" in df_meta.columns:
    df_meta.rename(columns={"subject id": "Subject"}, inplace=True)
else:
    print("No 'subject id' column found in metadata!")


df_meta["Subject"] = df_meta["Subject"].astype(str)

print("\nMetadata columns after renaming:")
print(df_meta.columns)


df_merged = pd.merge(df_eeg, df_meta, on="Subject", how="inner")
print("\nMerged Data (first 5 rows):")
print(df_merged.head())

output_dir = os.path.dirname(eeg_csv_path)
merged_csv_path = os.path.join(output_dir, "merged_EEG_3channels_metadata.csv")
df_merged.to_csv(merged_csv_path, index=False, sep=';')
print(f"\nMerged data saved to: {merged_csv_path}")


variables_of_interest = [
    "Mean_Alpha_Power", "Mean_Beta_Power", "Mean_Theta_Power", "Mean_Delta_Power",
    "age", "PHQ-9", "CTQ-SF", "LES", "SSRS", "GAD-7", "PSQI"
]
available_vars = [var for var in variables_of_interest if var in df_merged.columns]
print("\nVariables used for correlation analysis:")
print(available_vars)

corr_matrix = df_merged[available_vars].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0)
plt.title("Correlation Heatmap: EEG Features & Clinical Variables")
plt.tight_layout()
heatmap_path = os.path.join(output_dir, "EEG_Metadata_Correlation_Heatmap.png")
plt.savefig(heatmap_path)
print(f"\nCorrelation heatmap saved to: {heatmap_path}")
plt.show()

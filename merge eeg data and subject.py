import pandas as pd


eeg_df = pd.read_excel("EEG_bandpower_features.xlsx")


lanzhou_df = pd.read_excel("Lanzhou University Second Hospital MODMA participants scales.xlsx")


print("EEG Columns:", eeg_df.columns)
print("Lanzhou Columns:", lanzhou_df.columns)


columns_to_extract = [lanzhou_df.columns[0]] + list(lanzhou_df.columns[2:16])
lanzhou_subset = lanzhou_df[columns_to_extract]


merged_df = pd.merge(eeg_df, lanzhou_subset, on="subject", how="left")


merged_df.to_excel("merged_eeg.xlsx", index=False)

print("Merge completed. Output saved as 'merged_eeg.xlsx'.")

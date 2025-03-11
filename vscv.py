import os
import pandas as pd

# Define your file paths (update these as needed)
files = {
    "128 EEG": r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\128channels_resting\output\final_merged_EEG_128channels_analysis.xlsx",
    "Dot Probe Output": r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\audio\dot_probe_audio\dot probe output.xlsx",
    "Resting Audio": r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\audio\resting_audio\output\resting_state_audio.xlsx",
    "Behavioral": r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\Behavioral_Data\behavioral_data.xlsx",
    "Dot Probe": r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\EEG\dot probe.xlsx",
    "Resting State": r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\EEG\resting state.xlsx",
    "3-Channels Resting": r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\EEG_3channels_resting_lanzhou_2015\3_channels_resting.xlsx"
}


for label, path in files.items():
    try:

        if path.lower().endswith('.xlsx'):
            df = pd.read_excel(path)
        elif path.lower().endswith('.csv'):
            df = pd.read_csv(path)
        else:
            print(f"{label}: Unsupported file format.")
            continue


        original_cols = df.columns.tolist()
        lower_cols = [col.lower() for col in original_cols]

        print(f"\n{label} columns (original):")
        print(original_cols)
        print(f"{label} columns (lowercase):")
        print(lower_cols)


        if "subject" not in lower_cols:
            print(f"WARNING: {label} does NOT have a column called 'subject' (case-insensitive).")
        else:
            print(f"{label} has a 'subject' column.")
    except Exception as e:
        print(f"Error loading {label}: {e}")

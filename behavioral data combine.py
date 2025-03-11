import os
import pandas as pd
import csv

data_dir = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\Behavioral_Data"


all_dfs = []


for filename in os.listdir(data_dir):
    if filename.lower().endswith(".txt"):
        file_path = os.path.join(data_dir, filename)
        print("Reading file:", filename)
        try:

            try:
                df = pd.read_csv(file_path, sep="\t", encoding="utf-16")
            except UnicodeError:

                df = pd.read_csv(file_path, sep="\t", encoding="utf-8")
        except Exception as e:
            print("Error reading", filename, ":", e)
            continue


        df["subject"] = os.path.splitext(filename)[0]
        all_dfs.append(df)


if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print("Combined data preview:")
    print(combined_df.head())

    output_path = os.path.join(data_dir, "combined_data.csv")
    combined_df.to_csv(output_path, index=False, sep=";", quoting=csv.QUOTE_ALL)
    print("Saved combined_data.csv to", output_path)
else:
    print("No files were read successfully.")

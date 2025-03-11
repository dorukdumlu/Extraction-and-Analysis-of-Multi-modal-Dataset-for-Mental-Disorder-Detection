import os
import pandas as pd


data_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\Behavioral_Data"


edat_files = [f for f in os.listdir(data_folder) if f.endswith(".edat2")]



def extract_edat_data(file_path):
    with open(file_path, "r", encoding="latin1") as f:
        lines = f.readlines()

    data = {}
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) > 1:
            key, value = parts[0], parts[1]
            data[key] = value

    return data


all_data = []
for file in edat_files:
    file_path = os.path.join(data_folder, file)
    data = extract_edat_data(file_path)
    all_data.append(data)

df = pd.DataFrame(all_data)


df.to_csv("Dot_Probe_Task_Results.csv", index=False)

print("Conversion complete. Data saved as 'Dot_Probe_Task_Results.csv'")

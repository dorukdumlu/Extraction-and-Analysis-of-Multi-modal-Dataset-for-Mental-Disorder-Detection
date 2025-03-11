import numpy as np
import glob
import os
import pandas as pd
from scipy.signal import welch


folder_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\EEG_3channels_resting_lanzhou_2015"

output_csv = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\EEG_3channels_resting_lanzhou_2015\merged_EEG_3channels_metadata.csv"

fs = 1000


file_list = glob.glob(os.path.join(folder_path, "*.txt"))
print(f"Found {len(file_list)} files.")


results = []


for file_path in file_list:
    print("Processing file:", file_path)


    data = np.loadtxt(file_path)
    print(f"  Loaded data shape: {data.shape}")



    ch1_mean = np.mean(data[:, 0])
    ch1_std = np.std(data[:, 0])
    ch1_min = np.min(data[:, 0])
    ch1_max = np.max(data[:, 0])


    ch2_mean = np.mean(data[:, 1])
    ch2_std = np.std(data[:, 1])
    ch2_min = np.min(data[:, 1])
    ch2_max = np.max(data[:, 1])


    ch3_mean = np.mean(data[:, 2])
    ch3_std = np.std(data[:, 2])
    ch3_min = np.min(data[:, 2])
    ch3_max = np.max(data[:, 2])


    f1, psd1 = welch(data[:, 0], fs=fs, nperseg=1024)
    alpha_power1 = np.mean(psd1[(f1 >= 8) & (f1 <= 12)])


    f2, psd2 = welch(data[:, 1], fs=fs, nperseg=1024)
    alpha_power2 = np.mean(psd2[(f2 >= 8) & (f2 <= 12)])


    f3, psd3 = welch(data[:, 2], fs=fs, nperseg=1024)
    alpha_power3 = np.mean(psd3[(f3 >= 8) & (f3 <= 12)])


    file_name = os.path.basename(file_path)


    results.append([
        file_name,
        ch1_mean, ch1_std, ch1_min, ch1_max, alpha_power1,
        ch2_mean, ch2_std, ch2_min, ch2_max, alpha_power2,
        ch3_mean, ch3_std, ch3_min, ch3_max, alpha_power3
    ])


columns = [
    "File",
    "Ch1_Mean", "Ch1_Std", "Ch1_Min", "Ch1_Max", "Ch1_Alpha_Power",
    "Ch2_Mean", "Ch2_Std", "Ch2_Min", "Ch2_Max", "Ch2_Alpha_Power",
    "Ch3_Mean", "Ch3_Std", "Ch3_Min", "Ch3_Max", "Ch3_Alpha_Power"
]

df_results = pd.DataFrame(results, columns=columns)
df_results.to_csv(output_csv, index=False, sep=';')


print("Analysis complete. Results saved to:", output_csv)

import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import welch


folder_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_128channels_resting_lanzhou_2015"
output_excel = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_128channels_resting_lanzhou_2015\Yeni Microsoft Excel Çalışma Sayfası.xlsx"


mat_files = glob.glob(os.path.join(folder_path, "*.mat"))
print(f"Found {len(mat_files)} .mat files in {folder_path}.")

results = []

for file_path in mat_files:
    print("\nProcessing file:", file_path)


    mat_data = sio.loadmat(file_path)


    if 'samplingRate' in mat_data:
        fs = float(mat_data['samplingRate'][0, 0])
    else:

        fs = 1000.0
    print("Sampling Rate (Hz):", fs)

    non_default_keys = [key for key in mat_data.keys() if not key.startswith('__')]
    for extra in ['samplingRate', 'Impedances_0']:
        if extra in non_default_keys:
            non_default_keys.remove(extra)

    if not non_default_keys:
        raise ValueError(f"No valid EEG data key found in file: {file_path}")

    eeg_key = non_default_keys[0]  # Use the first remaining key
    eeg_data = mat_data[eeg_key]
    num_channels, num_timepoints = eeg_data.shape
    print("EEG Data shape:", eeg_data.shape)


    file_results = {"File": os.path.basename(file_path)}


    for ch in range(num_channels):
        channel_signal = eeg_data[ch, :]


        ch_mean = np.mean(channel_signal)
        ch_std = np.std(channel_signal)
        ch_min = np.min(channel_signal)
        ch_max = np.max(channel_signal)


        f, psd = welch(channel_signal, fs=fs, nperseg=1024)


        alpha_power = np.mean(psd[(f >= 8) & (f <= 12)])


        file_results[f"Ch{ch+1}_Mean"] = ch_mean
        file_results[f"Ch{ch+1}_Std"] = ch_std
        file_results[f"Ch{ch+1}_Min"] = ch_min
        file_results[f"Ch{ch+1}_Max"] = ch_max
        file_results[f"Ch{ch+1}_Alpha_Power"] = alpha_power

    results.append(file_results)


df_results = pd.DataFrame(results)

df_results.to_excel(output_excel, index=False, float_format="%.6f")
print("\nAnalysis complete. Results saved to:", output_excel)

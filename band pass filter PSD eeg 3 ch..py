import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch



def load_eeg_file(filepath):
    data = np.loadtxt(filepath)
    return data



def bandpass_filter(data, fs, lowcut=1, highcut=40, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data



def compute_psd(data, fs):
    nperseg = int(2 * fs)
    f, psd = welch(data, fs=fs, nperseg=nperseg, axis=0)
    return f, psd


def band_power(f, psd, band):
    # band: tuple (low, high) in Hz
    idx = np.logical_and(f >= band[0], f <= band[1])

    mean_power = np.mean(psd[idx, :])
    return mean_power



def main():

    folder_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_3channels_resting_lanzhou_2015"
    # Output CSV file path for extracted features
    output_csv = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_3channels_resting_lanzhou_2015\EEG_3channels_PSD_features.csv"

    fs = 250


    eeg_files = glob.glob(os.path.join(folder_path, "*.txt"))
    print("Found", len(eeg_files), "EEG files.")

    features_list = []


    delta_band = (1, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)

    for file in eeg_files:
        print("Processing:", file)
        # Load raw data (shape: n_samples x n_channels)
        data = load_eeg_file(file)
        print("Raw data shape:", data.shape)



        filtered_data = bandpass_filter(data, fs=fs, lowcut=1, highcut=40, order=4)


        f, psd = compute_psd(filtered_data, fs=fs)


        mean_delta = band_power(f, psd, delta_band)
        mean_theta = band_power(f, psd, theta_band)
        mean_alpha = band_power(f, psd, alpha_band)
        mean_beta = band_power(f, psd, beta_band)


        features = {
            "Filename": os.path.basename(file),
            "Mean_Delta_Power": mean_delta,
            "Mean_Theta_Power": mean_theta,
            "Mean_Alpha_Power": mean_alpha,
            "Mean_Beta_Power": mean_beta
        }
        features_list.append(features)


    df_features = pd.DataFrame(features_list)
    df_features.to_csv(output_csv, index=False, sep=';')
    print("Saved extracted features to:", output_csv)


if __name__ == "__main__":
    main()

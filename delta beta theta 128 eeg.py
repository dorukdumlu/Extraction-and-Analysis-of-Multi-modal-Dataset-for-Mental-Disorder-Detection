import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch

def compute_band_power(psd, freqs, band):

    low_f, high_f = band
    idx = (freqs >= low_f) & (freqs <= high_f)

    band_power = np.trapz(psd[idx], freqs[idx])
    return band_power

def main():

    folder_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\128channels_resting"

    output_csv = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\128channels_resting\output\beta theta delta.csv"


    delta_band = (0.5, 4)
    theta_band = (4, 8)
    beta_band  = (13, 30)

    file_list = glob.glob(os.path.join(folder_path, "*.mat"))
    results = []

    for file_path in file_list:
        print(f"Processing file: {file_path}")
        try:
            mat_data = loadmat(file_path)

            possible_keys = []
            for k in mat_data.keys():
                if k.startswith('a0') or 'rest' in k.lower():
                    possible_keys.append(k)

            if not possible_keys:
                print(f"  Could not find EEG data in {file_path}. Available keys: {list(mat_data.keys())}")
                continue

            data_key = possible_keys[0]
            raw_data = mat_data[data_key]


            if raw_data.ndim != 2:
                print(f"  Data shape not 2D in {file_path}, shape={raw_data.shape}")
                continue


            if raw_data.shape[0] < raw_data.shape[1]:

                raw_data = raw_data.T


            if "samplingRate" in mat_data:
                fs = float(mat_data["samplingRate"][0, 0])  # typically a 2D array
            else:
                fs = 250.0

            n_samples, n_channels = raw_data.shape
            row_dict = {"Filename": os.path.basename(file_path)}


            for ch in range(n_channels):
                signal_ch = raw_data[:, ch]
                freqs, psd = welch(signal_ch, fs=fs, nperseg=1024)

                delta_val = compute_band_power(psd, freqs, delta_band)
                theta_val = compute_band_power(psd, freqs, theta_band)
                beta_val  = compute_band_power(psd, freqs, beta_band)

                row_dict[f"Ch{ch+1}_Delta_Power"] = delta_val
                row_dict[f"Ch{ch+1}_Theta_Power"] = theta_val
                row_dict[f"Ch{ch+1}_Beta_Power"]  = beta_val

            results.append(row_dict)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


    df = pd.DataFrame(results)


    df.to_csv(output_csv, sep=';', decimal='.', float_format='%.6f', index=False)
    print(f"Saved bandpowers to {output_csv}")

if __name__ == "__main__":
    main()

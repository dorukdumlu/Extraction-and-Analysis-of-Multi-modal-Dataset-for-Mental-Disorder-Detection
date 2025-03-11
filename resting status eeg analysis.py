import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.io import wavfile
import csv
from mne.time_frequency import psd_array_welch



input_folder_resting = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\wav resting"
output_folder_resting = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\resting_status_eeg_analysis"

if not os.path.exists(output_folder_resting):
    os.makedirs(output_folder_resting)


csv_output_file = os.path.join(output_folder_resting, "resting_state_features.csv")
csv_header = ["Filename", "Mean_Alpha_Power", "Mean_Beta_Power", "Mean_Theta_Power", "Mean_Delta_Power"]

with open(csv_output_file, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_header)


wav_files = glob.glob(os.path.join(input_folder_resting, "*.wav"))
print(f"Found {len(wav_files)} resting state WAV files.")

for file in wav_files:
    print(f"\nProcessing file: {file}")

    try:
        fs, data = wavfile.read(file)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue

    print("Sampling rate:", fs)
    print("Original data shape:", data.shape)


    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_val
        print("Data converted to float and normalized.")

    # Reshape data so that we have (n_channels, n_samples)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim == 2:

        if data.shape[0] < data.shape[1]:
            data = data.T
    print("Data shape for processing (n_channels, n_samples):", data.shape)

    # Optionally, print basic statistics to ensure data variability
    print(f"Data stats -- min: {np.min(data):.4f}, max: {np.max(data):.4f}, std: {np.std(data):.4f}")


    n_channels = data.shape[0]
    ch_names = [f"EEG{i + 1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")


    raw = mne.io.RawArray(data, info)
    print(f"Created RawArray with {n_channels} channel(s).")


    n_fft_value = 16384
    data_array = raw.get_data()  # shape: (n_channels, n_samples)
    try:
        psds, freqs_psd = psd_array_welch(data_array, sfreq=fs, fmin=1, fmax=40, n_fft=n_fft_value)
    except Exception as e:
        print(f"Error computing PSD for {file}: {e}")
        continue

    print("Computed PSD; PSD shape:", psds.shape)


    mean_psd = np.mean(psds, axis=0)


    plt.figure()
    plt.semilogy(freqs_psd, mean_psd)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB)")
    plt.title(f"Average PSD - {os.path.basename(file)}")
    psd_plot_file = os.path.join(output_folder_resting, f"{os.path.basename(file)}_PSD.png")
    plt.savefig(psd_plot_file)
    plt.close()



    def band_power(psd, freqs, low, high):
        mask = (freqs >= low) & (freqs < high)
        return np.trapz(psd[:, mask], freqs[mask], axis=1)



    delta_power = band_power(psds, freqs_psd, 1, 4)  # Delta: 1-4 Hz
    theta_power = band_power(psds, freqs_psd, 4, 8)  # Theta: 4-8 Hz
    alpha_power = band_power(psds, freqs_psd, 8, 12)  # Alpha: 8-12 Hz
    beta_power = band_power(psds, freqs_psd, 12, 30)  # Beta: 12-30 Hz


    mean_delta = np.mean(delta_power)
    mean_theta = np.mean(theta_power)
    mean_alpha = np.mean(alpha_power)
    mean_beta = np.mean(beta_power)

    print(f"File: {os.path.basename(file)}")
    print(f"  Mean Delta Power: {mean_delta:.4f}")
    print(f"  Mean Theta Power: {mean_theta:.4f}")
    print(f"  Mean Alpha Power: {mean_alpha:.4f}")
    print(f"  Mean Beta Power:  {mean_beta:.4f}")


    with open(csv_output_file, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([os.path.basename(file), mean_alpha, mean_beta, mean_theta, mean_delta])

print("Resting state processing complete.")
print("Spectral features saved to:", csv_output_file)

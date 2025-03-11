import os
import glob
import numpy as np
import matplotlib.pyplot as plt
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
    nperseg = int(2 * fs)  # Use 2-second segments
    f, psd = welch(data, fs=fs, nperseg=nperseg, axis=0)
    return f, psd


def main():
    folder_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_3channels_resting_lanzhou_2015"
    output_plots_dir = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_3channels_resting_lanzhou_2015\PSD_Plots"
    os.makedirs(output_plots_dir, exist_ok=True)
    fs = 250  # Sampling rate in Hz

    eeg_files = glob.glob(os.path.join(folder_path, "*.txt"))
    print("Found", len(eeg_files), "EEG files.")

    for file in eeg_files:
        print("Processing:", file)
        data = load_eeg_file(file)
        print("Raw data shape:", data.shape)


        filtered_data = bandpass_filter(data, fs=fs, lowcut=1, highcut=40, order=4)
        f, psd = compute_psd(filtered_data, fs=fs)


        plt.figure(figsize=(10, 6))
        for ch in range(filtered_data.shape[1]):
            plt.semilogy(f, psd[:, ch], label=f"Channel {ch + 1}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (units²/Hz)")
        plt.title("PSD for " + os.path.basename(file))
        plt.legend()
        plt.tight_layout()


        plot_filename = os.path.join(output_plots_dir, os.path.basename(file) + "_PSD.png")
        plt.savefig(plot_filename)
        print(f"Plot saved to: {plot_filename}")
        plt.show()


if __name__ == "__main__":
    main()

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import glob



input_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\wav resting"


output_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\preprocessed_wav_resting"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


lowcut = 1.0  # lower cutoff frequency
highcut = 40.0  # upper cutoff frequency
filter_order = 4  # filter order


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)

    filtered = np.array([filtfilt(b, a, channel) for channel in data])
    return filtered



wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
print(f"Found {len(wav_files)} WAV files.")

for eeg_wav_file in wav_files:
    print("\nProcessing file:", eeg_wav_file)

    try:
        fs, data = wavfile.read(eeg_wav_file)
    except Exception as e:
        print(f"Error reading {eeg_wav_file}: {e}")
        continue

    print("Sampling Rate:", fs)
    print("Original data shape:", data.shape)


    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_val
        print("Data converted to float and normalized.")


    if data.ndim == 1:
        # Mono: reshape to (1, n_samples)
        data = data[np.newaxis, :]
    elif data.ndim == 2:

        data = data.T
    else:
        print("Unexpected data dimensions:", data.shape)
        continue

    print("Data shape for processing (n_channels, n_samples):", data.shape)


    filtered_data = apply_filter(data, lowcut, highcut, fs, order=filter_order)
    print(f"Applied bandpass filter from {lowcut} Hz to {highcut} Hz.")


    avg_reference = np.mean(filtered_data, axis=0)
    filtered_data = filtered_data - avg_reference
    print("Re-referenced data to average reference.")


    plt.figure(figsize=(10, 4))
    plt.plot(filtered_data[0, :])
    plt.title("Preprocessed EEG - Channel 1")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


    preprocessed_data = filtered_data.T


    preprocessed_data_int16 = np.int16(preprocessed_data / np.max(np.abs(preprocessed_data)) * 32767)


    base_name = os.path.basename(eeg_wav_file)
    output_file = os.path.join(output_folder, f"preprocessed_{base_name}")

    try:
        wavfile.write(output_file, fs, preprocessed_data_int16)
        print("Preprocessed EEG saved to:", output_file)
    except Exception as e:
        print(f"Error saving {output_file}: {e}")

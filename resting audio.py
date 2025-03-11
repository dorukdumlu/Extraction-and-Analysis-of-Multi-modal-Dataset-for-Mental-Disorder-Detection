import os
import glob
import numpy as np
import pandas as pd
import mne
from scipy.io import wavfile
from mne.time_frequency import psd_array_welch


input_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\audio\resting_audio\preprocessed_wav_resting\wav resting"

output_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\audio\resting_audio\output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


default_epoch_duration = 2.0  # desired epoch duration
overlap = 0.0  # no overlap


tfr_freqs = np.arange(1, 40, 1)  # Frequencies from 1 to 39 Hz
tfr_n_cycles = tfr_freqs / 2.0  # Number of cycles per frequency


fmin, fmax = 1, 40  # Frequency range for PSD
n_fft_value = 16384  # FFT length


wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
print(f"Found {len(wav_files)} WAV files for resting state analysis.")

for wav_file in wav_files:
    print(f"\nProcessing file: {wav_file}")


    try:
        fs, data = wavfile.read(wav_file)
    except Exception as e:
        print(f"Error reading {wav_file}: {e}")
        continue

    print("Sampling rate:", fs)
    print("Original data shape:", data.shape)


    print(f"Raw data stats before normalization: min={np.min(data)}, max={np.max(data)}, std={np.std(data)}")


    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_val
        print("Data converted to float and normalized.")

    print(f"Raw data stats after normalization: min={np.min(data):.4f}, max={np.max(data):.4f}, std={np.std(data):.4f}")


    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim == 2:
        if data.shape[0] < data.shape[1]:
            data = data.T
    print("Data shape for processing (n_channels, n_samples):", data.shape)


    n_channels = data.shape[0]
    ch_names = [f"EEG{i + 1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    print(f"Created RawArray with {n_channels} channel(s).")

    total_time = raw.times[-1]
    print(f"Total recording time: {total_time:.3f} seconds")


    if total_time < default_epoch_duration:
        epoch_duration = total_time * 0.8
        print(f"Recording shorter than {default_epoch_duration} s. Setting epoch_duration to {epoch_duration:.3f} s.")
    else:
        epoch_duration = default_epoch_duration


    try:
        epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, overlap=overlap, preload=True,
                                              verbose=False)
    except Exception as e:
        print(f"Error creating fixed-length epochs: {e}")
        continue
    print(f"Number of fixed-length epochs created: {len(epochs)}")


    psds, freqs_psd = psd_array_welch(raw.get_data(), sfreq=fs, fmin=fmin, fmax=fmax, n_fft=n_fft_value)
    mean_psd = np.mean(psds, axis=0)


    idx_alpha = np.where((freqs_psd >= 8) & (freqs_psd <= 12))[0]
    mean_alpha = np.mean(mean_psd[idx_alpha])
    print(f"Mean alpha power (8-12 Hz): {mean_alpha:.6f}")

    psd_df = pd.DataFrame({
        "Frequency (Hz)": freqs_psd,
        "PSD": mean_psd
    })

    base_name = os.path.basename(wav_file).split('.')[0]
    psd_csv = os.path.join(output_folder, f"{base_name}_PSD.csv")
    psd_df.to_csv(psd_csv, index=False, sep=";")
    print(f"PSD data saved to: {psd_csv}")


    resampled_epochs = epochs.copy().resample(250, npad="auto", verbose=False)
    print(f"Epochs resampled to {resampled_epochs.info['sfreq']} Hz for TFR analysis.")

    try:
        tfr = resampled_epochs.compute_tfr(method="morlet", freqs=tfr_freqs,
                                           n_cycles=tfr_n_cycles, decim=3, return_itc=False, verbose=False)
    except Exception as e:
        print(f"Error computing TFR: {e}")
        continue
    power_avg = tfr.average()
    tfr_data = power_avg.data[0]  # For channel 0
    tfr_times = power_avg.times
    tfr_freqs = power_avg.freqs


    idx_freq_alpha = np.where((tfr_freqs >= 8) & (tfr_freqs <= 12))[0]
    mean_tfr_alpha = np.mean(tfr_data[idx_freq_alpha, :])
    print(f"Mean TFR power in alpha band (8-12 Hz) across the epoch: {mean_tfr_alpha:.6f}")

    tfr_df = pd.DataFrame(tfr_data, index=tfr_freqs, columns=tfr_times)
    tfr_csv = os.path.join(output_folder, f"{base_name}_TFR.csv")
    tfr_df.to_csv(tfr_csv, sep=";")
    print(f"TFR data saved to: {tfr_csv}")

print("\nProcessing complete. All resting state feature data have been saved in the designated folder.")

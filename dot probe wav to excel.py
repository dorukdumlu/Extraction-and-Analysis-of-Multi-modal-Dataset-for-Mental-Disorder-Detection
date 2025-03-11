import os
import glob
import numpy as np
import pandas as pd
import mne
from scipy.io import wavfile
from mne.time_frequency import psd_array_welch

# ---------------------------
# User Settings for Dot Probe WAV Files
# ---------------------------
input_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\dot_probe_audio\preprocessed_wav_dot_probe\wav dot probe"
output_excel_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\dot_probe_audio\dot probe output.xlsx"

# Epoching parameters (in seconds)
tmin, tmax = -0.2, 0.8
# Number of simulated events per file (for demonstration)
n_events_sim = 20

event_id = {"ConditionA": 1, "ConditionB": 2}


erp_tmin, erp_tmax = 0.1, 0.3


fmin, fmax = 1, 40  # frequency range for PSD
n_fft_value = 16384


wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
print(f"Found {len(wav_files)} WAV files.")


features_list = []

for file in wav_files:
    print(f"\nProcessing file: {file}")

    # Read the WAV file
    try:
        fs, data = wavfile.read(file)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue

    print("Sampling rate:", fs)
    print("Original data shape:", data.shape)

    # Convert to float and normalize if needed
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_val
        print("Data converted to float and normalized.")


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
    event_onsets = np.linspace(1, total_time - 1, n_events_sim)
    event_samples = (event_onsets * fs).astype(int)
    event_codes = np.array([1 if i % 2 == 0 else 2 for i in range(n_events_sim)])
    events = np.column_stack((event_samples, np.zeros(n_events_sim, dtype=int), event_codes))
    print("Simulated events shape:", events.shape)

    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=(None, 0), preload=True, verbose=False)
    n_epochs = len(epochs)
    if n_epochs == 0:
        print("No epochs extracted. Skipping file.")
        continue
    print("Number of epochs extracted:", n_epochs)


    evoked_A = epochs["ConditionA"].average()
    evoked_B = epochs["ConditionB"].average()


    times = evoked_A.times
    idx_window = np.where((times >= erp_tmin) & (times <= erp_tmax))[0]
    erp_A_mean = np.mean(evoked_A.data[0, idx_window])
    erp_B_mean = np.mean(evoked_B.data[0, idx_window])


    data_array = raw.get_data()
    psds, freqs = psd_array_welch(data_array, sfreq=fs, fmin=fmin, fmax=fmax, n_fft=n_fft_value)
    mean_psd = np.mean(psds, axis=0)  # mean across channels
    overall_psd_mean = np.mean(mean_psd)  # overall mean power


    idx_alpha = np.where((freqs >= 8) & (freqs <= 12))[0]
    alpha_power = np.mean(mean_psd[idx_alpha])

    file_features = {
        "File": os.path.basename(file),
        "ERP_CondA_Mean": erp_A_mean,
        "ERP_CondB_Mean": erp_B_mean,
        "Overall_PSD_Mean": overall_psd_mean,
        "Alpha_Power": alpha_power,
        "Sampling_Rate": fs,
        "N_Channels": n_channels,
        "N_Epochs": n_epochs
    }
    features_list.append(file_features)


features_df = pd.DataFrame(features_list)
features_df.to_excel(output_excel_path, index=False)
print(f"Extracted features saved to: {output_excel_path}")

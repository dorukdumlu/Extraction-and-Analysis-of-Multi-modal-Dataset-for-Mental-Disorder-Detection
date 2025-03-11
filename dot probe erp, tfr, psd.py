import os
import glob
import numpy as np
import pandas as pd
import mne
from scipy.io import wavfile
from mne.time_frequency import psd_array_welch


input_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\audio\dot_probe_audio\preprocessed_wav_dot_probe\wav dot probe"

output_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\done\audio\dot_probe_audio\output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


tmin, tmax = -0.2, 0.8


n_events_sim = 20


event_id = {"ConditionA": 1, "ConditionB": 2}


freqs = np.arange(1, 40, 1)  # Frequencies 1 to 39 Hz
n_cycles = freqs / 2.0


fmin, fmax = 1, 40  # Frequency range for PSD
n_fft_value = 16384


wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
print(f"Found {len(wav_files)} WAV files.")

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


    print(f"Raw data stats after normalization: min={np.min(data)}, max={np.max(data)}, std={np.std(data)}")


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


    if total_time <= 2:
        print("Warning: Total recording time is too short for event simulation. Skipping file.")
        continue
    event_onsets = np.linspace(1, total_time - 1, n_events_sim)
    event_samples = (event_onsets * fs).astype(int)
    event_codes = np.array([1 if i % 2 == 0 else 2 for i in range(n_events_sim)])
    events = np.column_stack((event_samples, np.zeros(n_events_sim, dtype=int), event_codes))
    print("Simulated event onsets (s):", event_onsets)
    print("Simulated events shape:", events.shape)


    try:
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                            baseline=(None, 0), preload=True, verbose=False)
    except Exception as e:
        print(f"Error during epoching: {e}")
        continue

    if len(epochs) == 0:
        print("No epochs extracted. Skipping file.")
        continue
    print("Number of epochs extracted:", len(epochs))


    evoked_A = epochs["ConditionA"].average()
    evoked_B = epochs["ConditionB"].average()


    times = evoked_A.times
    idx_window = np.where((times >= 0.1) & (times <= 0.3))[0]
    mean_erp_A = np.mean(evoked_A.data[0, idx_window])
    std_erp_A = np.std(evoked_A.data[0, idx_window])
    print(f"ERP ConditionA (0.1-0.3 s): mean={mean_erp_A:.6f}, std={std_erp_A:.6f}")

    erp_df = pd.DataFrame({
        "Time (s)": evoked_A.times,
        "ERP_ConditionA": evoked_A.data[0],
        "ERP_ConditionB": evoked_B.data[0]
    })

    base_name = os.path.basename(wav_file).split('.')[0]
    erp_csv = os.path.join(output_folder, f"{base_name}_ERP.csv")
    erp_df.to_csv(erp_csv, index=False, sep=";")
    print(f"ERP data saved to: {erp_csv}")


    psds, freqs_psd = psd_array_welch(raw.get_data(), sfreq=fs, fmin=fmin, fmax=fmax, n_fft=n_fft_value)
    mean_psd = np.mean(psds, axis=0)


    idx_alpha = np.where((freqs_psd >= 8) & (freqs_psd <= 12))[0]
    mean_alpha = np.mean(mean_psd[idx_alpha])
    print(f"Mean alpha power (8-12 Hz): {mean_alpha:.6f}")

    psd_df = pd.DataFrame({
        "Frequency (Hz)": freqs_psd,
        "PSD": mean_psd
    })

    psd_csv = os.path.join(output_folder, f"{base_name}_PSD.csv")
    psd_df.to_csv(psd_csv, index=False, sep=";")
    print(f"PSD data saved to: {psd_csv}")


    power = epochs["ConditionA"].compute_tfr(method="morlet", freqs=freqs,
                                             n_cycles=n_cycles, decim=3, return_itc=False)
    power_avg = power.average()
    tfr_data = power_avg.data[0]  # For channel 0
    tfr_times = power_avg.times
    tfr_freqs = power_avg.freqs


    idx_freq_alpha = np.where((tfr_freqs >= 8) & (tfr_freqs <= 12))[0]
    idx_time_window = np.where((tfr_times >= 0.1) & (tfr_times <= 0.3))[0]
    mean_tfr_alpha = np.mean(tfr_data[np.ix_(idx_freq_alpha, idx_time_window)])
    print(f"Mean TFR power in alpha band (8-12 Hz, 0.1-0.3 s): {mean_tfr_alpha:.6f}")

    tfr_df = pd.DataFrame(tfr_data, index=tfr_freqs, columns=tfr_times)
    tfr_csv = os.path.join(output_folder, f"{base_name}_TFR.csv")
    tfr_df.to_csv(tfr_csv, sep=";")
    print(f"TFR data saved to: {tfr_csv}")

print("\nProcessing complete. All feature data have been saved in the designated folder.")

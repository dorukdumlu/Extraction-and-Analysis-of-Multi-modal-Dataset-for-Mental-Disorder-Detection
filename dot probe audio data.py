import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.io import wavfile
from mne.time_frequency import psd_array_welch

input_folder_dot_probe = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\wav dot probe"
output_folder_dot_probe = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\dot_probe_eeg_analysis"
if not os.path.exists(output_folder_dot_probe):
    os.makedirs(output_folder_dot_probe)


tmin, tmax = -0.2, 0.8

n_events_sim = 20


event_id = {"ConditionA": 1, "ConditionB": 2}


wav_files = glob.glob(os.path.join(input_folder_dot_probe, "*.wav"))
print(f"Found {len(wav_files)} dot probe WAV files.")

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


    print(f"Data stats -- min: {np.min(data):.4f}, max: {np.max(data):.4f}, std: {np.std(data):.4f}")


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
                        baseline=(None, 0), preload=True)
    print("Number of epochs extracted:", len(epochs))

    if len(epochs) == 0:
        print("No epochs extracted. Skipping file.")
        continue

    evoked_A = epochs["ConditionA"].average()
    evoked_B = epochs["ConditionB"].average()


    fig_A = evoked_A.plot(picks=[0], show=False, time_unit='s')
    fig_A.suptitle("ERP - ConditionA")
    out_fig_A = os.path.join(output_folder_dot_probe, f"{os.path.basename(file)}_ERP_CondA.png")
    fig_A.savefig(out_fig_A)
    plt.close(fig_A)


    fig_B = evoked_B.plot(picks=[0], show=False, time_unit='s')
    fig_B.suptitle("ERP - ConditionB")
    out_fig_B = os.path.join(output_folder_dot_probe, f"{os.path.basename(file)}_ERP_CondB.png")
    fig_B.savefig(out_fig_B)
    plt.close(fig_B)


    evoked_A.save(os.path.join(output_folder_dot_probe, f"{os.path.basename(file)}_evoked_CondA-ave.fif"),
                  overwrite=True)
    evoked_B.save(os.path.join(output_folder_dot_probe, f"{os.path.basename(file)}_evoked_CondB-ave.fif"),
                  overwrite=True)

    frequencies = np.arange(1, 40, 1)  # Frequencies from 1 to 40 Hz
    n_cycles = frequencies / 2.0  # Number of cycles per frequency

    power = epochs["ConditionA"].compute_tfr(method="morlet", freqs=frequencies,
                                             n_cycles=n_cycles, decim=3, return_itc=False)

    power_avg = power.average()


    fig_tfr = power_avg.plot(picks=[0], baseline=(None, 0), mode="ratio", show=False)

    if isinstance(fig_tfr, list):
        fig_to_save = fig_tfr[0]
    else:
        fig_to_save = fig_tfr
    out_tfr = os.path.join(output_folder_dot_probe, f"{os.path.basename(file)}_TFR_CondA.png")
    fig_to_save.savefig(out_tfr)
    plt.close(fig_to_save)


    n_fft_value = 16384
    data_array = raw.get_data()
    psds, freqs_psd = psd_array_welch(data_array, sfreq=fs, fmin=1, fmax=40, n_fft=n_fft_value)
    mean_psd = np.mean(psds, axis=0)
    plt.figure()
    plt.semilogy(freqs_psd, mean_psd)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB)")
    plt.title("Average PSD")
    out_psd = os.path.join(output_folder_dot_probe, f"{os.path.basename(file)}_PSD.png")
    plt.savefig(out_psd)
    plt.close()

print("Dot Probe processing complete.")

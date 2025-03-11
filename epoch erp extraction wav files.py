import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.io import wavfile
from scipy.stats import ttest_ind
from scipy.signal import resample


eeg_wav_file = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\preprocessed_wav\preprocessed_sub-001_task-dot_probe_eeg.EDF.wav"


fs, data = wavfile.read(eeg_wav_file)
print("Loaded WAV file with sampling rate:", fs)
print("Original data shape from WAV:", data.shape)


if data.ndim == 1:
    data = data[np.newaxis, :]


if data.shape[0] < data.shape[1]:
    raw_data = data
else:
    raw_data = data.T

n_channels, n_samples = raw_data.shape
print("Using raw_data shape (n_channels, n_samples):", raw_data.shape)


ch_names = [f"EEG{i+1}" for i in range(n_channels)]

# Create an MNE Info object.
info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")


raw = mne.io.RawArray(raw_data, info)

raw.plot(duration=5, n_channels=min(10, n_channels), title="Raw Preprocessed EEG Data")


n_events = 20
times = raw.times  # time vector in seconds
event_onsets = np.linspace(1, times[-1]-1, n_events)
event_onsets_samples = (event_onsets * fs).astype(int)

event_codes = np.array([1 if i % 2 == 0 else 2 for i in range(n_events)])

events = np.column_stack((event_onsets_samples, np.zeros(n_events, dtype=int), event_codes))
print("Simulated events shape:", events.shape)


event_id = {"CondA": 1, "CondB": 2}


tmin, tmax = -0.2, 0.8
epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                    baseline=(None, 0), preload=True)
print("Number of epochs extracted:", len(epochs))


evoked_A = epochs["CondA"].average()
evoked_B = epochs["CondB"].average()

# Plot the ERPs for a selected channel (e.g., channel 0).
evoked_A.plot(picks=[0], spatial_colors=True, time_unit='s', title="ERP for Condition A")
evoked_B.plot(picks=[0], spatial_colors=True, time_unit='s', title="ERP for Condition B")


psds, freqs = mne.time_frequency.psd_welch(raw, fmin=1, fmax=40, n_fft=256)
print("PSDs shape:", psds.shape)


mean_psd = np.mean(psds, axis=0)
plt.figure()
plt.semilogy(freqs, mean_psd)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (dB)")
plt.title("Average PSD of Preprocessed EEG")
plt.show()


alpha_mask = np.logical_and(freqs >= 8, freqs <= 12)
alpha_power = np.trapz(psds[:, alpha_mask], freqs[alpha_mask], axis=1)
total_power = np.trapz(psds, freqs, axis=1)
relative_alpha = alpha_power / total_power
print("Relative alpha power (mean across channels):", np.mean(relative_alpha))

frequencies = np.arange(1, 40, 1)  # Frequencies from 1 to 40 Hz
n_cycles = frequencies / 2.0
power = mne.time_frequency.tfr_morlet(epochs["CondA"], freqs=frequencies, n_cycles=n_cycles,
                                      use_fft=True, return_itc=False, decim=3, n_jobs=1)

power.plot_topo(baseline=(None, 0), mode="logratio", title="Time–Frequency Power for Condition A")


time_window = (0.3, 0.5)

time_idx = np.where((evoked_A.times >= time_window[0]) & (evoked_A.times <= time_window[1]))[0]

data_A = epochs["CondA"].get_data(picks=[0])  # shape: (n_epochs_A, 1, n_times)
data_B = epochs["CondB"].get_data(picks=[0])

mean_amp_A = data_A[:, 0, time_idx].mean(axis=1)
mean_amp_B = data_B[:, 0, time_idx].mean(axis=1)

t_stat, p_val = ttest_ind(mean_amp_A, mean_amp_B)
print("T-test comparing ERP amplitude (channel 0, 0.3-0.5 s):")
print("t-statistic =", t_stat, ", p-value =", p_val)

import os
import mne
import matplotlib.pyplot as plt


input_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\raw_data"
qc_output_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\edf qc"
os.makedirs(qc_output_folder, exist_ok=True)

l_freq = 1.0
h_freq = 40.0
desired_sfreq = 250

for filename in os.listdir(input_folder):
    fname_lower = filename.lower()

    if ('evoked' in fname_lower) or ('ave.fif' in fname_lower):
        print(f"Skipping '{filename}' (evoked/averaged file)")
        continue
    if not (fname_lower.endswith(".edf") or fname_lower.endswith(".fif")):
        continue

    file_path = os.path.join(input_folder, filename)
    print(f"Processing: {filename}")

    try:
        raw = mne.io.read_raw(file_path, preload=True)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

    print(f"Sampling rate: {raw.info['sfreq']} Hz")

    if raw.info['sfreq'] != desired_sfreq:
        raw.resample(desired_sfreq)
        print(f"Resampled to {desired_sfreq} Hz.")

    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    print(f"Applied bandpass filter: {l_freq}-{h_freq} Hz")


    duration_sec = 30
    fig1 = raw_filtered.plot(duration=duration_sec, n_channels=10, show=False, title=f"{filename} - Time Series")
    ts_plot_path = os.path.join(qc_output_folder,
                                filename.replace(".edf", "_timeseries.png").replace(".fif", "_timeseries.png"))
    fig1.savefig(ts_plot_path)
    plt.close(fig1)
    print(f"Time series plot saved to: {ts_plot_path}")


    psd = raw_filtered.compute_psd(method='welch', fmax=h_freq)

    fig2 = psd.plot(average=True, show=False)
    fig2.suptitle(f"{filename} - PSD")

    psd_plot_path = os.path.join(qc_output_folder, filename.replace(".edf", "_psd.png").replace(".fif", "_psd.png"))
    fig2.savefig(psd_plot_path)
    plt.close(fig2)
    print(f"PSD plot saved to: {psd_plot_path}")

print("QC processing completed.")

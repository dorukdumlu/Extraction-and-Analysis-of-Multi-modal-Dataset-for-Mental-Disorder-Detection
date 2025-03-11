import os
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

cleaned_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\cleaned_raw_edf"
output_excel = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\EEG_bandpower_features.xlsx"


bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 40)
}

results = []


for filename in os.listdir(cleaned_folder):
    if not filename.lower().endswith("_clean.fif"):
        continue
    file_path = os.path.join(cleaned_folder, filename)
    print(f"Processing {filename} for band power extraction...")

    try:
        raw = mne.io.read_raw_fif(file_path, preload=True)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

    sfreq = raw.info["sfreq"]

    picks = mne.pick_types(raw.info, eeg=True, eog=False, meg=False)
    data, times = raw[picks, :]


    nperseg = int(2 * sfreq)  # 2-second segments
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)


    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_power = np.mean(psd[:, idx_band])
        band_powers[band + "_power"] = band_power


    subject_id = filename.split("_")[0]

    result = {"Subject": subject_id, "Filename": filename}
    result.update(band_powers)
    results.append(result)

# Create a DataFrame
columns = [
    "Subject", "Filename",
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"
]
df_features = pd.DataFrame(results, columns=columns)


with pd.ExcelWriter(output_excel) as writer:
    df_features.to_excel(writer, index=False)
print(f"Band power features saved to: {output_excel}")

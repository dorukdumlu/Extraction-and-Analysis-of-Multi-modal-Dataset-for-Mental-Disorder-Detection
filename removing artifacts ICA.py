import os
import mne
from mne.preprocessing import ICA, create_eog_epochs


input_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\raw_data_fif"
output_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\fif_output"
os.makedirs(output_folder, exist_ok=True)


for filename in os.listdir(input_folder):
    if filename.endswith(".fif"):
        file_path = os.path.join(input_folder, filename)
        print(f"Processing file: {filename}")


        raw = mne.io.read_raw_fif(file_path, preload=True)
        print(f"Loaded {filename} with sampling rate = {raw.info['sfreq']} Hz.")


        ica = ICA(n_components=20, random_state=97, max_iter='auto')
        ica.fit(raw)


        try:
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            print(f"Identified EOG-related components: {eog_indices}")
            ica.exclude = eog_indices
        except Exception as e:
            print("EOG detection failed or not applicable:", e)


        raw_clean = ica.apply(raw.copy())
        print("Applied ICA to clean the data.")


        cleaned_filename = filename.replace(".fif", "_clean.fif")
        cleaned_path = os.path.join(output_folder, cleaned_filename)
        raw_clean.save(cleaned_path, overwrite=True)
        print(f"Cleaned file saved to: {cleaned_path}\n")

import os
import mne


folder_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\raw_data"


for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)


    clean_filename = filename.strip().lower()
    new_path = os.path.join(folder_path, clean_filename)


    if old_path != new_path:
        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' -> '{clean_filename}'")


    if clean_filename.endswith(".edf"):
        try:
            raw = mne.io.read_raw_edf(new_path, preload=True)
            print(f"Successfully loaded '{clean_filename}' with sampling rate = {raw.info['sfreq']} Hz.")
        except Exception as e:
            print(f"Error loading '{clean_filename}': {e}")
    else:

        print(f"Skipping '{clean_filename}' (not an .edf file).")

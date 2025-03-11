import os
import mne
from mne.preprocessing import ICA, create_eog_epochs


input_folder =  r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\raw_data"
cleaned_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\cleaned_raw_edf"
os.makedirs(cleaned_folder, exist_ok=True)


for filename in os.listdir(input_folder):
    fname_lower = filename.lower()

    if not (fname_lower.endswith(".edf") or fname_lower.endswith(".fif")):
        continue
    if "evoked" in fname_lower or "ave.fif" in fname_lower:
        print(f"Skipping '{filename}' (not continuous raw data)")
        continue

    file_path = os.path.join(input_folder, filename)
    print(f"\nProcessing file for ICA: {filename}")

    try:
        raw = mne.io.read_raw(file_path, preload=True)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        continue

    # Print sampling rate and basic info
    print(f"Loaded {filename} with sampling rate = {raw.info['sfreq']} Hz.")

    # Run ICA: Adjust n_components if needed (here we use 20 components)
    ica = ICA(n_components=20, random_state=97, max_iter='auto')
    ica.fit(raw)
    print("ICA fitting completed.")

    # Optionally, find EOG-related components if you have EOG channels
    try:
        eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=3.0)
        if eog_indices:
            print(f"Identified EOG-related components: {eog_indices}")
            ica.exclude = eog_indices
        else:
            print("No significant EOG-related components found.")
    except Exception as e:
        print("EOG detection not applicable or failed:", e)

    # Visual inspection step (optional): Uncomment the following lines if you want to review components manually.
    # ica.plot_components(title=f"ICA Components for {filename}")
    # ica.plot_properties(raw, picks=ica.exclude)

    # Apply ICA to remove the artifacts
    raw_clean = ica.apply(raw.copy())
    print("ICA applied; artifacts removed.")

    # Save the cleaned data
    cleaned_filename = filename.replace(".fif", "_clean.fif").replace(".edf", "_clean.fif")
    cleaned_path = os.path.join(cleaned_folder, cleaned_filename)
    raw_clean.save(cleaned_path, overwrite=True)
    print(f"Cleaned file saved to: {cleaned_path}")

print("Artifact removal with ICA completed for all files.")

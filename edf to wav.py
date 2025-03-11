import os
import mne
from scipy.io.wavfile import write
import numpy as np


edf_directory = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_LZU_2015_2_resting state"
output_directory = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\wav"


os.makedirs(output_directory, exist_ok=True)


edf_files = [os.path.join(root, file)
             for root, _, files in os.walk(edf_directory)
             for file in files if file.lower().endswith(".edf")]  # Convert to lowercase


print(f"🔍 Found {len(edf_files)} EDF files!")


for edf_path in edf_files:
    try:
        print(f"📂 Processing: {edf_path}")


        raw = mne.io.read_raw_edf(edf_path, preload=True)


        audio_data, times = raw[:, :]


        audio_channel = audio_data[0]


        audio_norm = np.int16(audio_channel / np.max(np.abs(audio_channel)) * 32767)


        subject_id = os.path.basename(edf_path).replace(".edf", "")
        wav_filename = os.path.join(output_directory, f"{subject_id}.wav")

        # Save as WAV
        write(wav_filename, 44100, audio_norm)

        print(f"✅ Saved: {wav_filename}")

    except Exception as e:
        print(f"❌ Error processing {edf_path}: {e}")

print("🎉 Conversion complete!")

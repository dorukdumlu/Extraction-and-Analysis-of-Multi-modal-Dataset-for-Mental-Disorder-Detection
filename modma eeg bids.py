import patoolib

zip_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\MODMA_EEG_BIDS_format.z03"
extract_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1"

patoolib.extract_archive(zip_path, outdir=extract_path)
print(f"✅ Extraction Complete! Files are in: {extract_path}")

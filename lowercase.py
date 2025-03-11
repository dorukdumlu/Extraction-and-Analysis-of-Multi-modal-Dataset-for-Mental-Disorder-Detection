import os

folder_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\project\raw_data"

for filename in os.listdir(folder_path):
    lower_filename = filename.lower().strip()

    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, lower_filename)

    if old_path != new_path:
        os.rename(old_path, new_path)
        print(f"Renamed {filename} to {lower_filename}")

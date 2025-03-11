import olefile
import os
import pandas as pd
import re
import xml.etree.ElementTree as ET


data_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\Behavioral_Data\Behavioral_Data"
output_csv = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\Dot_Probe_Task_Results.csv"


edat_files = [f for f in os.listdir(data_folder) if f.endswith(".edat2")]


TARGET_STREAMS = ["stream_recordformat", "stream_variables", "stream_data"]


def clean_text(text):
    text = text.replace("\x00", "").strip()  # Remove null bytes

    text = re.sub(r"[^\x20-\x7E]", "", text)

    text = re.sub(r"ÿþÿ|\x01\x80\x04", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def remove_interleaved_spaces(text):

    if len(text) >= 2 and all(text[i] == " " for i in range(1, min(len(text), 20), 2)):
        return text[::2]
    return text



def extract_key_value_pairs(text):
    pairs = {}

    lines = text.split("\n")
    for line in lines:

        if "\t" in line:
            parts = line.split("\t")
            for i in range(0, len(parts) - 1, 2):
                key = parts[i].strip()
                value = parts[i + 1].strip()
                if key:
                    pairs[key] = value

        elif ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                if key:
                    pairs[key] = value
    return pairs



all_data = []

for file in edat_files:
    file_path = os.path.join(data_folder, file)
    extracted_data = {"Filename": file}

    try:
        if olefile.isOleFile(file_path):
            ole = olefile.OleFileIO(file_path)
            streams = ole.listdir()

            stream_names = ["/".join(stream) for stream in streams]


            for stream_name in stream_names:
                if stream_name in TARGET_STREAMS:
                    try:
                        with ole.openstream(stream_name) as s:
                            raw_content = s.read()


                            for encoding in ["utf-16-le", "utf-16-be", "utf-8", "latin1", "utf-32"]:
                                try:
                                    content = raw_content.decode(encoding)

                                    content = remove_interleaved_spaces(content)
                                    break
                                except UnicodeDecodeError:
                                    continue


                            content = clean_text(content)


                            kv_pairs = extract_key_value_pairs(content)
                            if kv_pairs:
                                extracted_data.update(kv_pairs)
                            else:

                                extracted_data[stream_name] = content
                    except Exception as e:
                        extracted_data[stream_name] = f"Error reading stream: {e}"
            all_data.append(extracted_data)
    except Exception as e:
        print(f"⚠️ Warning: Could not process {file}: {e}")


if all_data:
    df = pd.DataFrame(all_data)

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"✅ Successfully extracted structured data! CSV saved as {output_csv}")
else:
    print("❌ ERROR: No valid data extracted! Check the `.edat2` files.")

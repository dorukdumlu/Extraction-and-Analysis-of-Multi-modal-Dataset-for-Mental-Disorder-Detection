import os
import glob
import mne


input_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\EEG_128channels_ERP_lanzhou_2015"



output_folder = r"C:\Users\Lenovo\OneDrive\Masaüstü\analysis1\ERP_Output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


target_label = 'fcue'


tmin = -0.2  # start 200 ms before event
tmax = 0.8


raw_files = glob.glob(os.path.join(input_folder, "*.raw"))
print(f"Found {len(raw_files)} raw files.")


for raw_file in raw_files:
    print("\n====================================")
    print("Processing file:", raw_file)

    try:

        raw = mne.io.read_raw_egi(raw_file, preload=True)
    except Exception as e:
        print("Error loading file:", raw_file, "\nError:", e)
        continue


    print("Available channels:")
    print(raw.info["ch_names"])


    raw.filter(1., 40., fir_design='firwin')


    events, annot_event_id = mne.events_from_annotations(raw)
    print("Annotation Event ID Mapping:")
    print(annot_event_id)


    if target_label in annot_event_id:
        target_event = {target_label: annot_event_id[target_label]}
        print(f"Using target event: {target_event}")
    else:
        if annot_event_id:
            default_label = list(annot_event_id.keys())[0]
            target_event = {default_label: annot_event_id[default_label]}
            print(f"Target event '{target_label}' not found. Defaulting to event: {target_event}")
        else:
            print("No annotation events found in file. Skipping file:", raw_file)
            continue


    try:
        epochs = mne.Epochs(raw, events, event_id=target_event, tmin=tmin, tmax=tmax,
                            baseline=(None, 0), preload=True)
    except Exception as e:
        print("Error during epoching in file:", raw_file, "\nError:", e)
        continue

    print(f"Number of epochs extracted: {len(epochs)}")


    if len(epochs) == 0:
        print("No epochs found for this event. Skipping file:", raw_file)
        continue


    evoked = epochs.average()


    subject_name = os.path.splitext(os.path.basename(raw_file))[0]
    evoked_fname = os.path.join(output_folder, f"{subject_name}-{target_label}-ave.fif")


    evoked.save(evoked_fname)
    print("ERP saved to:", evoked_fname)

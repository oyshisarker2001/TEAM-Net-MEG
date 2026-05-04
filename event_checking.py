import os
import glob
import numpy as np
import mne

DATASET_ROOT = "ds007640-download"   # change if needed

def find_single_file(folder, pattern):
    files = glob.glob(os.path.join(folder, pattern))
    if len(files) == 0:
        return None
    if len(files) > 1:
        print(f"[WARN] Multiple files found for pattern {pattern} in {folder}. Using first one.")
    return files[0]

def inspect_events_and_triggers(meg_file):
    print(f"\n[LOADING] {meg_file}")
    raw = mne.io.read_raw_fif(meg_file, preload=False, verbose=False)

    print("\n================ CHANNEL INFO ================")
    print(f"Total channels: {len(raw.ch_names)}")

    ch_types = raw.get_channel_types()
    stim_channels = []

    for name, ctype in zip(raw.ch_names, ch_types):
        if ctype == "stim" or "STI" in name.upper() or "TRIG" in name.upper():
            stim_channels.append(name)

    print("Stim/trigger-like channels found:")
    if len(stim_channels) == 0:
        print("  None")
    else:
        for ch in stim_channels:
            print(" ", ch)

    print("\n================ ANNOTATIONS ================")
    if raw.annotations is not None and len(raw.annotations) > 0:
        print(f"Number of annotations: {len(raw.annotations)}")
        print("First 10 annotations:")
        for i in range(min(10, len(raw.annotations))):
            print(
                f"  {i}: onset={raw.annotations.onset[i]:.3f}s, "
                f"duration={raw.annotations.duration[i]:.3f}s, "
                f"description={raw.annotations.description[i]}"
            )

        try:
            events_ann, event_id_ann = mne.events_from_annotations(raw, verbose=False)
            print("\nEvents from annotations:")
            print("  events shape:", events_ann.shape)
            print("  event_id mapping:", event_id_ann)
            print("  first 10 rows:")
            print(events_ann[:10])
        except Exception as e:
            print("[WARN] events_from_annotations failed:", e)
    else:
        print("No annotations found.")

    print("\n================ EVENTS FROM STIM CHANNELS ================")
    if len(stim_channels) == 0:
        print("No stim channels available for mne.find_events().")
    else:
        for stim in stim_channels:
            print(f"\n--- Stim channel: {stim} ---")
            try:
                events_stim = mne.find_events(raw, stim_channel=stim, verbose=False)
                print("events shape:", events_stim.shape)
                if len(events_stim) > 0:
                    unique_ids, counts = np.unique(events_stim[:, 2], return_counts=True)
                    print("Unique event IDs and counts:")
                    for eid, cnt in zip(unique_ids, counts):
                        print(f"  event_id={eid}, count={cnt}")
                    print("First 10 events:")
                    print(events_stim[:10])
                else:
                    print("No events found on this stim channel.")
            except Exception as e:
                print(f"[WARN] mne.find_events failed on {stim}: {e}")

    print("\n================ OPTIONAL RAW PLOT ================")
    print("To visually inspect trigger channels, run this manually:")
    print("raw.plot(duration=20, n_channels=40, scalings='auto')")

# --------------------------------------------------
# Example: inspect first subject/session automatically
# --------------------------------------------------
subject_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "sub-*")))

if len(subject_dirs) == 0:
    raise FileNotFoundError(f"No subject folders found in {DATASET_ROOT}")

first_sub = subject_dirs[0]
session_dirs = sorted(glob.glob(os.path.join(first_sub, "ses-*")))

if len(session_dirs) == 0:
    raise FileNotFoundError(f"No session folders found in {first_sub}")

first_ses = session_dirs[0]
meg_dir = os.path.join(first_ses, "meg")
meg_file = find_single_file(meg_dir, "*_meg.fif")

if meg_file is None:
    raise FileNotFoundError(f"No *_meg.fif file found in {meg_dir}")

inspect_events_and_triggers(meg_file)
import pandas as pd

beh_file = "ds007640-download/sub-01/ses-01/beh/sub-01_ses-01_task-HAHV_beh.tsv"
labels = pd.read_csv(beh_file, sep="\t")
print("Number of label rows:", len(labels))
print(labels.head())


def choose_trial_events(events, expected_n_trials):
    """
    If multiple event IDs exist, choose the one whose count is closest
    to expected_n_trials. If only one event type exists, return all.
    """
    if events is None or len(events) == 0:
        return None

    unique_ids, counts = np.unique(events[:, 2], return_counts=True)

    if len(unique_ids) == 1:
        return events

    best_id = None
    best_diff = float("inf")

    for eid, cnt in zip(unique_ids, counts):
        diff = abs(cnt - expected_n_trials)
        if diff < best_diff:
            best_diff = diff
            best_id = eid

    chosen = events[events[:, 2] == best_id]
    print(f"[INFO] Selected event_id={best_id} with {len(chosen)} events (expected ~{expected_n_trials})")
    return chosen

import numpy as np
import pandas as pd
import mne

meg_file = "ds007640-download/sub-01/ses-01/meg/sub-01_ses-01_task-HAHV_meg.fif"
beh_file = "ds007640-download/sub-01/ses-01/beh/sub-01_ses-01_task-HAHV_beh.tsv"

labels = pd.read_csv(beh_file, sep="\t")
expected_n_trials = len(labels)

raw = mne.io.read_raw_fif(meg_file, preload=False, verbose=False)
events = mne.find_events(raw, stim_channel="Stimulus", verbose=False)

unique_ids, counts = np.unique(events[:, 2], return_counts=True)

print("Expected trials:", expected_n_trials)
print("\nAll event counts:")
for eid, cnt in zip(unique_ids, counts):
    print(f"event_id={eid}, count={cnt}, diff={abs(cnt - expected_n_trials)}")

best_id = None
best_diff = float("inf")

for eid, cnt in zip(unique_ids, counts):
    diff = abs(cnt - expected_n_trials)
    if diff < best_diff:
        best_diff = diff
        best_id = eid

print("\nSelected best_id:", best_id)

trial_events = events[events[:, 2] == best_id]
print("Selected trial events shape:", trial_events.shape)
print("First 10 selected trial events:")
print(trial_events[:10])

print("read from here)" \
"")

import os
import numpy as np
import pandas as pd
import mne

meg_file = "ds007640-download/sub-01/ses-01/meg/sub-01_ses-01_task-HAHV_meg.fif"
beh_file = "ds007640-download/sub-01/ses-01/beh/sub-01_ses-01_task-HAHV_beh.tsv"

raw = mne.io.read_raw_fif(meg_file, preload=False, verbose=False)
events = mne.find_events(raw, stim_channel="Stimulus", verbose=False)
sfreq = raw.info["sfreq"]

labels = pd.read_csv(beh_file, sep="\t")
print("Number of label rows:", len(labels))
print(labels[["trial", "video_id_presented", "SAM_valence", "SAM_arousal"]])

unique_ids, counts = np.unique(events[:, 2], return_counts=True)

candidate_ids = [eid for eid, cnt in zip(unique_ids, counts) if cnt == len(labels)]
print("\nCandidate event IDs with same count as labels:", candidate_ids)

for eid in candidate_ids:
    these = events[events[:, 2] == eid]
    times_sec = these[:, 0] / sfreq
    print(f"\nEvent ID {eid}")
    print("Times (sec):")
    print(np.round(times_sec, 3))
    if len(times_sec) > 1:
        print("Gap between consecutive events (sec):")
        print(np.round(np.diff(times_sec), 3))


import numpy as np
import mne

meg_file = "ds007640-download/sub-01/ses-01/meg/sub-01_ses-01_task-HAHV_meg.fif"

raw = mne.io.read_raw_fif(meg_file, preload=False, verbose=False)
events = mne.find_events(raw, stim_channel="Stimulus", verbose=False)
sfreq = raw.info["sfreq"]

candidate_ids = [33, 34, 35, 36, 112, 113, 118, 119]
cand_events = events[np.isin(events[:, 2], candidate_ids)]

# Convert sample index to seconds
times_sec = cand_events[:, 0] / sfreq
ids = cand_events[:, 2]

# Sort by time just to be safe
order = np.argsort(times_sec)
times_sec = times_sec[order]
ids = ids[order]

print("All candidate events in chronological order:\n")
for t, eid in zip(times_sec, ids):
    print(f"{t:8.3f} sec   event_id={eid}")
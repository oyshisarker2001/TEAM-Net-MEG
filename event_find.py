
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
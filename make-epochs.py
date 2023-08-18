"""
Top-level trial data pre-processing

Authored by Eric Easthope
"""

import os
import warnings
import numpy as np
from pathlib import Path
from pynwb import NWBHDF5IO
from utils import overlap

# Filter warnings
warnings.filterwarnings("ignore")

DATA = "data"
DERIVED = "derived"

subjects = ["EC2", "EC9", "GP31", "GP33"]

MIN_REST_LENGTH = 0.5
OVERWRITE = True

# Make derived directory if it does not exist
try:
    os.mkdir(DERIVED)
except FileExistsError:
    pass

for SUBJECT in subjects:

    # Make subject directory if it does not exist
    try:
        os.mkdir(DERIVED + f"/{SUBJECT}")
    except FileExistsError:
        pass

    for path in [p for p in Path(DATA).glob(f"{SUBJECT}*.nwb")]:
        _, SESSION = path.stem.split("_")
        OUT = DERIVED + f"/{SUBJECT}/{SESSION}"

        if os.path.isdir(f"{OUT}/trials") and not OVERWRITE:
            print("Trials exist already, skipping ...")
        else:
            # Make session directory if it does not exist
            try:
                os.mkdir(OUT)
            except FileExistsError:
                pass

            # Make trials directory if it does not exist
            try:
                os.mkdir(f"{OUT}/trials")
            except FileExistsError:
                pass

            print(f"File {SUBJECT}_{SESSION}.")
            nwb = NWBHDF5IO(path, "r", load_namespaces=True).read()

            # Sampling, Nyquist frequency
            fs = round(nwb.acquisition["ElectricalSeries"].rate)

            print("Loading ...")
            data = nwb.acquisition["ElectricalSeries"].data[:].astype(np.float64)
            nyq = fs // 2

            # GOOD CHANNELS, opposite of bad channels
            # Anatomical locations of electodes
            goods = np.invert(nwb.electrodes["bad"][:])
            locx = nwb.electrodes.location[:]

            # EPOCHS
            # Speech trials
            # Start, consonant-vowel transition, stop, syllable condition, if/not spoke
            trials = np.array(
                [
                    nwb.trials.start_time[:],
                    nwb.trials.cv_transition_time[:],
                    nwb.trials.stop_time[:],
                    nwb.trials.condition[:],
                    nwb.trials.speak[:],
                ]
            ).T

            # CONSONANT, VOWEL, IRP1/2
            start_time, cv_time, stop_time, condition, spoke = trials.T
            consonants = np.array(list(zip(start_time, cv_time)))
            vowels = np.array(list(zip(cv_time, stop_time)))
            rests = np.array(list(zip(stop_time[:-1], start_time[1:])))
            cx = np.array(
                [
                    consonants.ptp(axis=1)[i] if condition[i].endswith("e") else None
                    for i in range(len(trials))
                ]
            )
            vx = np.array(
                [
                    vowels.ptp(axis=1)[i] if condition[i].endswith("e") else None
                    for i in range(len(trials))
                ]
            )
            condx = np.array(
                [c if c.endswith("e") else None for c in condition], dtype="object"
            )

            # Q: How many session trials are there, in how many did the subject speak?
            print(f"Session had {len(spoke)} trials.")
            print(f"Subject spoke in {len(trials)} trials.")

            # Rest-/i/-rest time triples, round to indices
            # TEST: Assert epoch + padding definitely within signal
            r = int(MIN_REST_LENGTH * fs)
            pad = int(0.5 * fs)
            times = np.concatenate(
                [rests[:-1], consonants[1:-1], vowels[1:-1], rests[1:]], axis=1
            ).reshape((-1, 4, 2))
            indices = np.round(fs * times).astype(int)
            assert indices.min() - r - pad >= 0 and indices.max() + r + pad < len(data)

            print("Finding rest-/i/-rest trials ...")

            # Rest-/i/-rest indices
            ij = [
                [speech_start - r - pad, speech_stop + r + pad]
                for ([_, speech_start], _, _, [speech_stop, _]) in indices
            ]

            # Rest-/i/-rest ECoG
            epochs = np.array(
                [None] + [data[i:j] for (i, j) in ij] + [None],
                dtype=object,
            )
            epochs = np.array(
                [
                    epochs[i] if condition[i].endswith("e") else None
                    for i in range(len(trials))
                ]
            )

            # INDICES
            idx = [None]
            for i in indices:
                idx.append(
                    np.sort(
                        np.append(
                            np.cumsum(i.ptp(axis=1)[1:-1]) + r + pad,
                            [
                                0,
                                pad,
                                pad + r,
                                pad + r + np.sum(i.ptp(axis=1)[1:-1]) + r,
                                pad + r + np.sum(i.ptp(axis=1)[1:-1]) + pad + r,
                            ],
                        )
                    )
                )
            idx.append(None)
            idx = [
                idx[i] if condition[i].endswith("e") else None
                for i in range(len(trials))
            ]
            idx = np.array(idx, dtype=object)

            # EXCLUSIONS
            firstlast = np.full((len(trials),), True)
            firstlast[0] = False
            firstlast[-1] = False

            # IRP1/2 shorter than MIN_REST_LENGTH
            longenough = np.array(
                [True]
                + np.all(indices.ptp(axis=2)[:, [0, -1]] >= fs // 2, axis=1).tolist()
                + [True]
            ).astype(bool)

            # Rest-/i/-rest times
            st = [
                [speech_start - MIN_REST_LENGTH, speech_stop + MIN_REST_LENGTH]
                for ([_, speech_start], _, _, [speech_stop, _]) in times
            ]

            # Overlap with invalid times
            valids = np.full((len(trials),), True)
            if nwb.invalid_times is not None:
                invalids = nwb.invalid_times[:].iloc[:, 0:2].values.tolist()
                for i in range(1, len(trials) - 1):
                    if any(overlap(inv, st[i - 1]) for inv in invalids):
                        valids[i] = False

            includes = firstlast * longenough * valids

            # SAVE
            print("Saving ...")

            # Trials
            def sift(xx):
                return [
                    x if x is not None and includes[i] else None
                    for i, x in enumerate(xx)
                ]

            np.save(
                f"{OUT}/trials/{SUBJECT}_{SESSION}-trials-ee-epochs.npy",
                sift(epochs),
                allow_pickle=True,
            )
            np.save(
                f"{OUT}/trials/{SUBJECT}_{SESSION}-trials-ee-indices.npy",
                sift(idx),
                allow_pickle=True,
            )
            np.save(
                f"{OUT}/trials/{SUBJECT}_{SESSION}-trials-ee-consonants.npy", sift(cx)
            )
            np.save(f"{OUT}/trials/{SUBJECT}_{SESSION}-trials-ee-vowels.npy", sift(vx))
            np.save(
                f"{OUT}/trials/{SUBJECT}_{SESSION}-trials-ee-conditions.npy",
                sift(condx),
                allow_pickle=True,
            )

            # Good channels, anatomical locations
            np.save(
                f"{OUT}/{SUBJECT}_{SESSION}-good-channels.npy",
                goods,
                allow_pickle=True,
            )
            np.save(
                f"{OUT}/{SUBJECT}_{SESSION}-locations.npy",
                locx,
                allow_pickle=True,
            )

            # Valid trials to include
            np.save(
                f"{OUT}/{SUBJECT}_{SESSION}-includes.npy",
                includes,
                allow_pickle=True,
            )

            print("Done.")
            print()

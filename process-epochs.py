"""
Top-level trial data bandpower processing

Authored by Eric Easthope
"""

import os
import warnings
import numpy as np
import scipy.signal as sig
import resampy
import multiprocess as mp
from pathlib import Path
from utils import log_scale_bands

# Filter warnings
warnings.filterwarnings("ignore")

DATA = "data"
DERIVED = "derived"
subjects = ["EC2", "EC9", "GP31", "GP33"]
BANDS = (
    log_scale_bands(12, 35, 2)
    + log_scale_bands(35, 70, 2)
    + log_scale_bands(70, 140, 2)
)

MIN_REST_LENGTH = 0.5
OVERWRITE = True
fs = 3052
nyq = fs // 2
r = int(MIN_REST_LENGTH * fs)
pad = int(0.5 * fs)

# Average consonant/vowel length, cross-session
avg_consonant_len = np.concatenate(
    [
        [c for c in np.load(p, allow_pickle=True) if c is not None]
        for p in Path(DERIVED).glob(f"*/**/*-ee-consonants.npy")
    ]
).mean()
avg_vowel_len = np.concatenate(
    [
        [v for v in np.load(p, allow_pickle=True) if v is not None]
        for p in Path(DERIVED).glob(f"*/**/*-ee-vowels.npy")
    ]
).mean()

for SUBJECT in subjects:
    for path in [p for p in Path(DATA).glob(f"{SUBJECT}*.nwb")]:
        _, SESSION = path.stem.split("_")
        OUT = DERIVED + f"/{SUBJECT}/{SESSION}"
        print(f"File {SUBJECT}_{SESSION}.")

        if os.path.isdir(f"{OUT}/powers") and not OVERWRITE:
            print("Powers exist already, skipping ...")
        else:
            # LOAD
            cx = np.load(
                f"{OUT}/trials/{SUBJECT}_{SESSION}-trials-ee-consonants.npy",
                allow_pickle=True,
            )
            vx = np.load(
                f"{OUT}/trials/{SUBJECT}_{SESSION}-trials-ee-vowels.npy",
                allow_pickle=True,
            )
            idx = np.load(
                f"{OUT}/trials/{SUBJECT}_{SESSION}-trials-ee-indices.npy",
                allow_pickle=True,
            )
            goods = np.load(
                f"{OUT}/{SUBJECT}_{SESSION}-good-channels.npy",
                allow_pickle=True,
            )
            epochs = np.load(
                f"{OUT}/trials/{SUBJECT}_{SESSION}-trials-ee-epochs.npy",
                allow_pickle=True,
            )

            # RE-REFERENCE, subtract average signal from each channel
            def rereference(s):
                return (s.T - s[:, goods].mean(axis=1)).T

            # NOTCH FILTER, 60/120/180 Hz line noise
            def notch(s, Hz):
                # NOTCH FILTER
                Q = 30
                b, a = sig.iirnotch(Hz, Q=Q, fs=fs)
                return sig.filtfilt(b, a, s, axis=0)

            # BANDPASS FILTER
            def bandpass(s, band):
                # BANDPASS FILTER
                [l, h] = band
                low = l / nyq
                high = h / nyq
                ORDER = 3
                sos = sig.butter(ORDER, [low, high], "band", analog=False, output="sos")
                return sig.sosfiltfilt(sos, s, axis=0)

            # ANALYTIC SIGNAL, Hilbert transform
            def analytic(s):
                return sig.hilbert(s, axis=0)

            # POWER
            def power(s):
                return np.abs(s) ** 2.0

            # STRETCH, Time-warp consonant/vowel to average lengths w/ sinc interpolation
            def stretch(s, i):
                [pad1, irp1, con, vow, irp2, pad2] = np.split(s, i[1:-1])
                [rest1, start, cv, stop, rest2] = i[1:-1]

                c_scale = fs * avg_consonant_len / len(con)
                v_scale = fs * avg_vowel_len / len(vow)

                c_fs = np.floor(fs * c_scale).astype(int)
                v_fs = np.floor(fs * v_scale).astype(int)
                c_pad = np.floor(pad * c_scale).astype(int)
                v_pad = np.floor(pad * v_scale).astype(int)

                c_stretched = resampy.resample(
                    s[start - pad : cv + pad],
                    fs,
                    c_fs,
                    filter="sinc_window",
                    axis=0,
                )[c_pad:-c_pad]
                v_stretched = resampy.resample(
                    s[cv - pad : stop + pad],
                    fs,
                    v_fs,
                    axis=0,
                    filter="sinc_window",
                )[v_pad:-v_pad]

                return np.concatenate(
                    [pad1, irp1, c_stretched, v_stretched, irp2, pad2]
                ), np.cumsum(
                    [
                        0,
                        len(pad1),
                        len(irp1),
                        len(c_stretched),
                        len(v_stretched),
                        len(irp2),
                        len(pad2),
                    ]
                )

            # Total process
            def process(e, i, low, high):
                return (
                    stretch(
                        power(
                            analytic(
                                bandpass(
                                    notch(notch(notch(rereference(e), 60), 120), 180),
                                    [low, high],
                                )
                            )
                        ),
                        i,
                    )
                    if e is not None and i is not None
                    else (None, None)
                )

            # Make powers directory if it does not exist
            try:
                os.mkdir(f"{OUT}/powers")
            except FileExistsError:
                pass

            for (low, high) in BANDS:
                l = np.ceil(low).astype(int)
                h = np.ceil(high).astype(int)
                print(f"{l}-{h} Hz ...")
                with mp.Pool(processes=8) as pool:
                    results = pool.starmap(
                        process,
                        [(epochs[i], idx[i], low, high) for i in range(len(epochs))],
                    )
                powers, indices = zip(*results)
                np.save(
                    f"{OUT}/powers/{SUBJECT}_{SESSION}-powers-ee-{l}-{h}.npy",
                    powers,
                    allow_pickle=True,
                )
            np.save(
                f"{OUT}/trials/{SUBJECT}_{SESSION}-trials-ee-indices-stretched.npy",
                indices,
                allow_pickle=True,
            )

            print("Done.")
            print()

"""
Top-level trial data bandpower statistics

Authored by Eric Easthope
"""

import os
import gc
import numpy as np
from pathlib import Path
from utils import log_scale_bands

subjects = ["EC2", "EC9", "GP31", "GP33"]
SUBBANDS = (
    log_scale_bands(12, 35, 2)
    + log_scale_bands(35, 70, 2)
    + log_scale_bands(70, 140, 2)
)

DATA = "data"
DERIVED = "derived"

for p in sorted(Path(DATA).glob(f"*.nwb")):
    SUBJECT, SESSION = p.stem.split("_")
    session = Path(DERIVED) / SUBJECT / SESSION
    print(p.stem)

    # Make bands directory if it does not exist
    OUT = session / "bands"
    try:
        os.mkdir(OUT)
    except FileExistsError:
        pass

    # LOAD
    # Indices
    ix = np.load(
        session / "trials" / f"{SUBJECT}_{SESSION}-trials-ee-indices-stretched.npy",
        allow_pickle=True,
    )
    ix = [i for i in ix if i is not None]

    for band in SUBBANDS:
        low, high = np.ceil(band).astype(int)
        print(low, high)

        # Powers
        px = np.load(
            session / "powers" / f"{SUBJECT}_{SESSION}-powers-ee-{low}-{high}.npy",
            allow_pickle=True,
        )
        px = [p for p in px if p is not None]

        # Truncate powers
        # Common length
        powers = [p[ix[i][1] : ix[i][-2], :] for i, p in enumerate(px)]
        l = min(map(len, powers))
        powers = np.array([p[:l, :] for p in powers])

        # Z-score trials w.r.t. average trial
        # Z-score w/ exclusions for |z| > 3
        powerz = (powers - powers.mean(axis=(0, 1))) / powers.std(axis=(0, 1))
        x = (powerz.mean(axis=2).min(axis=1) >= -3) * (
            powerz.mean(axis=2).max(axis=1) <= 3
        )
        powerzx = powerz[x, :, :]

        # SAVE, average, keep channels
        np.save(
            session
            / "bands"
            / f"{SUBJECT}_{SESSION}-bandpower-{low}-{high}-zscores.npy",
            powerz.mean(axis=0),
        )
        np.save(
            session
            / "bands"
            / f"{SUBJECT}_{SESSION}-bandpower-{low}-{high}-zscores-exclude.npy",
            powerzx.mean(axis=0),
        )
        np.save(
            session / "bands" / f"{SUBJECT}_{SESSION}-bandpower-{low}-{high}-volts.npy",
            powers.mean(axis=0),
        )

        del px, powerz, powerzx, powers
        gc.collect()

    for band in np.split(np.array(SUBBANDS), 3):
        (l1, l2), (h1, h2) = np.ceil(band).astype(int)
        print((l1, l2), (h1, h2))

        # Powers
        px = np.load(
            session / "powers" / f"{SUBJECT}_{SESSION}-powers-ee-{l1}-{l2}.npy",
            allow_pickle=True,
        )
        px = [p for p in px if p is not None]

        px2 = np.load(
            session / "powers" / f"{SUBJECT}_{SESSION}-powers-ee-{h1}-{h2}.npy",
            allow_pickle=True,
        )
        px2 = [p for p in px2 if p is not None]

        # Truncate powers
        # Common length
        powers = [p[ix[i][1] : ix[i][-2], :] for i, p in enumerate(px)]
        powers2 = [p[ix[i][1] : ix[i][-2], :] for i, p in enumerate(px2)]

        l = min(map(len, powers))

        powers = np.array([p[:l, :] for p in powers])
        powers2 = np.array([p[:l, :] for p in powers2])
        poweravg = np.mean([powers, powers2], axis=0)

        # Z-score trials w.r.t. average trial, per channel
        # Z-score w/ exclusions for |z| > 3
        poweravgz = (poweravg - poweravg.mean(axis=(0, 1))) / poweravg.std(axis=(0, 1))
        x = (poweravgz.mean(axis=2).min(axis=1) >= -3) * (
            poweravgz.mean(axis=2).max(axis=1) <= 3
        )
        poweravgzx = poweravgz[x, :, :]

        # SAVE, average, keep channels
        np.save(
            session / "bands" / f"{SUBJECT}_{SESSION}-bandpower-{l1}-{h2}-zscores.npy",
            poweravgz.mean(axis=0),
        )
        np.save(
            session
            / "bands"
            / f"{SUBJECT}_{SESSION}-bandpower-{l1}-{h2}-zscores-exclude.npy",
            poweravgzx.mean(axis=0),
        )
        np.save(
            session / "bands" / f"{SUBJECT}_{SESSION}-bandpower-{l1}-{h2}-volts.npy",
            poweravg.mean(axis=0),
        )

        del px, powers, powers2, poweravg, poweravgz, poweravgzx
        gc.collect()

    print()

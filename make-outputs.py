"""
Top-level trial data bandpower export

Authored by Eric Easthope
"""

import numpy as np
from pathlib import Path

DATA = "data"
DERIVED = "derived"
fs = 3052
MIN_REST_LEN = 1 / 2

bandnames = [
    "beta1",
    "beta2",
    "gamma1",
    "gamma2",
    "highgamma1",
    "highgamma2",
    "beta",
    "gamma",
    "highgamma",
]
BANDS = [
    [12, 21],
    [21, 35],
    [35, 50],
    [50, 70],
    [70, 99],
    [99, 140],
    [12, 35],
    [35, 70],
    [70, 140],
]
sessions = [p.stem.split("_") for p in sorted(Path(DATA).glob(f"*.nwb"))]
bsx = {"t": [], "idx": []}

for name, band in list(zip(bandnames, BANDS)):
    # DEV
    low, high = band

    # LOAD
    # Bandpowers from analytic signal
    bandpowers = [
        np.load(
            Path(DERIVED)
            / SUBJECT
            / SESSION
            / "bands"
            / (SUBJECT + "_" + SESSION + f"-bandpower-{low}-{high}-zscores.npy")
        )
        for SUBJECT, SESSION in sessions
    ]

    # Indices, times
    ix = [
        np.load(
            Path(DERIVED)
            / SUBJECT
            / SESSION
            / "trials"
            / f"{SUBJECT}_{SESSION}-trials-ee-indices-stretched.npy",
            allow_pickle=True,
        )
        for SUBJECT, SESSION in sessions
    ]
    ix = (
        np.array([i for idx in ix for i in idx if i is not None])
        .mean(axis=0)
        .astype(int)
    )
    start, cv, stop = ix[2:-2] - int(fs * MIN_REST_LEN)
    T = np.linspace(0, (ix[-2] - ix[1]) / fs, ix[-2] - ix[1])

    # Common length
    l = min(map(len, bandpowers))
    bandpowers = np.array([bp[:l, :] for bp in bandpowers])
    T = T[:l] - MIN_REST_LEN

    # Good channels, SMC locations
    goods = [
        np.load(
            Path(DERIVED)
            / SUBJECT
            / SESSION
            / (SUBJECT + "_" + SESSION + f"-good-channels.npy"),
            allow_pickle=True,
        )
        for SUBJECT, SESSION in sessions
    ]
    locations = [
        np.load(
            Path(DERIVED)
            / SUBJECT
            / SESSION
            / (SUBJECT + "_" + SESSION + f"-locations.npy"),
            allow_pickle=True,
        )
        for SUBJECT, SESSION in sessions
    ]

    # DEV: hard-coded fix
    smcs = [(l == "precentral") + (l == "postcentral") for l in locations]
    pres = [(l == "precentral") for l in locations]
    posts = [(l == "postcentral") for l in locations]
    smcs[8] = smcs[9]
    pres[8] = pres[9]
    posts[8] = posts[9]

    # All, SMC, outside bandpowers
    # Array-ify
    bp_alls = np.array(
        [bandpowers[s, :, goods[s]].T.mean(axis=1) for s in range(len(bandpowers))]
    )

    bp_smcs = np.array(
        [
            bandpowers[s, :, goods[s] * smcs[s]].T.mean(axis=1)
            for s in range(len(bandpowers))
        ]
    )
    bp_pres = np.array(
        [
            bandpowers[s, :, goods[s] * pres[s]].T.mean(axis=1)
            for s in range(len(bandpowers))
        ]
    )
    bp_posts = np.array(
        [
            bandpowers[s, :, goods[s] * posts[s]].T.mean(axis=1)
            for s in range(len(bandpowers))
        ]
    )

    bp_outs = np.array(
        [
            bandpowers[s, :, goods[s] * np.invert(smcs[s])].T.mean(axis=1)
            for s in range(len(bandpowers))
        ]
    )

    # Z-score
    bp_allz = (bp_alls - bp_alls.mean()) / bp_alls.std()
    bp_smcz = (bp_smcs - bp_smcs.mean()) / bp_smcs.std()
    bp_prez = (bp_pres - bp_pres.mean()) / bp_pres.std()
    bp_postz = (bp_posts - bp_posts.mean()) / bp_posts.std()
    bp_outz = (bp_outs - bp_outs.mean()) / bp_outs.std()

    bsx[name] = {"smc": bp_smcz, "out": bp_outz}
    bsx["t"].append(T)
    bsx["idx"].append(ix[2:-2] - int(fs * MIN_REST_LEN))

# Average, cast to arrays
bsx["t"] = np.average(bsx["t"], axis=0)
bsx["idx"] = np.average(bsx["idx"], axis=0).astype(int)

# SAVE
np.save(f"SESSION-BANDS.npy", bsx)

"""M2 dataset analysis for PCam: split composition, slide/centre provenance, colour and stain statistics,
example patches, and a colour-only reference classifier (how far colour alone gets you).

    python eda.py            # writes reports/m2/dataset_stats.json and reports/m2/fig_*.png
"""
import json
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import rgb2hed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pcam.data import DATA_DIR
from pcam.metrics import binary_metrics

OUT = Path("reports/m2")
SPLITS = ("train", "valid", "test")
N_SAMPLE = 8192  # patches per split for colour statistics
N_LR_TRAIN = 32768  # training patches for the colour-only logistic-regression baseline
SEED = 0


def centre_of(wsi: str) -> str:
    """CAMELYON16 training slides: RUMC = tumor_001-070 + normal_001-100 (+ tumor_111, renamed from
    normal_086); UMCU = tumor_071-110 + normal_101-160. Test slides are a mix of both and not labelled."""
    m = re.match(r"camelyon16_train_(tumor|normal)_(\d+)", wsi)
    if not m:
        return "test (mixed)"
    kind, n = m.group(1), int(m.group(2))
    if kind == "tumor":
        return "RUMC" if n <= 70 or n == 111 else "UMCU"
    return "RUMC" if n <= 100 else "UMCU"


def load_meta(split):
    m = pd.read_csv(DATA_DIR / f"camelyonpatch_level_2_split_{split}_meta.csv", index_col=0)
    with h5py.File(DATA_DIR / f"camelyonpatch_level_2_split_{split}_y.h5", "r") as f:
        y = f["y"][:].reshape(-1)
    assert (m.center_tumor_patch.values == y.astype(bool)).all(), f"{split}: label != center_tumor_patch"
    m["y"] = y
    m["centre"] = m.wsi.map(centre_of)
    return m


def read_rows(split, idx):
    with h5py.File(DATA_DIR / f"camelyonpatch_level_2_split_{split}_x.h5", "r") as f:
        return f["x"][np.sort(idx)]  # h5py fancy indexing needs increasing indices


def colour_features(x):
    """Per-patch colour descriptors: RGB mean/std, H and E optical-density mean/std, and 16-bin RGB histograms."""
    xf = x.astype(np.float32) / 255.0
    hed = rgb2hed(xf.reshape(-1, 96, 3)).reshape(x.shape)
    rgb_m, rgb_s = xf.mean((1, 2)), xf.std((1, 2))
    he_m, he_s = hed[..., :2].mean((1, 2)), hed[..., :2].std((1, 2))
    hist = np.stack([np.stack([np.histogram(p[..., c], bins=16, range=(0, 256))[0] for c in range(3)]).ravel()
                     for p in x]).astype(np.float32) / (96 * 96)
    return {"rgb_mean": rgb_m, "rgb_std": rgb_s, "he_mean": he_m, "he_std": he_s, "hist": hist}


def quality_pass(split, chunk=8192):
    """One sequential pass over all patches: a content hash (exact duplicates) and simple
    background measures (fraction of near-white pixels, mean intensity)."""
    import hashlib
    hashes, white, mean = [], [], []
    with h5py.File(DATA_DIR / f"camelyonpatch_level_2_split_{split}_x.h5", "r") as f:
        d = f["x"]
        for i in range(0, len(d), chunk):
            x = d[i : i + chunk]
            hashes += [hashlib.blake2b(p.tobytes(), digest_size=12).digest() for p in x]
            white.append((x.min(-1) > 215).mean((1, 2)))
            mean.append(x.mean((1, 2, 3)))
    return np.array(hashes), np.concatenate(white), np.concatenate(mean)


def feature_matrix(f):
    return np.concatenate([f["rgb_mean"], f["rgb_std"], f["he_mean"], f["he_std"], f["hist"]], axis=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    meta = {s: load_meta(s) for s in SPLITS}
    stats = {"splits": {}}

    # 1. Composition and provenance
    for s, m in meta.items():
        per_wsi = m.groupby("wsi").size()
        stats["splits"][s] = {
            "patches": len(m),
            "positive_frac": round(float(m.y.mean()), 4),
            "slides": int(m.wsi.nunique()),
            "patches_per_slide": {"median": int(per_wsi.median()), "min": int(per_wsi.min()), "max": int(per_wsi.max())},
            "negatives_with_tumour_outside_centre": int((m.tumor_patch & (m.y == 0)).sum()),
            "by_centre": {c: {"patches": int(len(g)), "slides": int(g.wsi.nunique()), "positive_frac": round(float(g.y.mean()), 4)}
                          for c, g in m.groupby("centre")},
        }
        if s != "test":
            kinds = m.drop_duplicates("wsi").wsi.str.extract(r"_(tumor|normal)_")[0].value_counts()
            stats["splits"][s]["slides_by_type"] = {k: int(v) for k, v in kinds.items()}
            stats["splits"][s]["patches_from_normal_slides"] = int(m.wsi.str.contains("_normal_").sum())
    ws = {s: set(m.wsi) for s, m in meta.items()}
    stats["slide_overlap"] = {f"{a}&{b}": len(ws[a] & ws[b]) for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]}
    stats["missing_values_in_metadata"] = {s: int(m.isna().sum().sum()) for s, m in meta.items()}

    # Data quality over every patch: exact duplicates, label conflicts between duplicates, background
    hashes, quality = {}, {}
    for s in SPLITS:
        h, white, mean = quality_pass(s)
        hashes[s] = h
        y = meta[s].y.values
        _, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
        dup = cnt[inv] > 1
        conflict = pd.Series(y).groupby(inv).nunique().values[inv] > 1
        bg = white > 0.9
        quality[s] = {"duplicate_patches": int(dup.sum()), "unique_patches": int(len(cnt)),
                      "duplicates_with_conflicting_labels": int(conflict.sum()),
                      "background_patches(>90% white)": int(bg.sum()),
                      "background_patches_positive": int((bg & (y == 1)).sum()),
                      "mean_intensity_by_class": {"normal": round(float(mean[y == 0].mean()), 2), "tumour": round(float(mean[y == 1].mean()), 2)}}
        print(f"quality {s}: {quality[s]}", flush=True)
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        quality[f"identical_patches_{a}&{b}"] = len(set(hashes[a].tolist()) & set(hashes[b].tolist()))
    stats["quality"] = quality
    print(json.dumps(stats, indent=1))

    # 2. Colour / stain statistics on a random sample per split
    samples, feats = {}, {}
    for s in SPLITS:
        idx = np.sort(rng.choice(len(meta[s]), N_SAMPLE, replace=False))
        samples[s] = (idx, read_rows(s, idx))
        feats[s] = colour_features(samples[s][1])
        print(f"colour features {s}: {len(idx)} patches", flush=True)
    colour = {}
    for s in SPLITS:
        y = meta[s].y.values[samples[s][0]]
        cen = meta[s].centre.values[samples[s][0]]
        f = feats[s]
        groups = {"all": np.ones_like(y, bool), "tumour": y == 1, "normal": y == 0}
        groups.update({f"centre={c}": cen == c for c in np.unique(cen) if s != "test"})
        colour[s] = {g: {"n": int(mask.sum()),
                         "rgb_mean": np.round(f["rgb_mean"][mask].mean(0), 4).tolist(),
                         "H_od_mean": round(float(f["he_mean"][mask, 0].mean()), 4),
                         "E_od_mean": round(float(f["he_mean"][mask, 1].mean()), 4)}
                     for g, mask in groups.items()}
    stats["colour_sample"] = colour

    # 3. Baseline 1, colour-only: logistic regression on 58 hand-crafted colour features (RGB and H/E
    #    optical-density mean+std, 3x16-bin RGB histograms). Trained on a stratified random 32,768-patch
    #    train subset, evaluated on the FULL validation and test splits. It tells us how much of the
    #    task is solvable from colour alone, without any morphology.
    tr_idx = np.sort(rng.choice(len(meta["train"]), N_LR_TRAIN, replace=False))
    Xtr = feature_matrix(colour_features(read_rows("train", tr_idx)))
    ytr = meta["train"].y.values[tr_idx]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    clf.fit(Xtr, ytr)
    ref = {}
    for s in ("valid", "test"):
        with h5py.File(DATA_DIR / f"camelyonpatch_level_2_split_{s}_x.h5", "r") as f:
            Xs = np.concatenate([feature_matrix(colour_features(f["x"][i : i + 8192])) for i in range(0, len(f["x"]), 8192)])
        ys = meta[s].y.values
        p = clf.predict_proba(Xs)[:, 1]
        ref[s] = binary_metrics(ys, p)
        if s == "test":
            np.savez_compressed(OUT / "colour_lr_test_preds.npz", y=ys, p=p.astype(np.float32))
    stats["colour_only_baseline"] = {"model": "logistic regression (L2, C=1) on 58 colour features",
                                     "n_features": int(Xtr.shape[1]), "n_train": int(len(ytr)), **ref}
    print("colour-only baseline:", {s: {k: round(v, 4) for k, v in r.items()} for s, r in ref.items()})

    (OUT / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    # Figures ---------------------------------------------------------------
    # (a) example patches with the 32x32 label centre
    m, (idx, x) = meta["train"], samples["train"]
    ys = m.y.values[idx]
    pick = np.concatenate([rng.choice(np.flatnonzero(ys == 1), 8, replace=False), rng.choice(np.flatnonzero(ys == 0), 8, replace=False)])
    fig, axes = plt.subplots(2, 8, figsize=(13, 3.6))
    for ax, i in zip(axes.ravel(), pick):
        ax.imshow(x[i]); ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(mpatches.Rectangle((31.5, 31.5), 32, 32, fill=False, ec="#00e5ff", lw=1.2))
    axes[0, 0].set_ylabel("tumour (1)", fontsize=11); axes[1, 0].set_ylabel("normal (0)", fontsize=11)
    fig.suptitle("PCam training patches (96×96 px); the label depends only on the 32×32 centre (box)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig_examples.png", dpi=160); plt.close(fig)

    # (b) split composition: patches per split by class, and by source centre
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    names = list(SPLITS)
    pos = [stats["splits"][s]["patches"] * stats["splits"][s]["positive_frac"] for s in names]
    neg = [stats["splits"][s]["patches"] - p for s, p in zip(names, pos)]
    ax[0].bar(names, neg, color="#4C72B0", label="normal (0)")
    ax[0].bar(names, pos, bottom=neg, color="#DD8452", label="tumour (1)")
    for i, s in enumerate(names):
        ax[0].text(i, stats["splits"][s]["patches"], f"{stats['splits'][s]['patches']:,}\n{stats['splits'][s]['slides']} slides",
                   ha="center", va="bottom", fontsize=8)
    ax[0].set(title="Patches per split", ylabel="patches", ylim=(0, 330000)); ax[0].legend(fontsize=8, loc="upper right")
    cents = ["RUMC", "UMCU"]
    for j, c in enumerate(cents):
        vals = [stats["splits"][s]["by_centre"].get(c, {}).get("patches", 0) for s in ("train", "valid")]
        ax[1].bar(np.arange(2) + (j - 0.5) * 0.38, vals, 0.38, label=c, color=["#55A868", "#8172B3"][j])
    ax[1].set_xticks([0, 1], ["train", "valid"]); ax[1].set(title="Source hospital (CAMELYON16 train slides)", ylabel="patches")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_splits.png", dpi=160); plt.close(fig)

    # (c) stain statistics: H and E optical density by class and by centre
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    f, idx = feats["train"], samples["train"][0]
    yv, cen = meta["train"].y.values[idx], meta["train"].centre.values[idx]
    bins = np.linspace(np.percentile(f["he_mean"][:, 0], 0.5), np.percentile(f["he_mean"][:, 0], 99.5), 60)
    for lab, col, name in [(0, "#4C72B0", "normal"), (1, "#DD8452", "tumour")]:
        ax[0].hist(f["he_mean"][yv == lab, 0], bins, alpha=0.6, color=col, label=name, density=True)
    ax[0].set(title="Haematoxylin density by class (train sample)", xlabel="mean H optical density per patch", ylabel="density")
    ax[0].legend(fontsize=8)
    for c, col in [("RUMC", "#55A868"), ("UMCU", "#8172B3")]:
        v = f["he_mean"][cen == c, 0]
        ax[1].hist(v, bins, alpha=0.6, color=col, density=True, label=f"{c} (mean {v.mean():.3f})")
        ax[1].axvline(v.mean(), color=col, ls="--", lw=1.2)
    ax[1].set(title="Haematoxylin density by hospital (train sample)", xlabel="mean H optical density per patch", ylabel="density")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_stain.png", dpi=160); plt.close(fig)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()

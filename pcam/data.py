"""PCam loading: images memory-mapped from a one-time .npy copy, plus nested label-budget subsets."""
from pathlib import Path

import h5py
import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _npy_copy(h5_path: Path, key: str = "x", chunk: int = 8192) -> Path:
    """Copy an HDF5 dataset to an .npy file next to it (once), so it can be memory-mapped.
    The full train split is 7.25 GB; reading it into RAM does not fit a free Colab runtime
    (12.7 GB) next to PyTorch. A memmap keeps it on disk and lets the OS cache what fits."""
    npy = h5_path.with_suffix(".npy")
    if npy.exists():
        return npy
    tmp = h5_path.with_suffix(".npy.tmp")
    with h5py.File(h5_path, "r") as f:
        d = f[key]
        print(f"one-time copy {h5_path.name} -> {npy.name} ({d.nbytes / 1e9:.2f} GB)", flush=True)
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=d.dtype, shape=d.shape)
        for i in range(0, len(d), chunk):
            out[i : i + chunk] = d[i : i + chunk]
        out.flush()
        del out
    tmp.rename(npy)  # only a finished copy gets the final name
    return npy


def load_split(split: str, data_dir: Path = DATA_DIR, limit: int = 0):
    """Return (x, y): x uint8 [N,96,96,3] (read-only memmap), y uint8 [N]. split in {train, valid, test}.
    limit > 0 reads only the first `limit` samples into RAM (for quick debugging)."""
    stem = Path(data_dir) / f"camelyonpatch_level_2_split_{split}"
    with h5py.File(f"{stem}_y.h5", "r") as f:
        y = f["y"][:].reshape(-1)
    if limit:
        with h5py.File(f"{stem}_x.h5", "r") as f:
            return f["x"][:limit], y[:limit]
    x = np.load(_npy_copy(Path(f"{stem}_x.h5")), mmap_mode="r")
    assert len(x) == len(y), (stem, x.shape, y.shape)
    return x, y


def label_budget_indices(y: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    """Stratified subset of size ~frac*N. Nested across fractions for a fixed seed
    (the 10% subset is contained in the 25% subset), so budgets differ only in size."""
    if frac >= 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        idx = idx[rng.permutation(len(idx))]
        keep.append(idx[: int(round(frac * len(idx)))])
    return np.sort(np.concatenate(keep))


def iterate_batches(x, y, batch_size, shuffle, rng=None, indices=None):
    """Yield (uint8 NHWC tensor, float label tensor) on CPU."""
    idx = np.arange(len(y)) if indices is None else indices
    if shuffle:
        idx = idx[rng.permutation(len(idx))]
    for i in range(0, len(idx), batch_size):
        b = np.sort(idx[i : i + batch_size])  # sorted gather is friendlier to memory
        yield torch.from_numpy(x[b]), torch.from_numpy(y[b].astype(np.float32))

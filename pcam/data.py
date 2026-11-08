"""PCam loading: whole split into RAM as uint8, plus nested label-budget subsets."""
from pathlib import Path

import h5py
import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_split(split: str, data_dir: Path = DATA_DIR, limit: int = 0):
    """Return (x, y): x uint8 [N,96,96,3], y uint8 [N]. split in {train, valid, test}.
    limit > 0 reads only the first `limit` samples (for quick debugging)."""
    sl = slice(0, limit) if limit else slice(None)
    stem = data_dir / f"camelyonpatch_level_2_split_{split}"
    with h5py.File(f"{stem}_x.h5", "r") as f:
        x = f["x"][sl]
    with h5py.File(f"{stem}_y.h5", "r") as f:
        y = f["y"][sl].reshape(-1)
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

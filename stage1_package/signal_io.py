
import numpy as np
import os
from scipy.io import loadmat
import h5py

def try_load_mat_var(mat_path, var_name):
    """
    Load a variable from a MATLAB .mat file (v7 or v7.3).
    Returns a 1D float numpy array.
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(mat_path)
    # Try old MAT
    try:
        data = loadmat(mat_path)
        if var_name in data:
            return np.asarray(data[var_name]).squeeze().astype(float)
        for k in data.keys():
            if k.lower() == var_name.lower():
                return np.asarray(data[k]).squeeze().astype(float)
    except Exception:
        pass
    # Try v7.3 (HDF5)
    try:
        with h5py.File(mat_path, "r") as f:
            if var_name in f:
                return np.array(f[var_name][()]).squeeze().astype(float)
            for k in f.keys():
                if k.lower() == var_name.lower():
                    return np.array(f[k][()]).squeeze().astype(float)
    except Exception as e:
        raise RuntimeError(f"Failed to read {var_name} from {mat_path}: {e}")
    raise KeyError(f"Variable {var_name} not found in {mat_path}")

def slice_indices(idx_in_file, seg_len_s, hop_s, fs_raw):
    """Return (start, end, fs_raw_int) indices for slicing at the original rate."""
    fs_raw = float(fs_raw)
    start = int(round(idx_in_file * hop_s * fs_raw))
    length = int(round(seg_len_s * fs_raw))
    return start, start + length, int(fs_raw)

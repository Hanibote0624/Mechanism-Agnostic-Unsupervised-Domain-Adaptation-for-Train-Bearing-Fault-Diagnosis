
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, get_window, stft as sp_stft, detrend as sp_detrend, resample_poly
from scipy.io import loadmat
import h5py
import os

def try_load_mat_var(mat_path, var_name):
    """
    Load a variable from .mat file. Supports v7 and v7.3.
    Returns a 1D numpy array.
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(mat_path)
    # Try old MAT first
    try:
        data = loadmat(mat_path)
        if var_name in data:
            arr = np.asarray(data[var_name]).squeeze()
            return np.array(arr, dtype=float)
        # case-insensitive fallbacks
        for k in data.keys():
            if k.lower() == var_name.lower():
                arr = np.asarray(data[k]).squeeze()
                return np.array(arr, dtype=float)
    except Exception:
        pass
    # Try v7.3 HDF5
    try:
        with h5py.File(mat_path, "r") as f:
            if var_name in f:
                d = f[var_name][()]
                return np.array(d).squeeze().astype(float)
            for k in f.keys():
                if k.lower() == var_name.lower():
                    d = f[k][()]
                    return np.array(d).squeeze().astype(float)
    except Exception as e:
        raise RuntimeError(f"Failed to read {var_name} from {mat_path}: {e}")
    raise KeyError(f"Variable {var_name} not found in {mat_path}")

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(1e-6, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(x, fs, fmin, fmax, order=4):
    b, a = butter_bandpass(fmin, fmax, fs, order)
    return filtfilt(b, a, x)

def compute_envelope(x):
    return np.abs(hilbert(x))

def compute_env_spectrum(env, fs, n_fft=0, window="hann", detrend="constant"):
    x = env.astype(float)
    if detrend:
        x = sp_detrend(x, type=detrend)
    N = len(x)
    if n_fft is None or n_fft == 0:
        n_fft = 1 << (N-1).bit_length()
    w = get_window(window, N) if window else np.ones(N)
    xw = x * w
    spec = np.fft.rfft(xw, n=n_fft)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(n_fft, d=1.0/fs)
    return freqs, mag

def compute_stft(x, fs, nperseg=2048, noverlap=1024, nfft=2048, window="hann"):
    f, t, Zxx = sp_stft(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary=None, padded=False)
    return f, t, np.abs(Zxx)

def resample_to(x, fs_in, fs_out):
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    up = int(fs_out // g)
    down = int(fs_in // g)
    y = resample_poly(x, up, down)
    return y

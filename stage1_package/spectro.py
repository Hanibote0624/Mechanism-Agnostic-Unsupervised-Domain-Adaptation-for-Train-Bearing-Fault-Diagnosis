
import numpy as np
from scipy.signal import butter, filtfilt, get_window, stft as sp_stft, detrend as sp_detrend, resample_poly

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(1e-6, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    if high <= low:
        raise ValueError("Invalid bandpass range after normalization")
    from scipy.signal import butter
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(x, fs, fmin, fmax, order=4):
    b, a = butter_bandpass(fmin, fmax, fs, order)
    return filtfilt(b, a, x)

def resample_to(x, fs_in, fs_out):
    # Rational resampling for quality
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    up = int(fs_out // g)
    down = int(fs_in // g)
    return resample_poly(x, up, down)

def compute_stft_mag(x, fs, nperseg, noverlap, nfft, window="hann"):
    f, t, Zxx = sp_stft(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary=None, padded=False)
    mag = np.abs(Zxx).astype(np.float32)
    return f, t, mag

def mag_postprocess(mag, log1p=True, qmin=0.01, qmax=0.99, eps=1e-6):
    M = mag
    if log1p:
        M = np.log1p(M)
    lo = np.quantile(M, qmin)
    hi = np.quantile(M, qmax)
    if hi <= lo:
        hi = lo + 1e-6
    M = np.clip(M, lo, hi)
    M = (M - lo) / (hi - lo + eps)
    return M

def crop_fmax(f, M, fmax):
    if fmax is None or fmax <= 0:
        return f, M
    idx = np.where(f <= fmax)[0]
    if len(idx) == 0:
        return f, M
    last = idx[-1] + 1
    return f[:last], M[:last, :]

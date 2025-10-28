
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, json
import pandas as pd
import numpy as np
from pathlib import Path
from signal_utils import try_load_mat_var, bandpass_filter, compute_envelope, compute_env_spectrum, compute_stft, resample_to
from pathlib import PurePosixPath, PureWindowsPath
import os

def parse_args():
    ap = argparse.ArgumentParser(description="Route B: reconstruct segments from raw .mat and extract envelope/stft features")
    ap.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    return ap.parse_args()

def load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def slice_from_meta(row, cols):
    idx = int(row[cols["idx_in_file"]])
    seg_len_s = float(row[cols["segment_len_s"]])
    hop_s = float(row.get(cols["hop_s"], seg_len_s))
    fs_raw = float(row.get(cols.get("fs_raw","fs"), row["fs"]))
    start = int(round(idx * hop_s * fs_raw))
    length = int(round(seg_len_s * fs_raw))
    return start, start + length, int(fs_raw)

def band_energy(freqs, mag, center, rel_width=0.05):
    if not np.isfinite(center) or center <= 0:
        return np.nan
    half = max(1.0, center * rel_width)
    lo, hi = center - half, center + half
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return 0.0
    return float(np.trapz(mag[m], freqs[m]))

def sideband_sum(freqs, mag, center, fr, k_max=3, rel_width=0.05):
    if not (np.isfinite(center) and np.isfinite(fr)) or center<=0 or fr<=0:
        return np.nan
    total = 0.0
    for k in range(1, k_max+1):
        total += band_energy(freqs, mag, center + k*fr, rel_width)
        total += band_energy(freqs, mag, center - k*fr, rel_width)
    return total


def normalize_path(p):
    # Expand ~ and env vars; normalize separators
    p = os.path.expanduser(os.path.expandvars(str(p)))
    return os.path.normpath(p)

def apply_path_map(p, path_map):
    p_norm = normalize_path(p)
    if not path_map:
        return p_norm
    for rule in path_map:
        src = rule.get("from") or rule.get("src") or ""
        dst = rule.get("to") or rule.get("dst") or ""
        if src and p_norm.startswith(src):
            candidate = normalize_path(p_norm.replace(src, dst, 1))
            return candidate
    return p_norm


def process_row(row, cfg, cols, out_dir):
    uid = str(row[cols["uid"]])
    domain = str(row.get(cols["domain"], "unknown"))
    file_path = apply_path_map(str(row[cols["file_path"]]), cfg.get("path_map"))
    var_name = str(row[cols["var_name"]])
    bpfo = float(row.get(cols["BPFO"], np.nan))
    bpfi = float(row.get(cols["BPFI"], np.nan))
    bsf  = float(row.get(cols["BSF"],  np.nan))
    ftf  = float(row.get(cols["FTF"],  np.nan))
    fr   = float(row.get(cols["fr_hz"], np.nan))

    try:
        full = try_load_mat_var(file_path, var_name)
    except Exception as e:
        if cfg["proc"]["skip_if_missing"]:
            return None, f"[SKIP] {uid}: cannot read {file_path}:{var_name} ({e})"
        else:
            raise

    s0, s1, fs_raw = slice_from_meta(row, cols)
    if s0<0 or s1>len(full):
        return None, f"[SKIP] {uid}: slice out of bounds ({s0}:{s1} vs {len(full)})"
    seg_raw = np.asarray(full[s0:s1], dtype=float)

    target_fs = int(cfg["target_fs"])
    if fs_raw != target_fs:
        seg = resample_to(seg_raw, fs_raw, target_fs)
        fs = target_fs
    else:
        seg = seg_raw
        fs = fs_raw

    fmin = cfg["bandpass"]["fmin"]
    fmax = cfg["bandpass"]["fmax"]
    order = cfg["bandpass"]["order"]
    try:
        from signal_utils import butter_bandpass
        # if filter fails due to too-short segment, skip filtering
        seg_bp = bandpass_filter(seg, fs, fmin, fmax, order=order)
    except Exception:
        seg_bp = seg
    env = compute_envelope(seg_bp)
    freqs, mag = compute_env_spectrum(env, fs, n_fft=cfg["env_spec"]["n_fft"], window=cfg["env_spec"]["window"], detrend=cfg["env_spec"]["detrend"])

    stft_mag = None
    if cfg["stft"]["enable"]:
        f, t, stft_mag = compute_stft(seg_bp, fs, nperseg=cfg["stft"]["nperseg"], noverlap=cfg["stft"]["noverlap"], nfft=cfg["stft"]["nfft"], window=cfg["stft"]["window"])

    relw = float(cfg["feat_band_rel_width"])
    kmax = int(cfg["sidebands"]["k_max"])
    feats = {}
    for name, c in [("BPFO", bpfo), ("BPFI", bpfi), ("BSF", bsf)]:
        feats[f"{name}_main_e"] = band_energy(freqs, mag, c, relw)
        feats[f"{name}_sb_fr_e"] = sideband_sum(freqs, mag, c, fr, k_max=kmax, rel_width=relw)
    feats["FTF_main_e"] = band_energy(freqs, mag, ftf, relw)

    def safe_ratio(a, b):
        a = float(a) if np.isfinite(a) else 0.0
        b = float(b) if np.isfinite(b) and b>0 else np.nan
        return a/b if np.isfinite(b) else np.nan
    feats["BPFI_sbfr_ratio"] = safe_ratio(feats["BPFI_sb_fr_e"], feats["BPFI_main_e"])
    feats["BPFO_sbfr_ratio"] = safe_ratio(feats["BPFO_sb_fr_e"], feats["BPFO_main_e"])
    feats["BSF_sbftf_ratio"] = safe_ratio(sideband_sum(freqs, mag, bsf, ftf, k_max=kmax, rel_width=relw), feats["BSF_main_e"])

    dom_dir = "preproc_src" if domain=="source" else "preproc_tgt"
    base = Path(out_dir) / dom_dir
    sig_dir = base / "signals"
    feat_dir = base / "features"
    sig_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    if cfg["save"]["raw_resampled"]:
        np.save(sig_dir / f"{uid}__raw.npy", seg.astype(np.float32))
    if cfg["save"]["envelope"]:
        np.save(sig_dir / f"{uid}__env.npy", env.astype(np.float32))
    if cfg["save"]["env_spectrum"]:
        np.save(feat_dir / f"{uid}__env_freqs.npy", freqs.astype(np.float32))
        np.save(feat_dir / f"{uid}__env_mag.npy", mag.astype(np.float32))
    if cfg["save"]["stft_mag"] and (stft_mag is not None):
        np.save(feat_dir / f"{uid}__stft_mag.npy", stft_mag.astype(np.float32))

    out = {"uid": uid, "domain": domain, "file_path": file_path, "var_name": var_name, "fs_out": fs}
    for k, v in feats.items():
        out[k] = v
    for k in ["label","sensor","load_hp","rpm","BPFO","BPFI","BSF","FTF","fr_hz","fault_size_in","bearing_type"]:
        if k in row and pd.notna(row[k]):
            out[k] = row[k]
    return out, f"[OK] {uid}"

def main():
    args = parse_args()
    cfg = load_config(args.config)
    meta = pd.read_csv(cfg["meta_csv"])
    cols = cfg["columns"]
    out_dir = cfg["out_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    feats_rows = []
    logs = []
    max_rows = int(cfg["proc"]["max_rows"] or 0)
    for i, row in meta.iterrows():
        if max_rows>0 and i>=max_rows:
            break
        out, msg = process_row(row, cfg, cols, out_dir)
        if out is not None:
            feats_rows.append(out)
        logs.append(msg)
        if (i+1) % int(cfg["proc"]["log_every"]) == 0:
            print(f"Processed {i+1}/{len(meta)}")

    with open(Path(out_dir)/"route_b.log", "w", encoding="utf-8") as f:
        f.write("\n".join(logs))

    if len(feats_rows):
        df = pd.DataFrame(feats_rows)
        for dom, sub in df.groupby("domain"):
            sub.to_csv(Path(out_dir)/f"features_{dom}.csv", index=False)

    print("Done. Outputs under:", out_dir)

if __name__ == "__main__":
    main()

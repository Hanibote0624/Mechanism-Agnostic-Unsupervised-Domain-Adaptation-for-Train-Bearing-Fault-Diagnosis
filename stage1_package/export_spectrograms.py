
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from src.signal_io import try_load_mat_var, slice_indices
from src.spectro import apply_bandpass, resample_to, compute_stft_mag, mag_postprocess, crop_fmax
from src.save_image import spectrogram_to_image
from scipy.signal import detrend as sp_detrend

def parse_args():
    ap = argparse.ArgumentParser(description="Stage 1: Export unified spectrograms from meta and raw .mat files")
    ap.add_argument("--config", type=str, default="exp/configs/base.yaml")
    return ap.parse_args()

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    meta = pd.read_csv(cfg["meta_csv"])
    cols = cfg["columns"]

    out_dir = Path(cfg["out_dir"])
    src_dir = out_dir / "preproc" / "spectrograms" / "source"
    tgt_dir = out_dir / "preproc" / "spectrograms" / "target"
    safe_mkdir(src_dir)
    safe_mkdir(tgt_dir)

    # Parameters
    target_fs = int(cfg["target_fs"])
    bp = cfg["bandpass"]
    detrend = bool(cfg.get("detrend", True))
    st = cfg["stft"]
    mag_cfg = cfg["magnitude"]
    im_cfg  = cfg["image"]
    proc = cfg["proc"]

    log_every = int(proc.get("log_every", 200))
    max_rows  = int(proc.get("max_rows", 0))
    skip_missing = bool(proc.get("skip_if_missing", True))

    logs = []
    n_done = 0

    for i, row in meta.iterrows():
        if max_rows > 0 and n_done >= max_rows:
            break

        uid = str(row[cols["uid"]])
        domain = str(row.get(cols["domain"], "source")).lower()
        file_path = str(row[cols["file_path"]])
        var_name  = str(row[cols["var_name"]])

        # 1) Load full signal
        try:
            full = try_load_mat_var(file_path, var_name)
        except Exception as e:
            msg = f"[SKIP] {uid}: cannot read {file_path}:{var_name} ({e})"
            logs.append(msg); print(msg)
            if not skip_missing: raise
            continue

        # 2) Slice segment at raw fs
        fs_raw = float(row.get(cols.get("fs_raw","fs"), row["fs"]))
        idx = int(row[cols["idx_in_file"]])
        seg_len_s = float(row[cols["segment_len_s"]])
        hop_s = float(row.get(cols["hop_s"], seg_len_s))
        s0, s1, _ = slice_indices(idx, seg_len_s, hop_s, fs_raw)
        if s0 < 0 or s1 > len(full):
            msg = f"[SKIP] {uid}: slice out of bounds ({s0}:{s1} vs {len(full)})"
            logs.append(msg); print(msg)
            continue
        x = np.asarray(full[s0:s1], dtype=float)

        # 3) Resample to target fs
        if int(fs_raw) != target_fs:
            x = resample_to(x, fs_raw, target_fs)
            fs = target_fs
        else:
            fs = int(fs_raw)

        # 4) Detrend + Bandpass
        if detrend:
            x = sp_detrend(x, type="constant")
        if bp.get("enable", True):
            try:
                x = apply_bandpass(x, fs, bp["fmin"], bp["fmax"], order=bp["order"])
            except Exception as e:
                # If bandpass fails (e.g., too short), fallback to raw
                pass

        # 5) STFT -> magnitude
        f, t, M = compute_stft_mag(x, fs, st["nperseg"], st["noverlap"], st["nfft"], window=st["window"])
        f, M = crop_fmax(f, M, st.get("fmax_plot", None))

        # 6) Post-process magnitude (log, quantile clip, norm)
        M = mag_postprocess(M, log1p=mag_cfg["log1p"], qmin=mag_cfg["qmin"], qmax=mag_cfg["qmax"], eps=mag_cfg["epsilon"])

        # 7) Save image (dark = high energy)
        img = spectrogram_to_image(M, width=im_cfg["width"], height=im_cfg["height"], invert_for_dark_high=im_cfg["invert_for_dark_high"])
        out_path = (src_dir if domain=="source" else tgt_dir) / f"{uid}.{im_cfg['format']}"
        img.save(out_path)

        n_done += 1
        if n_done % log_every == 0:
            print(f"Exported {n_done} spectrograms")

    # Write log
    (out_dir / "stage1_export.log").write_text("\n".join(logs), encoding="utf-8")
    print(f"Done. Exported {n_done} spectrogram(s). Output dir: {out_dir}")

if __name__ == "__main__":
    main()

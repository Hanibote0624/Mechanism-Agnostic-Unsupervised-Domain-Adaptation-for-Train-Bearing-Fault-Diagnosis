
# Stage 1 Package — Unified Spectrogram Export

## What this does
- Reads `meta_segments.csv` (uid/domain/file_path/var_name/idx_in_file/segment_len_s/hop_s/fs_raw).
- Reconstructs each segment from the original `.mat` at the **original sampling rate**, then resamples to `target_fs` (default 32kHz).
- Detrend → (optional) bandpass 500–10kHz → STFT (default 2048/1024/2048 Hann).
- Magnitude processing: `log(1+x)` → quantile clip (q1–q99) → normalize to [0,1].
- Exports **grayscale spectrograms** with **dark = high energy** (fixed size), to:
  - `preproc/spectrograms/source/uid.png`
  - `preproc/spectrograms/target/uid.png`

## Files
- `exp/configs/base.yaml` — all parameters & paths.
- `src/signal_io.py` — .mat loading & slicing indices.
- `src/spectro.py` — resample, bandpass, STFT, post-process.
- `src/save_image.py` — grayscale export (dark=high).
- `src/export_spectrograms.py` — main entry.
- `viz/plot_grid.py` — make a 4x4 contact sheet for quick QA.

## Quick start (inside this folder)
```bash
# 1) Optional: dry-run — process first 50 rows
sed -i 's/max_rows: 0/max_rows: 50/' exp/configs/base.yaml  # macOS: use gsed or edit manually

# 2) Export spectrograms
python3 src/export_spectrograms.py --config exp/configs/base.yaml

# 3) Preview a grid (choose one)
python3 viz/plot_grid.py --dir preproc/spectrograms/source --out figs_source_grid.jpg
python3 viz/plot_grid.py --dir preproc/spectrograms/target --out figs_target_grid.jpg
```

## Notes
- If a `.mat` is missing or a slice is out of bounds, the script logs it and continues (configurable).
- You can tune `stft.nperseg`, `bandpass`, `magnitude.qmin/qmax`, `image.width/height`, and `stft.fmax_plot` to suit your data.
- The exported images are **normalized & style-unified** for downstream CNN/UDA training.


# Stage 2 — Source-Domain Training (Only Time–Frequency Features)

Train a CNN (ResNet18/ConvNeXt-T) on unified spectrogram images exported in Stage 1.
Group-aware splitting prevents file-level leakage. Outputs metrics and plots for your paper.

## Quick Start
```bash
unzip stage2_package.zip -d stage2_package
cd stage2_package
python3 -m pip install -r requirements.txt

# Edit config to point to your meta & spectrogram dirs (Stage 1 outputs)
vim exp/configs/base.yaml

# Train on source domain
python3 src/train_source.py --config exp/configs/base.yaml
```

## Inputs
- `meta_csv`: the same meta used in Stage 1 (must include uid/domain/label/file_path).
- `spectrogram_dir`: base dir containing `source/` spectrogram images from Stage 1.
- Only `domain=="source"` rows with existing images are used for training.

## Outputs
- `outputs/split.csv` — group-aware train/val/test split.
- `outputs/label_map.json` — label↔id mapping.
- `outputs/ckpt_best.pt` — best model by validation macro-F1.
- `outputs/metrics_test.json` — macro-F1, per-class F1, confusion matrix, PR curves info.
- `outputs/confusion_matrix_test.png`, `outputs/reliability_test.png`, `outputs/calibration.json`.
- `outputs/train_log.jsonl` — learning curve log.


# Stage 3 — UDA + Pseudo-Labeling (Mechanism-Agnostic)

基于 Stage 1 的统一谱图与 Stage 2 的源域模型，本包实现：
- **无监督域适配**（DANN / MMD / CORAL 可选，带 warm-up）
- **动态源样本加权**（更像目标域的源样本权重大）
- **高阈值伪标签微调**（两轮 0.90→0.85，少量微调）
- **可视化**：训练曲线与 t-SNE

## 使用方法
```bash
unzip stage3_package.zip -d stage3_package
cd stage3_package
python3 -m pip install -r requirements.txt

# 训练 UDA（默认 DANN）
python3 src/train_uda.py --config exp/configs/base.yaml

# 伪标签微调 + 最终预测（片段/文件级）
python3 src/pseudo_label.py --config exp/configs/base.yaml --ckpt outputs/ckpt_uda_best.pt

# 曲线出图
python3 viz/plot_curves.py --hist outputs/train_hist.json --out_prefix outputs/curves
```

> 仅使用**时频图像**作为学习输入；目标域不使用机理量做判别。

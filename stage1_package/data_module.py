
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

class SpectrogramDataset(Dataset):
    def __init__(self, df, img_dir, img_ext=".png", augment=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.img_ext = img_ext
        self.augment = augment
        labels = sorted(self.df["label"].dropna().unique().tolist())
        self.label2id = {l:i for i,l in enumerate(labels)}
        self.id2label = {i:l for l,i in self.label2id.items()}
        self.df["y"] = self.df["label"].map(self.label2id).astype(int)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        uid = str(r["uid"]); y = int(r["y"])
        path = self.img_dir / f"{uid}{self.img_ext}"
        img = Image.open(path).convert("L")
        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
        if self.augment is not None:
            x = self.augment(x.unsqueeze(0)).squeeze(0)
        return {"image": x, "label": y, "uid": uid}

def load_meta_filter(meta_csv, spectrogram_dir, cfg_cols):
    meta = pd.read_csv(meta_csv)
    # use only source domain rows
    if cfg_cols["domain"] in meta.columns:
        meta = meta[meta[cfg_cols["domain"]].str.lower()=="source"].copy()
    img_dir = Path(spectrogram_dir) / "source"
    exts = [".png",".jpg",".jpeg",".bmp"]
    exist = set([p.stem for p in img_dir.glob("*") if p.suffix.lower() in exts])
    meta = meta[meta[cfg_cols["uid"]].astype(str).isin(exist)].copy()
    group_col = cfg_cols.get("group_id","group_id")
    if group_col not in meta.columns or meta[group_col].isna().all():
        meta["group_id"] = meta[cfg_cols["file_path"]].astype(str)
    meta = meta.dropna(subset=[cfg_cols["label"]])
    meta = meta.rename(columns={cfg_cols["uid"]:"uid", cfg_cols["label"]:"label", cfg_cols["file_path"]:"file_path"})
    return meta

def split_groups(meta, train_ratio, val_ratio, test_ratio, seed=3407, use_groups=True):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    if use_groups:
        groups = meta["group_id"].astype(str).values
    else:
        groups = meta["file_path"].astype(str).values
    idx = np.arange(len(meta))
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio+test_ratio, random_state=seed)
    tr_idx, temp_idx = next(gss.split(idx, groups=groups))
    temp = meta.iloc[temp_idx].reset_index(drop=True)
    tr = meta.iloc[tr_idx].reset_index(drop=True)
    # split temp into val/test
    test_size = test_ratio / (val_ratio + test_ratio + 1e-12)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed+1)
    va_idx, te_idx = next(gss2.split(np.arange(len(temp)), groups=temp["group_id"].astype(str).values))
    va = temp.iloc[va_idx].reset_index(drop=True)
    te = temp.iloc[te_idx].reset_index(drop=True)
    return tr, va, te

def make_dataloaders(tr, va, te, img_dir, img_ext, batch_size, num_workers, augment):
    ds_tr = SpectrogramDataset(tr, img_dir, img_ext, augment=augment)
    ds_va = SpectrogramDataset(va, img_dir, img_ext, augment=None)
    ds_te = SpectrogramDataset(te, img_dir, img_ext, augment=None)
    # unify label map
    label2id = ds_tr.label2id
    for ds in [ds_va, ds_te]:
        ds.label2id = label2id
        ds.id2label = {i:l for i,l in enumerate(sorted(label2id, key=lambda k: label2id[k]))}
        ds.df["y"] = ds.df["label"].map(label2id).astype(int)
    from torch.utils.data import DataLoader
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te

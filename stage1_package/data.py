
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

class ImgDataset(Dataset):
    def __init__(self, df, img_dir, ext=".png", labels=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.ext = ext
        self.labels = labels
        if labels:
            labels_unique = sorted(self.df["label"].dropna().unique().tolist())
            self.label2id = {l:i for i,l in enumerate(labels_unique)}
            self.id2label = {i:l for l,i in self.label2id.items()}
            self.df["y"] = self.df["label"].map(self.label2id).astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        uid = str(r["uid"])
        path = self.img_dir / f"{uid}{self.ext}"
        img = Image.open(path).convert("L")
        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
        out = {"image": x, "uid": uid}
        if self.labels:
            out["label"] = int(r["y"])
        return out

def filter_existing(meta_csv, spectrogram_dir, cols, domain):
    meta = pd.read_csv(meta_csv)
    if cols["domain"] in meta.columns:
        meta = meta[meta[cols["domain"]].str.lower()==domain].copy()
    img_dir = Path(spectrogram_dir) / domain
    exts = [".png",".jpg",".jpeg",".bmp"]
    exist = set([p.stem for p in img_dir.glob("*") if p.suffix.lower() in exts])
    meta = meta[meta[cols["uid"]].astype(str).isin(exist)].copy()
    if cols.get("group_id","group_id") in meta.columns:
        meta["group_id"] = meta[cols["group_id"]].astype(str)
    else:
        meta["group_id"] = meta[cols["file_path"]].astype(str)
    if domain == "source":
        meta = meta.dropna(subset=[cols["label"]]).rename(columns={cols["uid"]:"uid", cols["label"]:"label", cols["file_path"]:"file_path"})
    else:
        meta = meta.rename(columns={cols["uid"]:"uid", cols["file_path"]:"file_path"})
    return meta

def split_source(meta_src, ratios, seed=3407, use_groups=True):
    train_ratio, val_ratio, test_ratio = ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    groups = meta_src["group_id"].astype(str).values if use_groups else meta_src["file_path"].astype(str).values
    idx = np.arange(len(meta_src))
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio+test_ratio, random_state=seed)
    tr_idx, temp_idx = next(gss.split(idx, groups=groups))
    temp = meta_src.iloc[temp_idx].reset_index(drop=True)
    tr = meta_src.iloc[tr_idx].reset_index(drop=True)
    test_size = test_ratio / (val_ratio + test_ratio + 1e-12)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed+1)
    va_idx, te_idx = next(gss2.split(np.arange(len(temp)), groups=temp["group_id"].astype(str).values))
    va = temp.iloc[va_idx].reset_index(drop=True)
    te = temp.iloc[te_idx].reset_index(drop=True)
    return tr, va, te


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.utils import set_seed, save_json
from src.data import filter_existing, ImgDataset
from src.backbone import FeatureBackbone

@torch.no_grad()
def infer_target(model, loader, device):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc="infer_tgt", leave=False):
        x = batch["image"].to(device)
        uid = batch["uid"]
        logits, _ = model(x, return_feat=True)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        for i, u in enumerate(uid):
            rows.append({"uid": u, "prob": prob[i].tolist()})
    return rows

def fine_tune_with_pseudo(model, pseudo_df, img_dir_tgt, img_ext, cfg_pl, device):
    from torch.utils.data import Dataset, DataLoader
    class PseudoDS(Dataset):
        def __init__(self, df):
            self.df = df.reset_index(drop=True)
        def __len__(self): return len(self.df)
        def __getitem__(self, idx):
            r = self.df.iloc[idx]
            path = Path(img_dir_tgt) / f"{r['uid']}{img_ext}"
            from PIL import Image
            import numpy as np, torch
            img = Image.open(path).convert("L")
            x = torch.from_numpy(np.array(img, dtype=np.float32)/255.0).unsqueeze(0)
            y = int(r["y"])
            return {"image": x, "label": y}

    ds_pl = PseudoDS(pseudo_df)
    dl_pl = DataLoader(ds_pl, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg_pl["lr"], weight_decay=cfg_pl["weight_decay"])
    crit = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(cfg_pl["epochs_each"]):
        loss_sum = 0.0
        for batch in tqdm(dl_pl, desc=f"ft_pl_epoch{epoch+1}", leave=False):
            x = batch["image"].to(device); y = batch["label"].to(device)
            logits, _ = model(x, return_feat=True)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * x.size(0)
        print(f"[PL] epoch {epoch+1} loss={loss_sum/len(ds_pl):.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="exp/configs/base.yaml")
    ap.add_argument("--ckpt", type=str, default=None, help="UDA checkpoint (ckpt_uda_best.pt)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config,"r",encoding="utf-8"))

    set_seed(cfg.get("seed", 3407))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    spectro_dir = Path(cfg["spectrogram_dir"])

    meta_src = filter_existing(cfg["meta_csv"], spectro_dir, cfg["columns"], domain="source")
    meta_tgt = filter_existing(cfg["meta_csv"], spectro_dir, cfg["columns"], domain="target")

    labels = sorted(meta_src["label"].dropna().unique().tolist())
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for i,l in enumerate(labels)}
    n_classes = len(labels)

    from torch.utils.data import DataLoader
    ds_tgt = ImgDataset(meta_tgt, spectro_dir/"target", ext=cfg["image"]["ext"], labels=False)
    dl_tgt = DataLoader(ds_tgt, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    model = FeatureBackbone(cfg["train"]["backbone"], n_classes).to(device)
    ckpt = args.ckpt or (out_dir/"ckpt_uda_best.pt").as_posix()
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd, strict=False)

    # 1) infer
    rows = infer_target(model, dl_tgt, device)
    save_json({"rows": rows}, out_dir/"target_probs.json")

    # 2) first round
    th1 = float(cfg["plabel"]["threshold_first"])
    preds = []
    for r in rows:
        prob = np.array(r["prob"])
        k = int(prob.argmax()); p = float(prob[k])
        if p >= th1:
            preds.append({"uid": r["uid"], "y": k, "label": labels[k], "prob": p})
    import pandas as pd
    df1 = pd.DataFrame(preds)
    df1.to_csv(out_dir/"target_pseudo_round1.csv", index=False)

    if cfg["plabel"]["enable"] and len(df1) > 0:
        fine_tune_with_pseudo(model, df1, (spectro_dir/"target").as_posix(), cfg["image"]["ext"], cfg["plabel"], device)
        # 3) second round
        if cfg["plabel"]["max_rounds"] >= 2:
            rows2 = infer_target(model, dl_tgt, device)
            th2 = float(cfg["plabel"]["threshold_second"])
            preds2 = []
            for r in rows2:
                prob = np.array(r["prob"])
                k = int(prob.argmax()); p = float(prob[k])
                if p >= th2:
                    preds2.append({"uid": r["uid"], "y": k, "label": labels[k], "prob": p})
            df2 = pd.DataFrame(preds2)
            df2.to_csv(out_dir/"target_pseudo_round2.csv", index=False)
            if len(df2) > 0:
                fine_tune_with_pseudo(model, df2, (spectro_dir/"target").as_posix(), cfg["image"]["ext"], cfg["plabel"], device)

    torch.save(model.state_dict(), out_dir/"ckpt_uda_plabel.pt")

    # final predictions
    rows_final = infer_target(model, dl_tgt, device)
    seg_rows = []
    for r in rows_final:
        prob = np.array(r["prob"]); k = int(prob.argmax()); p = float(prob[k])
        seg_rows.append({"uid": r["uid"], "pred": k, "label": labels[k], "prob": p})
    df_seg = pd.DataFrame(seg_rows)
    df_seg.to_csv(out_dir/"target_segment_preds.csv", index=False)

    # aggregate by group_id if present
    meta_tgt2 = meta_tgt.merge(df_seg[["uid","pred","prob"]], on="uid", how="left")
    if "group_id" in meta_tgt2.columns:
        agg = meta_tgt2.groupby("group_id").agg(pred=("pred", lambda x: int(np.bincount(x.dropna().astype(int)).argmax()) if x.notna().any() else -1),
                                                mean_prob=("prob","mean"),
                                                n=("uid","count")).reset_index()
        agg["label"] = agg["pred"].apply(lambda k: labels[k] if k>=0 and k < len(labels) else "NA")
        agg.to_csv(out_dir/"target_file_preds.csv", index=False)

    print("Pseudo-labeling & prediction completed. Outputs saved to", out_dir)

if __name__ == "__main__":
    main()

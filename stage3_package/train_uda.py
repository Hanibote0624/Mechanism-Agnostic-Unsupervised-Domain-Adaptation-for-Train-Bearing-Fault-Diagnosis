
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
from sklearn.manifold import TSNE

from src.utils import set_seed, save_json
from src.data import filter_existing, split_source, ImgDataset
from src.backbone import FeatureBackbone
from src.domain import GRL, DomainDiscriminator, mmd_rbf

def class_weights_from_counts(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)

def make_loaders(meta_src_tr, meta_src_va, meta_tgt_all, img_dir, ext, bs_src, bs_tgt, num_workers):
    from torch.utils.data import DataLoader
    ds_src_tr = ImgDataset(meta_src_tr, img_dir/"source", ext=ext, labels=True)
    ds_src_va = ImgDataset(meta_src_va, img_dir/"source", ext=ext, labels=True)
    ds_tgt = ImgDataset(meta_tgt_all, img_dir/"target", ext=ext, labels=False)
    dl_src_tr = DataLoader(ds_src_tr, batch_size=bs_src, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_src_va = DataLoader(ds_src_va, batch_size=bs_src, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_tgt = DataLoader(ds_tgt, batch_size=bs_tgt, shuffle=True, num_workers=num_workers, pin_memory=True)
    return ds_src_tr, ds_src_va, ds_tgt, dl_src_tr, dl_va, dl_tgt

@torch.no_grad()
def eval_source(model, loader, device, classes):
    model.eval()
    ys, ps = [], []
    for batch in tqdm(loader, desc="eval_src", leave=False):
        x = batch["image"].to(device)
        y = batch["label"].cpu().numpy()
        logits, _ = model(x, return_feat=True)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        ys.append(y); ps.append(prob)
    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)
    from sklearn.metrics import f1_score, confusion_matrix
    y_pred = y_prob.argmax(axis=1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"macro_f1": float(macro_f1), "confusion_matrix": cm}

def run_tsne(model, loader_src, loader_tgt, device, n_samples=500, out_path=None):
    model.eval()
    feats = []; labs = []
    with torch.no_grad():
        for batch in loader_src:
            x = batch["image"].to(device)
            _, f = model(x, return_feat=True)
            feats.append(f.cpu().numpy())
            labs += [0]*f.size(0)
            if len(labs) >= n_samples: break
        cnt = 0
        for batch in loader_tgt:
            x = batch["image"].to(device)
            _, f = model(x, return_feat=True)
            feats.append(f.cpu().numpy())
            labs += [1]*f.size(0)
            cnt += f.size(0)
            if cnt >= n_samples: break
    X = np.concatenate(feats, axis=0); y = np.array(labs)
    X2 = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca').fit_transform(X)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,4))
    m0 = y==0; m1 = y==1
    plt.scatter(X2[m0,0], X2[m0,1], s=10, label='source')
    plt.scatter(X2[m1,0], X2[m1,1], s=10, label='target')
    plt.legend(); plt.tight_layout()
    if out_path: plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="exp/configs/base.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config,"r",encoding="utf-8"))

    set_seed(cfg.get("seed", 3407))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    spectro_dir = Path(cfg["spectrogram_dir"])
    cols = cfg["columns"]

    meta_src = filter_existing(cfg["meta_csv"], spectro_dir, cols, domain="source")
    meta_tgt = filter_existing(cfg["meta_csv"], spectro_dir, cols, domain="target")
    tr, va, te = split_source(meta_src, (cfg["split"]["train_ratio"], cfg["split"]["val_ratio"], cfg["split"]["test_ratio"]), seed=cfg["split"]["random_seed"], use_groups=cfg["split"]["use_groups"])
    save_json({"n_src_tr": len(tr), "n_src_va": len(va), "n_src_te": len(te), "n_tgt": len(meta_tgt)}, out_dir/"data_sizes.json")
    pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")]).to_csv(out_dir/"split_source.csv", index=False)

    ds_tr, ds_va, ds_tgt, dl_src_tr, dl_src_va, dl_tgt = make_loaders(tr, va, meta_tgt, spectro_dir, cfg["image"]["ext"], cfg["train"]["batch_size_src"], cfg["train"]["batch_size_tgt"], cfg["train"]["num_workers"])
    n_classes = len(ds_tr.label2id)
    label_map = {int(i):l for i,l in ds_tr.id2label.items()}
    save_json(label_map, out_dir/"label_map.json")

    model = FeatureBackbone(cfg["train"]["backbone"], n_classes).to(device)
    if os.path.exists(cfg["stage2_ckpt"]):
        sd = torch.load(cfg["stage2_ckpt"], map_location=device)
        model.load_state_dict(sd, strict=False)
    disc = DomainDiscriminator(model.feat_dim, hidden=256).to(device)
    grl = GRL(lambd=1.0)

    # losses & opt
    if cfg["train"]["class_weighting"].lower() == "balanced":
        cw = np.bincount(ds_tr.df["y"].values, minlength=n_classes).astype(float)
        cw = cw.sum() / (n_classes * np.maximum(cw, 1.0))
        cls_weight = torch.tensor(cw, dtype=torch.float32).to(device)
    else:
        cls_weight = None
    crit_cls = nn.CrossEntropyLoss(weight=cls_weight, label_smoothing=cfg["uda"]["cls_label_smoothing"])
    crit_dom = nn.CrossEntropyLoss()
    opt = optim.AdamW(list(model.parameters())+list(disc.parameters()), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    sch = optim.lr_scheduler.StepLR(opt, step_size=cfg["train"]["step_lr"]["step_size"], gamma=cfg["train"]["step_lr"]["gamma"]) if cfg["train"]["step_lr"]["enable"] else None

    best_f1, best_epoch = -1, -1
    hist = {"epoch": [], "val_macro_f1": [], "domain_acc": [], "mmd": []}

    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train(); disc.train()
        lam = min(1.0, epoch / max(1, cfg["uda"]["warmup_epochs"])) * cfg["uda"]["lambda_uda"]
        grl.lambd = lam
        it_src = iter(dl_src_tr); it_tgt = iter(dl_tgt)
        n_steps = min(len(dl_src_tr), len(dl_tgt))
        dom_correct = 0; dom_total = 0; mmd_vals = []

        for _ in range(n_steps):
            try: b_src = next(it_src)
            except StopIteration: it_src = iter(dl_src_tr); b_src = next(it_src)
            try: b_tgt = next(it_tgt)
            except StopIteration: it_tgt = iter(dl_tgt); b_tgt = next(it_tgt)

            xs = b_src["image"].to(device); ys = b_src["label"].to(device)
            xt = b_tgt["image"].to(device)
            logits_s, fs = model(xs, return_feat=True)
            _, ft = model(xt, return_feat=True)

            # source classification (dynamic weighting)
            loss_cls = crit_cls(logits_s, ys)
            if cfg["uda"]["dyn_weight"]["enable"]:
                with torch.no_grad():
                    pt = torch.softmax(disc(fs.detach()), dim=1)[:,1]
                    alpha = float(cfg["uda"]["dyn_weight"]["alpha"])
                    w = alpha + (1.0 - alpha) * pt
                loss_cls = loss_cls * w.mean()

            # domain loss / distance
            m = cfg["uda"]["method"].lower()
            if m == "dann":
                f_all = torch.cat([fs, ft], dim=0)
                d_logits = disc(grl(f_all))
                d_labels = torch.cat([torch.zeros(fs.size(0), dtype=torch.long), torch.ones(ft.size(0), dtype=torch.long)], dim=0).to(device)
                loss_uda = crit_dom(d_logits, d_labels)
                dom_pred = d_logits.argmax(dim=1)
                dom_correct += (dom_pred == d_labels).sum().item()
                dom_total += d_labels.numel()
                mmd_val = 0.0
            elif m == "mmd":
                loss_uda = mmd_rbf(fs, ft, sigma=None)
                mmd_val = float(loss_uda.detach().cpu().item())
            elif m == "coral":
                def coral(x, y):
                    xm = x - x.mean(dim=0, keepdim=True)
                    ym = y - y.mean(dim=0, keepdim=True)
                    cx = (xm.t() @ xm) / (x.size(0)-1+1e-8)
                    cy = (ym.t() @ ym) / (y.size(0)-1+1e-8)
                    return ((cx - cy)**2).mean()
                loss_uda = coral(fs, ft)
                mmd_val = float(loss_uda.detach().cpu().item())
            else:
                loss_uda = torch.tensor(0.0, device=device); mmd_val = 0.0

            loss = loss_cls + lam * loss_uda
            opt.zero_grad(); loss.backward(); opt.step()
            if mmd_val: mmd_vals.append(mmd_val)

        # eval on source val
        @torch.no_grad()
        def eval_source(model, loader):
            model.eval()
            ys, ps = [], []
            for batch in loader:
                x = batch["image"].to(device)
                y = batch["label"].cpu().numpy()
                logits, _ = model(x, return_feat=True)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                ys.append(y); ps.append(prob)
            y_true = np.concatenate(ys); y_prob = np.concatenate(ps)
            from sklearn.metrics import f1_score
            y_pred = y_prob.argmax(axis=1)
            return f1_score(y_true, y_pred, average="macro")

        val_f1 = eval_source(model, dl_src_va)
        dom_acc = (dom_correct / max(1, dom_total)) if dom_total>0 else 0.0
        hist["epoch"].append(epoch); hist["val_macro_f1"].append(float(val_f1)); hist["domain_acc"].append(float(dom_acc))
        hist["mmd"].append(float(np.mean(mmd_vals)) if mmd_vals else 0.0)
        save_json(hist, out_dir/"train_hist.json")
        print(f"[Epoch {epoch}] val_macro_f1={val_f1:.4f} domain_acc={dom_acc:.3f} mmd={hist['mmd'][-1]:.5f}")
        if sch: sch.step()
        if epoch % cfg["log"]["save_every"] == 0:
            torch.save(model.state_dict(), out_dir/f"ckpt_uda_epoch{epoch}.pt")
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save(model.state_dict(), out_dir/"ckpt_uda_best.pt")

    # TSNE after UDA (best model)
    model.load_state_dict(torch.load(out_dir/"ckpt_uda_best.pt", map_location=device))
    from torch.utils.data import DataLoader
    ds_src_va_small = ImgDataset(tr.sample(min(800,len(tr))), spectro_dir/"source", ext=cfg["image"]["ext"], labels=True)
    ds_tgt_eval = ImgDataset(meta_tgt.sample(min(800,len(meta_tgt))), spectro_dir/"target", ext=cfg["image"]["ext"], labels=False)
    dl_src_va_small = DataLoader(ds_src_va_small, batch_size=64, shuffle=False)
    dl_tgt_small = DataLoader(ds_tgt_eval, batch_size=64, shuffle=False)
    run_tsne(model, dl_src_va_small, dl_tgt_small, device, n_samples=400, out_path=(Path(cfg["out_dir"])/"tsne_after.png").as_posix())

if __name__ == "__main__":
    main()

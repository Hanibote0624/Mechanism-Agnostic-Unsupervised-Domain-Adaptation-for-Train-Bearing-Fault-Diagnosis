
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data_module import load_meta_filter, split_groups, make_dataloaders
from src.transforms import ComposeAug
from src.model_backbone import make_backbone
from src.metrics import compute_metrics, plot_confusion_matrix, expected_calibration_error, plot_reliability_diagram

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def class_weights_from_counts(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)

def train_one_epoch(model, loader, criterion, optimizer, device, augment_module=None):
    model.train()
    loss_sum = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        if augment_module is not None:
            x = augment_module(x)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
    return loss_sum / len(loader.dataset)

@torch.no_grad()
def eval_model(model, loader, device, n_classes):
    model.eval()
    ys, ps = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["image"].to(device)
        y = batch["label"].cpu().numpy()
        logits = model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        ys.append(y); ps.append(prob)
    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)
    return y_true, y_prob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="exp/configs/base.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    set_seed(cfg.get("seed", 3407))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_csv = cfg["meta_csv"]
    spectro_dir = Path(cfg["spectrogram_dir"])
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta_filter(meta_csv, spectro_dir, cfg["columns"])
    assert len(meta)>0, "No source spectrograms found. Please run Stage 1 first."
    tr, va, te = split_groups(meta, cfg["split"]["train_ratio"], cfg["split"]["val_ratio"], cfg["split"]["test_ratio"],
                              seed=cfg["split"]["random_seed"], use_groups=cfg["split"]["use_groups"])
    pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")]).to_csv(out_dir/"split.csv", index=False)

    augment = ComposeAug(cfg["augment"])
    subdir = cfg["image"]["subdir"]
    img_dir = (spectro_dir / subdir).as_posix()
    from src.data_module import make_dataloaders
    ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te = make_dataloaders(tr, va, te, img_dir, cfg["image"]["ext"], cfg["train"]["batch_size"], cfg["train"]["num_workers"], augment)
    n_classes = len(ds_tr.label2id)
    label_map = {int(i):l for i,l in ds_tr.id2label.items()}
    json.dump(label_map, open(out_dir/"label_map.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

    model = make_backbone(cfg["train"]["backbone"], n_classes).to(device)

    if cfg["train"]["class_weighting"].lower() == "balanced":
        cw = class_weights_from_counts(ds_tr.df["y"].values, n_classes).to(device)
    else:
        cw = None
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg["train"]["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    if cfg["train"]["step_lr"]["enable"]:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg["train"]["step_lr"]["step_size"], gamma=cfg["train"]["step_lr"]["gamma"])
    else:
        scheduler = None

    best_f1, best_epoch = -1, -1
    for epoch in range(1, cfg["train"]["epochs"]+1):
        tr_loss = train_one_epoch(model, dl_tr, criterion, optimizer, device, augment_module=augment)
        y_true_va, y_prob_va = eval_model(model, dl_va, device, n_classes)
        from src.metrics import compute_metrics
        m_va = compute_metrics(y_true_va, y_prob_va, [label_map[i] for i in range(n_classes)])
        if scheduler: scheduler.step()
        if epoch % cfg["log"]["save_every"] == 0:
            torch.save(model.state_dict(), out_dir/f"ckpt_epoch{epoch}.pt")
        if m_va["macro_f1"] > best_f1:
            best_f1, best_epoch = m_va["macro_f1"], epoch
            torch.save(model.state_dict(), out_dir/"ckpt_best.pt")
        with open(out_dir/"train_log.jsonl","a",encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": tr_loss, "val_macro_f1": m_va["macro_f1"]})+"\n")
        print(f"[Epoch {epoch}] loss={tr_loss:.4f} val_macro_f1={m_va['macro_f1']:.4f} (best@{best_epoch}:{best_f1:.4f})")

    model.load_state_dict(torch.load(out_dir/"ckpt_best.pt", map_location=device))
    y_true_te, y_prob_te = eval_model(model, dl_te, device, n_classes)
    from src.metrics import compute_metrics, plot_confusion_matrix, expected_calibration_error, plot_reliability_diagram
    m_te = compute_metrics(y_true_te, y_prob_te, [label_map[i] for i in range(n_classes)])
    json.dump(m_te, open(out_dir/"metrics_test.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    import numpy as np
    cm = np.array(m_te["confusion_matrix"])
    plot_confusion_matrix(cm, [label_map[i] for i in range(n_classes)], out_dir/"confusion_matrix_test.png")
    ece, xs, ys = expected_calibration_error(y_true_te, y_prob_te, n_bins=cfg["eval"]["reliability_bins"])
    plot_reliability_diagram(xs, ys, out_dir/"reliability_test.png")
    with open(out_dir/"calibration.json","w",encoding="utf-8") as f:
        json.dump({"ECE": ece}, f, indent=2)

    print("Done. Best epoch:", best_epoch, "Test macro-F1:", m_te["macro_f1"])

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import yaml, torch, numpy as np
from src.data_module import load_meta_filter, make_dataloaders, split_groups
from src.model_backbone import make_backbone
from src.metrics import compute_metrics, plot_confusion_matrix, expected_calibration_error, plot_reliability_diagram

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="outputs/ckpt_best.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    spectro_dir = Path(cfg["spectrogram_dir"])
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta_filter(cfg["meta_csv"], spectro_dir, cfg["columns"])
    tr, va, te = split_groups(meta, cfg["split"]["train_ratio"], cfg["split"]["val_ratio"], cfg["split"]["test_ratio"],
                              seed=cfg["split"]["random_seed"], use_groups=cfg["split"]["use_groups"])
    ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te = make_dataloaders(
        tr, va, te, (spectro_dir / cfg["image"]["subdir"]).as_posix(),
        cfg["image"]["ext"], cfg["train"]["batch_size"], cfg["train"]["num_workers"], augment=None
    )
    n_classes = len(ds_tr.label2id)
    id2label = {i:l for i,l in ds_tr.id2label.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_backbone(cfg["train"]["backbone"], n_classes).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    def run(dl):
        ys, ps = [], []
        with torch.no_grad():
            for b in dl:
                x = b["image"].to(device)
                y = b["label"].numpy()
                p = torch.softmax(model(x), dim=1).cpu().numpy()
                ys.append(y); ps.append(p)
        return np.concatenate(ys), np.concatenate(ps)

    y_true_te, y_prob_te = run(dl_te)
    m_te = compute_metrics(y_true_te, y_prob_te, [id2label[i] for i in range(n_classes)])
    json.dump(m_te, open(out_dir/"metrics_test.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

    import numpy as np
    cm = np.array(m_te["confusion_matrix"])
    plot_confusion_matrix(cm, [id2label[i] for i in range(n_classes)], out_dir/"confusion_matrix_test.png")
    ece, xs, ys = expected_calibration_error(y_true_te, y_prob_te, n_bins=cfg["eval"]["reliability_bins"])
    plot_reliability_diagram(xs, ys, out_dir/"reliability_test.png")
    with open(out_dir/"calibration.json","w",encoding="utf-8") as f:
        json.dump({"ECE": ece}, f, indent=2)

if __name__ == "__main__":
    main()

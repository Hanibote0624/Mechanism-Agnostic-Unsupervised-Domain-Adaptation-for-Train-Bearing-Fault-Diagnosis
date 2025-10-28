
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, pandas as pd
from pathlib import Path

def normalize_path(p):
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

def main(meta_csv, path_map=None, sample=50):
    df = pd.read_csv(meta_csv)
    miss = 0
    rows = []
    for i, row in df.iterrows():
        p = str(row["file_path"])
        q = apply_path_map(p, path_map)
        ok = os.path.exists(q)
        if not ok:
            miss += 1
            if len(rows) < sample:
                rows.append({"uid": row.get("uid", i), "orig": p, "mapped": q})
    total = len(df)
    print(f"Total rows: {total}, missing files: {miss}")
    if rows:
        print("\nExamples of missing after mapping:")
        for r in rows[:sample]:
            print(f"- uid={r['uid']}\n  orig:  {r['orig']}\n  mapped:{r['mapped']}")
    if path_map:
        print("\nUsing path_map:")
        for r in path_map:
            print(f"  {r}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=str, required=True, help="Path to meta_segments.csv")
    ap.add_argument("--path_map", type=str, default="", help="JSON string for a list of {from,to} rules")
    ap.add_argument("--sample", type=int, default=10, help="How many missing examples to print")
    args = ap.parse_args()
    import json
    pm = json.loads(args.path_map) if args.path_map else None
    main(args.meta, pm, sample=args.sample)

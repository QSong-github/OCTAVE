"""查 Hist2ST 的 NaN 根因：calcADJ 的 Grid 裁边是否留下孤立点。

gcn.py 的 aggregate:  num_neigh = adj.sum(1); mask = adj.div(num_neigh)
孤立点 ⇒ num_neigh=0 ⇒ 0/0 = nan ⇒ 整个前向 nan。
"""
import os, sys, json, glob
import numpy as np, h5py
sys.path.insert(0, "/blue/qsong1/wang.qing/systema4ST/src/shims")
sys.path.insert(1, "/blue/qsong1/wang.qing/systema4ST/methods/Hist2ST")
from graph_construction import calcADJ
import anndata as ad

B = "/blue/qsong1/wang.qing/he2st/HEST/eval/bench_data"
MAXSPOT = 4000
rng = np.random.default_rng(0)

for cohort in ("PRAD", "IDC", "SKCM"):
    ids = [l.split(",")[0] for l in
           open(os.path.join(B, cohort, "splits", "train_0.csv")).read().splitlines()[1:] if l.strip()]
    print(f"\n=== {cohort}  train_0 共 {len(ids)} 片 ===")
    for sid in ids[:4]:
        with h5py.File(os.path.join(B, cohort, "patches", f"{sid}.h5"), "r") as h:
            bk = "barcodes" if "barcodes" in h else "barcode"
            bc = np.asarray(h[bk][:]).flatten().astype(str).tolist()
        a = ad.read_h5ad(os.path.join(B, cohort, "adata", f"{sid}.h5ad"))
        bidx = {b: i for i, b in enumerate(map(str, a.obs_names))}
        keep = np.array([bidx[b] for b in bc])
        ctr = np.stack([a.obs["array_row"].to_numpy()[keep],
                        a.obs["array_col"].to_numpy()[keep]], 1).astype(np.int64)
        n_total = len(ctr)
        if n_total > MAXSPOT:
            gx = (ctr[:,0]-ctr[:,0].min()) // max(1,(np.ptp(ctr[:,0])+1)//40)
            gy = (ctr[:,1]-ctr[:,1].min()) // max(1,(np.ptp(ctr[:,1])+1)//40)
            key = gx*1000+gy; sel=[]
            for k in np.unique(key):
                idx = np.where(key==k)[0]
                sel.append(rng.choice(idx, min(max(1,int(round(MAXSPOT*len(idx)/n_total))), len(idx)), replace=False))
            ctr = ctr[np.concatenate(sel)[:MAXSPOT]]
        for tag in ("Grid", "NA"):
            adj = calcADJ(ctr.astype(np.float32), k=4, pruneTag=tag)
            deg = adj.sum(1).numpy() if hasattr(adj, "numpy") else np.asarray(adj).sum(1)
            iso = int((deg == 0).sum())
            print(f"  {sid[:12]:<12} n={len(ctr):5d} prune={tag:<5} 孤立点={iso:5d} "
                  f"({100*iso/len(ctr):.1f}%)  最小度={deg.min():.0f}")

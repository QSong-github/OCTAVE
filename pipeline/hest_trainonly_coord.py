# -*- coding: utf-8 -*-
"""补“坐标分区 + 训练侧均值”一行，与 xen_controls.py 的 trainonly_coord 并列。
分区：测试样本自身像素坐标 KMeans(20)，与 hest_controls.py 的 oracle_coord 完全相同（同 random_state、n_init、同 spot 子集）。
均值：同队列其它样本（官方折训练侧）的 spot 经 array_row/array_col → 测试样本像素系的仿射映射
（在测试样本上最小二乘拟合，Visium 阵列坐标与像素坐标严格仿射）放到测试样本的坐标系，
再按最近簇中心指派，簇均值来自其实测表达；训练侧没有 spot 的簇用训练侧总体均值。
与编码器无关，只跑一次；同时重算 oracle_coord 供核对。"""
import os, sys, glob, json, warnings, numpy as np, anndata as ad
from scipy import sparse
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
B = "/path/to/he2st/HEST/eval/bench_data"; EMB = "/path/to/project/results/hest_emb"
R = "/path/to/project/results"; K = 20; REF = "phikon_v2"   # 只用其 barcode 列表，取与 hest_controls 相同的 spot 子集
def per_gene_pcc(P, Y):
    P = P - P.mean(0); Y = Y - Y.mean(0); num = (P * Y).sum(0); den = np.sqrt((P ** 2).sum(0) * (Y ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"): r = num / den
    return float(np.nanmean(r))
def group_means(Y, lab):
    out = np.empty_like(Y)
    for c in np.unique(lab): m = lab == c; out[m] = Y[m].mean(0)
    return out
def coords(a):
    for k in ("spatial", "pxl", "xy", "coords"):
        if k in a.obsm: return np.asarray(a.obsm[k], np.float64)[:, :2]
    cols = [c for c in a.obs.columns if c.lower() in ("array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres", "x", "y")]
    assert len(cols) >= 2, f"找不到坐标: obsm={list(a.obsm.keys())} obs={list(a.obs.columns)[:12]}"
    return a.obs[cols[:2]].to_numpy(np.float64)
out = {}
for c in sorted(os.listdir(B)):
    if not os.path.isdir(os.path.join(B, c, "adata")): continue
    genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]; S = {}
    for f in sorted(glob.glob(os.path.join(B, c, "adata", "*.h5ad"))):
        sid = os.path.basename(f)[:-5]; ef = os.path.join(EMB, f"{sid}_{REF}.npz")
        if not os.path.exists(ef): continue
        bc = np.load(ef, allow_pickle=True)["bc"].tolist()
        a = ad.read_h5ad(f); pos = {b: i for i, b in enumerate(a.obs_names.astype(str))}; sub = a[np.array([pos[b] for b in bc])]
        gi = [list(sub.var_names.astype(str)).index(g) for g in genes]
        Y = np.log1p(np.asarray(sub.X.todense() if sparse.issparse(sub.X) else sub.X, np.float64)[:, gi]).astype(np.float32)
        ar = sub.obs[["array_row", "array_col"]].to_numpy(np.float64)
        S[sid] = (Y, coords(sub), ar)
    for sid, (Y, xy, ar) in S.items():
        n = Y.shape[0]; kk = min(K, n)
        km = KMeans(n_clusters=kk, n_init=4, random_state=0).fit(xy); lab_xy = km.labels_
        rec = {"cohort": c, "n": int(n), "oracle_coord": per_gene_pcc(group_means(Y, lab_xy), Y)}
        tr = [s for s in S if s != sid]
        if tr:
            A1 = np.c_[ar, np.ones(n)]; M, *_ = np.linalg.lstsq(A1, xy, rcond=None)
            resid = float(np.abs(A1 @ M - xy).max())
            Ytr = np.concatenate([S[s][0] for s in tr]); ARtr = np.concatenate([S[s][2] for s in tr])
            ltr = km.predict(np.c_[ARtr, np.ones(len(ARtr))] @ M); g = Ytr.mean(0)
            mu = np.stack([Ytr[ltr == u].mean(0) if (ltr == u).any() else g for u in range(kk)])
            rec["trainonly_coord"] = per_gene_pcc(mu[lab_xy], Y); rec["n_train_samples"] = len(tr)
            rec["clusters_unseen_in_train"] = int(sum((ltr == u).sum() == 0 for u in range(kk))); rec["affine_max_resid_px"] = resid
        else:
            rec["trainonly_coord"] = None; rec["n_train_samples"] = 0
        out[sid] = rec
        print(f"  {sid} n={n} coord={rec['oracle_coord']:.4f} trainonly_coord={rec['trainonly_coord']} unseen={rec.get('clusters_unseen_in_train')} resid={rec.get('affine_max_resid_px')}", flush=True)
o = f"{R}/hest_trainonly_coord.json"; json.dump({"K": K, "ref_barcodes": REF, "samples": out}, open(o, "w"), indent=1); print(f"{len(out)} 样本 -> {o}")

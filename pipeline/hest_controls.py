# -*- coding: utf-8 -*-
"""审稿意见（第二份）#1：HEST 基准上的分区对照，逐编码器、逐样本，配方与 hest_blocks.py 逐参数一致。
oracle_image      = 本样本图像嵌入 PCA(50)→KMeans(20)→实测域均值（应等于 hest_blocks 的 blk_k20，用作核对）
oracle_coord      = 仅用点坐标 KMeans(20) 的分区 + 实测域均值（空间 Voronoi 型对照）
oracle_random     = 把图像分区标签随机置换（等规模随机分区）+ 实测域均值
trainonly_image   = 可学习域预测器：分区仍来自本样本图像，但域均值只用同队列其它样本（官方折的训练侧）估计：
                    训练样本的嵌入投到本样本的 PCA 基，用本样本的 KMeans 指派簇，簇均值来自训练样本的实测表达。"""
import os, sys, json, glob, warnings, numpy as np, anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
B = "/blue/qsong1/wang.qing/he2st/HEST/eval/bench_data"; EMB = "/blue/qsong1/wang.qing/systema4ST/results/hest_emb"
R = "/blue/qsong1/wang.qing/systema4ST/results"; K = 20
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
enc = sys.argv[1]; out = {}
for c in sorted(os.listdir(B)):
    if not os.path.isdir(os.path.join(B, c, "adata")): continue
    genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]; S = {}
    for f in sorted(glob.glob(os.path.join(B, c, "adata", "*.h5ad"))):
        sid = os.path.basename(f)[:-5]; ef = os.path.join(EMB, f"{sid}_{enc}.npz")
        if not os.path.exists(ef): continue
        z = np.load(ef, allow_pickle=True); X, bc = z["X"].astype(np.float32), z["bc"].tolist()
        a = ad.read_h5ad(f); pos = {b: i for i, b in enumerate(a.obs_names.astype(str))}; sub = a[np.array([pos[b] for b in bc])]
        gi = [list(sub.var_names.astype(str)).index(g) for g in genes]
        Y = np.log1p(np.asarray(sub.X.todense() if sparse.issparse(sub.X) else sub.X, np.float64)[:, gi]).astype(np.float32)
        S[sid] = (X, Y, coords(sub))
    for sid, (X, Y, xy) in S.items():
        n = Y.shape[0]; kk = min(K, n); rng = np.random.default_rng(0)
        pca = PCA(n_components=min(50, X.shape[1], n - 1), random_state=0).fit(X); pc = pca.transform(X)
        km = KMeans(n_clusters=kk, n_init=4, random_state=0).fit(pc); lab = km.labels_
        lab_xy = KMeans(n_clusters=kk, n_init=4, random_state=0).fit_predict(xy)
        rec = {"cohort": c, "n": int(n), "oracle_image": per_gene_pcc(group_means(Y, lab), Y),
               "oracle_coord": per_gene_pcc(group_means(Y, lab_xy), Y),
               "oracle_random": per_gene_pcc(group_means(Y, rng.permutation(lab)), Y)}
        tr = [s for s in S if s != sid]
        if tr:
            Xtr = np.concatenate([S[s][0] for s in tr]); Ytr = np.concatenate([S[s][1] for s in tr])
            ltr = km.predict(pca.transform(Xtr)); g = Ytr.mean(0)
            mu = np.stack([Ytr[ltr == u].mean(0) if (ltr == u).any() else g for u in range(kk)])
            rec["trainonly_image"] = per_gene_pcc(mu[lab], Y); rec["n_train_samples"] = len(tr)
            rec["clusters_unseen_in_train"] = int(sum((ltr == u).sum() == 0 for u in range(kk)))
        else:
            rec["trainonly_image"] = None; rec["n_train_samples"] = 0
        out[sid] = rec
        print(f"  {sid} n={n} img={rec['oracle_image']:.4f} coord={rec['oracle_coord']:.4f} rand={rec['oracle_random']:.4f} trainonly={rec['trainonly_image']}", flush=True)
o = f"{R}/hest_controls_{enc}.json"; json.dump({"encoder": enc, "K": K, "samples": out}, open(o, "w"), indent=1); print(f"{enc}: {len(out)} 样本 -> {o}")

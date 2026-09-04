# -*- coding: utf-8 -*-
"""块预言机在 HEST 的 72 个样本 × 30 个编码器上。配方与 src/blocks_xen_bands.py 逐参数一致：
   该样本自己的图像嵌入 → PCA(50) → KMeans(k, n_init=4, random_state=0) → 每格填**实测**域均值。
它是预言机（用了测试样本的实测 Y），所以是天花板而非竞争者；命题 1 保证它是该划分的精确上限。
分母同时给两个：官方 ridge（α=100/(D·G)）与留一队列选 α 的 ridge。前者欠正则会压低分母、
把比值抬高，因此以后者为准，两个都报。"""
import os, sys, json, glob, warnings
import numpy as np, anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
B = "/blue/qsong1/wang.qing/he2st/HEST/eval/bench_data"
EMB = "/blue/qsong1/wang.qing/systema4ST/results/hest_emb"
R = "/blue/qsong1/wang.qing/systema4ST/results"
KS = [int(x) for x in os.environ.get("BLK_K", "20").split(",")]
ZS = os.environ.get("BLK_ZSCORE", "0") == "1"      # 2026-09-03：稳健性检查——PCA 前各维标准化

def per_gene_pcc(P, Y):
    P = P - P.mean(0); Y = Y - Y.mean(0)
    num = (P * Y).sum(0); den = np.sqrt((P ** 2).sum(0) * (Y ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return float(np.nanmean(r))

def group_means(Y, lab):
    out = np.empty_like(Y)
    for c in np.unique(lab):
        m = lab == c
        out[m] = Y[m].mean(0)
    return out

enc = sys.argv[1]
out = {}
for c in sorted(os.listdir(B)):
    if not os.path.isdir(os.path.join(B, c, "adata")): continue
    genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]
    for f in sorted(glob.glob(os.path.join(B, c, "adata", "*.h5ad"))):
        sid = os.path.basename(f)[:-5]
        ef = os.path.join(EMB, f"{sid}_{enc}.npz")
        if not os.path.exists(ef): continue
        z = np.load(ef, allow_pickle=True); X, bc = z["X"], z["bc"].tolist()
        a = ad.read_h5ad(f)
        pos = {b: i for i, b in enumerate(a.obs_names.astype(str))}
        sub = a[np.array([pos[b] for b in bc])]
        gi = [list(sub.var_names.astype(str)).index(g) for g in genes]
        Y = np.asarray(sub.X.todense() if sparse.issparse(sub.X) else sub.X, np.float64)[:, gi]
        Y = np.log1p(Y).astype(np.float32)
        Xf = X.astype(np.float32)
        if ZS:
            Xf = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-6)
        pc = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1), random_state=0).fit_transform(Xf)
        rec = {"cohort": c, "n": int(Y.shape[0])}
        for k in KS:
            kk = min(k, Y.shape[0])
            lab = KMeans(n_clusters=kk, n_init=4, random_state=0).fit_predict(pc)
            rec[f"blk_k{k}"] = per_gene_pcc(group_means(Y, lab), Y)
        out[sid] = rec
        print(f"  {sid} n={Y.shape[0]} " + " ".join(f"blk{k}={rec[f'blk_k{k}']:.4f}" for k in KS), flush=True)
o = f"{R}/hest_blocks_{'z_' if ZS else ''}{enc}.json"
json.dump({"encoder": enc, "K": KS, "recipe": ("zscore->" if ZS else "") + "PCA(50)->KMeans(n_init=4,seed=0)->measured group means",
           "samples": out}, open(o, "w"), indent=1)
print(f"{enc}: {len(out)} 样本 -> {o}")

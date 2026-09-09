# -*- coding: utf-8 -*-
"""审稿意见：归纳式（inductive）分区预测器 + 带空间上下文的岭回归（Xenium，同 16×16 块 CV）。
inductive：每折只用训练块的特征拟合 PCA(50)+KMeans(20)，测试块 bin 用该模型指派簇，簇均值来自训练块；分区与均值都不看测试块。
transductive（现有）：KMeans 在整片特征上拟合，均值只来自训练块。
context ridge：特征 = [自身, 8 邻域均值, 第二环（≤48 µm）均值] 的岭回归，检验空间上下文是否恢复细结构。
输出标量 PCC 与最细带 β1（同算子）。"""
import argparse, json, os, sys, numpy as np, anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
sys.path.insert(0, "/path/to/systema4ST/src")
from per_gene_xen import build_operator, per_gene_pcc, block_cv_predict
PREP = "/path/to/systema4ST/data/prepped_xen"; EMB = "/path/to/systema4ST/results/emb_xen"; OUTD = "/path/to/systema4ST/results/xen_inductive"
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--tower", default="hibou_l"); ap.add_argument("--ngene", type=int, default=200); a = ap.parse_args()
A_ = ad.read_h5ad(f"{PREP}/{a.name}_bin16.h5ad"); px = float(A_.uns["px_per_um"]); xy = np.asarray(A_.obsm["pxl"], np.float64) / px
Y_all = np.log1p(np.asarray(sparse.csr_matrix(A_.X).todense(), np.float32)); X = np.nan_to_num(np.load(f"{EMB}/emb_{a.tower}_{a.name}.npy").astype(np.float32))
gidx = np.argsort(-Y_all.var(0))[:a.ngene]; Y = Y_all[:, gidx]; n = len(xy)
def folds(xy, grid=16):
    qx = np.quantile(xy[:, 0], np.linspace(0, 1, grid + 1)); qx[-1] += 1; qy = np.quantile(xy[:, 1], np.linspace(0, 1, grid + 1)); qy[-1] += 1
    return (np.searchsorted(qx, xy[:, 0], "right") - 1) * grid + (np.searchsorted(qy, xy[:, 1], "right") - 1)
fold = folds(xy); W = build_operator(xy)
P_ridge = block_cv_predict(X, Y, xy); ok = np.isfinite(P_ridge).all(1)
# 上下文特征
tree = cKDTree(xy); d1, i1 = tree.query(xy, k=9); m1 = d1[:, 1:] <= 29.0
d2, i2 = tree.query(xy, k=25); m2 = (d2[:, 1:] > 29.0) & (d2[:, 1:] <= 48.0)
def ring_mean(mask, idx):
    out = np.zeros_like(X); cnt = mask.sum(1)
    for j in range(idx.shape[1] - 1):
        sel = mask[:, j]; out[sel] += X[idx[sel, j + 1]]
    out[cnt > 0] /= cnt[cnt > 0, None]; out[cnt == 0] = X[cnt == 0]; return out
Xc = np.concatenate([X, ring_mean(m1, i1), ring_mean(m2, i2)], 1)
P_ctx = block_cv_predict(Xc, Y, xy)
# 归纳式分区预测器（每折重拟合）与 transductive 对照
pc_all = PCA(n_components=50, random_state=0).fit_transform(X); lab_all = KMeans(n_clusters=20, n_init=4, random_state=0).fit_predict(pc_all)
P_ind = np.full_like(Y, np.nan); P_trd = np.full_like(Y, np.nan); unseen = 0
for f in np.unique(fold):
    te = fold == f
    if te.sum() < 20 or (~te).sum() < 2000: continue
    tr = ~te
    pca = PCA(n_components=50, random_state=0).fit(X[tr]); km = KMeans(n_clusters=20, n_init=4, random_state=0).fit(pca.transform(X[tr]))
    ltr = km.labels_; lte = km.predict(pca.transform(X[te])); mu = np.stack([Y[tr][ltr == u].mean(0) if (ltr == u).any() else Y[tr].mean(0) for u in range(20)])
    P_ind[te] = mu[lte]
    g = Y[tr].mean(0); mu2 = {u: Y[tr & (lab_all == u)].mean(0) for u in np.unique(lab_all[te]) if (tr & (lab_all == u)).any()}
    unseen += sum(1 for u in np.unique(lab_all[te]) if u not in mu2); P_trd[te] = np.stack([mu2.get(u, g) for u in lab_all[te]])
okk = ok & np.isfinite(P_ind).all(1) & np.isfinite(P_ctx).all(1) & np.isfinite(P_trd).all(1)
band1 = lambda M: M - (W @ M); BT = band1(Y)
def sc(Pm): Pf = np.nan_to_num(Pm); return dict(pcc=float(np.nanmean(per_gene_pcc(Pf[okk], Y[okk]))), beta1=float(np.nanmean(per_gene_pcc(band1(Pf)[okk], BT[okk]))))
O = np.zeros_like(Y)
for u in np.unique(lab_all): O[lab_all == u] = Y[lab_all == u].mean(0)
res = dict(name=a.name, n=int(n), n_ok=int(okk.sum()), ridge=sc(P_ridge), context_ridge=sc(P_ctx), inductive_partition=sc(P_ind), transductive_partition=sc(P_trd), oracle=sc(O), unseen_clusters=int(unseen))
for k in ("context_ridge", "inductive_partition", "transductive_partition", "oracle"):
    r, m_ = res["ridge"], res[k]; res[k]["d_scalar"] = (r["pcc"] - m_["pcc"]) / r["pcc"]; res[k]["d_fine"] = (r["beta1"] - m_["beta1"]) / r["beta1"]
os.makedirs(OUTD, exist_ok=True); json.dump(res, open(f"{OUTD}/{a.name}.json", "w"), indent=1)
print(f"[{a.name}] ridge {res['ridge']['pcc']:.3f}/{res['ridge']['beta1']:.3f} | context {res['context_ridge']['pcc']:.3f}/{res['context_ridge']['beta1']:.3f} | inductive {res['inductive_partition']['pcc']:.3f}/{res['inductive_partition']['beta1']:.3f} | transductive {res['transductive_partition']['pcc']:.3f}/{res['transductive_partition']['beta1']:.3f} | oracle {res['oracle']['pcc']:.3f}/{res['oracle']['beta1']:.3f}", flush=True)

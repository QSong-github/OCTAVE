# -*- coding: utf-8 -*-
"""可加的 PCC 分解（复核意见：78:22 不是可加归因）。对中心化后的每个基因，
ρ(ŷ,y) = ⟨Pŷ,Py⟩/(‖ŷ‖‖y‖) + ⟨(I−P)ŷ,(I−P)y⟩/(‖ŷ‖‖y‖)，P 为分区投影（正交，交叉项为零）。
同时统计 8-NN/29 µm 图的孤立结点数。"""
import argparse, json, os, sys, numpy as np, anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
sys.path.insert(0, "/path/to/systema4ST/src")
from per_gene_xen import block_cv_predict, build_operator
PREP = "/path/to/systema4ST/data/prepped_xen"; EMB = "/path/to/systema4ST/results/emb_xen"; OUTD = "/path/to/systema4ST/results/xen_addsplit"
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--tower", default="hibou_l"); ap.add_argument("--ngene", type=int, default=200); a = ap.parse_args()
A_ = ad.read_h5ad(f"{PREP}/{a.name}_bin16.h5ad"); px = float(A_.uns["px_per_um"]); xy = np.asarray(A_.obsm["pxl"], np.float64) / px
Y_all = np.log1p(np.asarray(sparse.csr_matrix(A_.X).todense(), np.float32)); X = np.nan_to_num(np.load(f"{EMB}/emb_{a.tower}_{a.name}.npy").astype(np.float32))
gidx = np.argsort(-Y_all.var(0))[:a.ngene]; Y = Y_all[:, gidx]; n = len(xy)
P = block_cv_predict(X, Y, xy); ok = np.isfinite(P).all(1)
pc = PCA(n_components=50, random_state=0).fit_transform(X); lab = KMeans(n_clusters=20, n_init=4, random_state=0).fit_predict(pc)
Yo, Po, lo = Y[ok], P[ok], lab[ok]
def proj(M, l):
    out = np.zeros_like(M)
    for u in np.unique(l): m = l == u; out[m] = M[m].mean(0)
    return out
Yc = Yo - Yo.mean(0); Pc = Po - Po.mean(0)              # 中心化（P 保持常数，⟨1, ·⟩ 项抵消）
PY_, PP_ = proj(Yc, lo), proj(Pc, lo); WY, WP = Yc - PY_, Pc - PP_
den = np.linalg.norm(Pc, axis=0) * np.linalg.norm(Yc, axis=0) + 1e-12
between = (PP_ * PY_).sum(0) / den; within = (WP * WY).sum(0) / den; total = (Pc * Yc).sum(0) / den
res = dict(name=a.name, n=int(n), n_ok=int(ok.sum()), pcc=float(np.nanmean(total)), between_term=float(np.nanmean(between)), within_term=float(np.nanmean(within)),
           check_max_abs_err=float(np.max(np.abs(total - between - within))), between_share=float(np.nanmean(between) / np.nanmean(total)), within_share=float(np.nanmean(within) / np.nanmean(total)))
# 孤立结点
d, idx = cKDTree(xy).query(xy, k=9); m = d[:, 1:] <= 29.0; deg = m.sum(1); res["isolated_nodes"] = int((deg == 0).sum()); res["min_degree"] = int(deg.min())
os.makedirs(OUTD, exist_ok=True); json.dump(res, open(f"{OUTD}/{a.name}.json", "w"), indent=1)
print(f"[{a.name}] PCC {res['pcc']:.3f} = between {res['between_term']:.3f} + within {res['within_term']:.3f} (误差 {res['check_max_abs_err']:.1e}) → 份额 {100*res['between_share']:.0f}:{100*res['within_share']:.0f} | 孤立结点 {res['isolated_nodes']}", flush=True)

# -*- coding: utf-8 -*-
"""第四份审稿意见的四项补充分析（Xenium，16 区域）：
(1) 模型侧分解：Π_P Ŷ（模型自己的域均值，不用测试标签）与 (I−Π_P)Ŷ 的保真度，标量与最细带；
(2) 交叉拟合 oracle：域均值来自计数的二项一半 A，在另一半 B 上评估（噪声不共享），标量与最细带的 Δ 比值；
(3) 仅训练块估计域均值的预测器的最细带 β1 与 Δ 比值；
(4) 图像分区的空间连通性：邻边标签一致率、每簇连通分量数、最大分量占比（与坐标分区对照）；
(5) 替代算子：对称归一化惰性游走、高斯空间核；最细带 Δ_fine/Δ_scalar。"""
import argparse, json, os, sys, numpy as np, anndata as ad
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
sys.path.insert(0, "/blue/qsong1/wang.qing/systema4ST/src")
from per_gene_xen import build_operator, per_gene_pcc, block_cv_predict
PREP = "/blue/qsong1/wang.qing/systema4ST/data/prepped_xen"; EMB = "/blue/qsong1/wang.qing/systema4ST/results/emb_xen"; OUTD = "/blue/qsong1/wang.qing/systema4ST/results/xen_attrib"
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--tower", default="hibou_l"); ap.add_argument("--ngene", type=int, default=200); a = ap.parse_args()
A_ = ad.read_h5ad(f"{PREP}/{a.name}_bin16.h5ad"); px = float(A_.uns["px_per_um"]); xy = np.asarray(A_.obsm["pxl"], np.float64) / px
C = sparse.csr_matrix(A_.X); Y_all = np.log1p(np.asarray(C.todense(), np.float32)); X = np.nan_to_num(np.load(f"{EMB}/emb_{a.tower}_{a.name}.npy").astype(np.float32))
gidx = np.argsort(-Y_all.var(0))[:a.ngene]; Y = Y_all[:, gidx]; Cg = np.rint(np.asarray(C[:, gidx].todense())).astype(np.int64); n = len(xy)
W = build_operator(xy); P = block_cv_predict(X, Y, xy); ok = np.isfinite(P).all(1); Pf = np.nan_to_num(P)
def gm(M, lab, mask=None):
    out = np.zeros_like(M); idx = np.arange(len(lab)) if mask is None else np.where(mask)[0]
    for u in np.unique(lab[idx]): m = idx[lab[idx] == u]; out[m] = M[m].mean(0)
    return out
def s(Pm, T): return float(np.nanmean(per_gene_pcc(Pm[ok], T[ok])))
def band1(M, Wm=W): return M - (Wm @ M)
def b1(Pm, T, Wm=W): return float(np.nanmean(per_gene_pcc(band1(np.nan_to_num(Pm), Wm)[ok], band1(T, Wm)[ok])))
pc = PCA(n_components=50, random_state=0).fit_transform(X); lab = KMeans(n_clusters=20, n_init=4, random_state=0).fit_predict(pc)
O = gm(Y, lab); res = {"name": a.name, "n": int(n), "n_ok": int(ok.sum())}
# (1) 模型侧分解（在 ok bin 上取域均值；非 ok 置 0 后滤波，与主流水线一致）
Pb = gm(Pf, lab, ok); Pw = Pf - Pb; Yb = gm(Y, lab, ok); Yw = Y - Yb
res["model_decomp"] = {"pcc_model": s(Pf, Y), "pcc_model_between_only": s(Pb, Y), "pcc_between_vs_between": s(Pb, Yb), "pcc_within_vs_within": s(Pw, Yw),
                       "pcc_oracle": s(O, Y), "beta1_model": b1(Pf, Y), "beta1_model_between_only": b1(Pb, Y), "beta1_within_vs_within": b1(Pw, Yw), "beta1_oracle": b1(O, Y),
                       "var_share_within_model": float(np.mean(Pw[ok].var(0) / (Pf[ok].var(0) + 1e-12))), "var_share_within_truth": float(np.mean(Yw[ok].var(0) / (Y[ok].var(0) + 1e-12)))}
# (2) 交叉拟合 oracle（两次拆半取平均）
rng = np.random.default_rng(0); cf = []
for r in range(2):
    H = rng.binomial(Cg, 0.5); Ya = np.log1p(H.astype(np.float32)); Ybh = np.log1p((Cg - H).astype(np.float32))
    Ocf = gm(Ya, lab); Osame = gm(Ybh, lab)
    pr, po, ps_ = s(Pf, Ybh), s(Ocf, Ybh), s(Osame, Ybh); br, bo, bs = b1(Pf, Ybh), b1(Ocf, Ybh), b1(Osame, Ybh)
    cf.append(dict(pcc_ridge=pr, pcc_oracle_cf=po, pcc_oracle_same=ps_, beta1_ridge=br, beta1_oracle_cf=bo, beta1_oracle_same=bs,
                   d_scalar_cf=(pr - po) / pr, d_fine_cf=(br - bo) / br, d_scalar_same=(pr - ps_) / pr, d_fine_same=(br - bs) / br))
res["crossfit"] = {k: float(np.mean([c[k] for c in cf])) for k in cf[0]}; res["crossfit"]["ratio_cf"] = res["crossfit"]["d_fine_cf"] / res["crossfit"]["d_scalar_cf"]; res["crossfit"]["ratio_same"] = res["crossfit"]["d_fine_same"] / res["crossfit"]["d_scalar_same"]
# (3) 仅训练块域均值预测器
def folds(xy, grid=16):
    qx = np.quantile(xy[:, 0], np.linspace(0, 1, grid + 1)); qx[-1] += 1; qy = np.quantile(xy[:, 1], np.linspace(0, 1, grid + 1)); qy[-1] += 1
    return (np.searchsorted(qx, xy[:, 0], "right") - 1) * grid + (np.searchsorted(qy, xy[:, 1], "right") - 1)
fold = folds(xy); T = np.zeros_like(Y); seen = np.zeros(n, bool)
for f in np.unique(fold):
    te = fold == f
    if te.sum() < 20 or (~te).sum() < 2000: continue
    g = Y[~te].mean(0); mu = {u: Y[(~te) & (lab == u)].mean(0) for u in np.unique(lab[te]) if ((~te) & (lab == u)).any()}
    T[te] = np.stack([mu.get(u, g) for u in lab[te]]); seen[te] = True
pt, bt = s(T, Y), b1(T, Y); pm, bm = res["model_decomp"]["pcc_model"], res["model_decomp"]["beta1_model"]
res["trainonly"] = {"pcc": pt, "beta1": bt, "d_scalar": (pm - pt) / pm, "d_fine": (bm - bt) / bm, "ratio": ((bm - bt) / bm) / ((pm - pt) / pm), "coverage": float(seen.mean())}
# (4) 连通性
d, idx = cKDTree(xy).query(xy, k=9); d, idx = d[:, 1:], idx[:, 1:]; m = d <= 29.0
rows = np.repeat(np.arange(n), 8)[m.ravel()]; cols = idx.ravel()[m.ravel()]; Aadj = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)); Aadj = Aadj.maximum(Aadj.T)
lab_xy = KMeans(n_clusters=20, n_init=4, random_state=0).fit_predict(xy)
def contig(l):
    agree = float(np.mean(l[Aadj.tocoo().row] == l[Aadj.tocoo().col])); comps, largest = [], []
    for u in np.unique(l):
        ii = np.where(l == u)[0]; sub = Aadj[ii][:, ii]; k, cl = connected_components(sub, directed=False); comps.append(k); largest.append(float(np.bincount(cl).max() / len(ii)))
    return {"neighbour_agreement": agree, "components_per_cluster_median": float(np.median(comps)), "components_per_cluster_max": int(max(comps)), "largest_component_share_median": float(np.median(largest))}
res["contiguity"] = {"image_partition": contig(lab), "coordinate_partition": contig(lab_xy), "random_partition": contig(rng.permutation(lab))}
# (5) 替代算子
deg = np.asarray(Aadj.sum(1)).ravel(); deg[deg == 0] = 1.0; Dm = sparse.diags(1.0 / np.sqrt(deg))
W_sym = (sparse.identity(n, format="csr") + Dm @ Aadj @ Dm) * 0.5
dg, ig = cKDTree(xy).query(xy, k=25); dg, ig = dg[:, 1:], ig[:, 1:]; wg = np.exp(-dg ** 2 / (2 * 16.0 ** 2)) * (dg <= 48.0)
G = sparse.csr_matrix((wg.ravel(), (np.repeat(np.arange(n), 24), ig.ravel())), shape=(n, n)); G = G.maximum(G.T); dgs = np.asarray(G.sum(1)).ravel(); dgs[dgs == 0] = 1.0
W_g = (sparse.identity(n, format="csr") + sparse.diags(1.0 / dgs) @ G) * 0.5
res["operators"] = {}
for nm, Wm in [("lazy_rw_default", W), ("symmetric_normalised", W_sym.astype(np.float32)), ("gaussian_16um", W_g.astype(np.float32))]:
    bmo, boo = b1(Pf, Y, Wm), b1(O, Y, Wm); ds = (pm - res["model_decomp"]["pcc_oracle"]) / pm; df = (bmo - boo) / bmo
    res["operators"][nm] = {"beta1_model": bmo, "beta1_oracle": boo, "d_fine": df, "ratio": df / ds}
os.makedirs(OUTD, exist_ok=True); json.dump(res, open(f"{OUTD}/{a.name}.json", "w"), indent=1)
md = res["model_decomp"]; print(f"[{a.name}] 模型 {md['pcc_model']:.3f} | 模型自身域均值 {md['pcc_model_between_only']:.3f} | 域内对域内 {md['pcc_within_vs_within']:.3f} | oracle {md['pcc_oracle']:.3f} || 交叉拟合比值 {res['crossfit']['ratio_cf']:.2f}（同半 {res['crossfit']['ratio_same']:.2f}）|| 训练块 β1 {bt:.3f} 比值 {res['trainonly']['ratio']:.2f} || 邻边一致率 图像 {res['contiguity']['image_partition']['neighbour_agreement']:.2f} 坐标 {res['contiguity']['coordinate_partition']['neighbour_agreement']:.2f} 随机 {res['contiguity']['random_partition']['neighbour_agreement']:.2f} || 算子比值 " + " ".join(f"{k[:9]} {v['ratio']:.2f}" for k, v in res["operators"].items()), flush=True)

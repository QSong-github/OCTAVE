# -*- coding: utf-8 -*-
"""评测 Xenium 上的官方 iStar 超分输出（half 折）并与同折的 ridge / 上下文 ridge / 域 oracle 比标量 PCC 与最细带 β1。
采样链复刻 src/istar_eval.py：he-raw → he.jpg(×scale) → 超分网格(//rescale_factor)，3×3 邻域均值（win=1）。
所有预测器在同一批测试 bin、同一 200 基因上评测；β1 用 per_gene_xen.build_operator 在测试 bin 坐标上构图（k=8, 29 µm）。"""
import argparse, glob, json, os, pickle, sys, numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/path/to/systema4ST/src")
from per_gene_xen import build_operator, per_gene_pcc
import anndata as ad
from scipy import sparse
PREP = "/path/to/systema4ST/data/prepped_xen"; EMB = "/path/to/systema4ST/results/emb_xen"; RUN = "/path/to/systema4ST/istar_run"; OUTD = "/path/to/systema4ST/results/istar_xen"
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--win", type=int, default=1); a = ap.parse_args()
d = f"{RUN}/xen_{a.name}_half/"; t = np.load(d + "test.npz", allow_pickle=True)
pxl_raw = t["pxl_raw"].astype(np.float64); xt = t["xy_um"].astype(np.float64); Yte = t["truth"].astype(np.float32); genes = [str(g) for g in t["eval_genes"]]; te_idx = t["test_idx"]; tr_idx = t["train_idx"]
scale = float(open(d + "pixel-size-raw.txt").read()) / float(open(d + "pixel-size.txt").read()); W_he, H_he = Image.open(d + "he.jpg").size
sup = {os.path.basename(p)[:-7]: p for p in glob.glob(d + "cnts-super/*.pickle")}; any_arr = pickle.load(open(next(iter(sup.values())), "rb")); Hg, Wg = np.asarray(any_arr).shape[:2]
rf_i, rf_j = H_he // Hg, W_he // Wg; gi = np.floor(pxl_raw[:, 1] * scale / rf_i).astype(int); gj = np.floor(pxl_raw[:, 0] * scale / rf_j).astype(int); inb = (gi >= 0) & (gi < Hg) & (gj >= 0) & (gj < Wg)
def sample(arr, w=a.win):
    v = np.full(len(gi), np.nan, np.float32)
    for k in np.where(inb)[0]:
        patch = arr[max(gi[k]-w, 0):gi[k]+w+1, max(gj[k]-w, 0):gj[k]+w+1]; v[k] = np.nanmean(patch) if np.isfinite(patch).any() else np.nan
    return v
P_istar = np.full_like(Yte, np.nan); miss = []
for k, g in enumerate(genes):
    if g in sup: P_istar[:, k] = sample(np.asarray(pickle.load(open(sup[g], "rb")), np.float32))
    else: miss.append(g)
ok = np.isfinite(P_istar).all(1)
# 同折的 ridge / 上下文 ridge / oracle
A_ = ad.read_h5ad(f"{PREP}/{a.name}_bin16.h5ad"); ppu = float(A_.uns["px_per_um"]); xy = np.asarray(A_.obsm["pxl"], np.float64) / ppu
Y_all = np.log1p(np.asarray(sparse.csr_matrix(A_.X).todense(), np.float32)); gn = np.asarray(A_.var_names.astype(str)); gcol = [int(np.where(gn == g)[0][0]) for g in genes]; Y = Y_all[:, gcol]
X = np.nan_to_num(np.load(f"{EMB}/emb_hibou_l_{a.name}.npy").astype(np.float32))
def ridge_fit(Xtr, Ytr, Xte, alpha=1e4):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8; return Ridge(alpha=alpha).fit((Xtr - mu) / sd, Ytr).predict((Xte - mu) / sd).astype(np.float32)
P_ridge = ridge_fit(X[tr_idx], Y[tr_idx], X[te_idx])
tree = cKDTree(xy); d1, i1 = tree.query(xy, k=9); m1 = d1[:, 1:] <= 29.0; d2, i2 = tree.query(xy, k=25); m2 = (d2[:, 1:] > 29.0) & (d2[:, 1:] <= 48.0)
def ring_mean(mask, idx):
    out = np.zeros_like(X); cnt = mask.sum(1)
    for j in range(idx.shape[1] - 1):
        sel = mask[:, j]; out[sel] += X[idx[sel, j + 1]]
    out[cnt > 0] /= cnt[cnt > 0, None]; out[cnt == 0] = X[cnt == 0]; return out
Xc = np.concatenate([X, ring_mean(m1, i1), ring_mean(m2, i2)], 1); P_ctx = ridge_fit(Xc[tr_idx], Y[tr_idx], Xc[te_idx])
pc = PCA(n_components=50, random_state=0).fit_transform(X); lab = KMeans(n_clusters=20, n_init=4, random_state=0).fit_predict(pc)
O = np.zeros_like(Y)
for u in np.unique(lab): O[lab == u] = Y[lab == u].mean(0)
P_or = O[te_idx]
# 仅训练侧均值（学习版域预测器）
mu = {u: Y[tr_idx][lab[tr_idx] == u].mean(0) for u in np.unique(lab[te_idx]) if (lab[tr_idx] == u).any()}; g_ = Y[tr_idx].mean(0); P_tr = np.stack([mu.get(u, g_) for u in lab[te_idx]])
Wt = build_operator(xt[ok]); band = lambda M: np.asarray(M - Wt @ M, np.float32); BT = band(Yte[ok])
def sc(Pm): Pm = Pm[ok]; return dict(pcc=float(np.nanmean(per_gene_pcc(Pm, Yte[ok]))), beta1=float(np.nanmean(per_gene_pcc(band(Pm), BT))), fine_var_share=float(np.mean(band(Pm).var(0) / (Pm.var(0) + 1e-12))))
res = dict(name=a.name, n_test=int(len(Yte)), n_ok=int(ok.sum()), n_genes=len(genes), n_missing=len(miss), win=a.win, truth_fine_var_share=float(np.mean(BT.var(0) / (Yte[ok].var(0) + 1e-12))),
           istar=sc(P_istar), ridge=sc(P_ridge), context_ridge=sc(P_ctx), oracle=sc(P_or), trainonly=sc(P_tr))
os.makedirs(OUTD, exist_ok=True); json.dump(res, open(f"{OUTD}/{a.name}.json", "w"), indent=1)
print(f"[{a.name}] test {ok.sum()}/{len(Yte)} | " + " | ".join(f"{k} {v['pcc']:.3f}/{v['beta1']:.3f}" for k, v in res.items() if isinstance(v, dict)), flush=True)

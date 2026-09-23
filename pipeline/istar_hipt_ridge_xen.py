# -*- coding: utf-8 -*-
"""骨干匹配对照（Xenium half 折）：在 iStar 自己抽好的 HIPT 特征（cls+sub+rgb）上跑 ridge，训练/测试 bin、基因与 istar_eval_xen.py 完全相同；
报告标量 PCC 与最细带 β1（同算子），用于区分「HIPT 骨干 vs Hibou-L 骨干」与「iStar 方法本身」两项。采样链 import 自 src/istar_matched.py。"""
import argparse, glob, json, os, pickle, sys, numpy as np, pandas as pd
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/path/to/project/src")
from istar_matched import grid_ij, sample_stack, load_pkl
from per_gene_xen import build_operator, per_gene_pcc
RUN = "/path/to/project/istar_run"; OUTD = "/path/to/project/results/istar_xen_hipt"
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--win", type=int, default=1); ap.add_argument("--pca", type=int, default=256); a = ap.parse_args()
d = f"{RUN}/xen_{a.name}_half/"
scale = float(open(d + "pixel-size-raw.txt").read()) / float(open(d + "pixel-size.txt").read()); W_he, H_he = Image.open(d + "he.jpg").size
sup = sorted(glob.glob(d + "cnts-super/*.pickle")); Hg, Wg = np.asarray(load_pkl(sup[0])).shape[:2]; rf_i, rf_j = H_he // Hg, W_he // Wg
embs = load_pkl(d + "embeddings-hist.pickle"); E = np.concatenate([np.asarray(embs["cls"]), np.asarray(embs["sub"]), np.asarray(embs["rgb"])]).astype(np.float32)
cnts = pd.read_csv(d + "cnts.tsv", sep="\t", index_col=0); locs = pd.read_csv(d + "locs-raw.tsv", sep="\t", index_col=0); ids = [i for i in cnts.index if i in locs.index]; cnts = cnts.loc[ids]; locs = locs.loc[ids]
gi, gj, inb = grid_ij(locs[["x", "y"]].to_numpy(np.float64), scale, rf_i, rf_j, Hg, Wg); Xtr = sample_stack(E, gi, gj, inb, a.win); ok_tr = np.isfinite(Xtr).all(1); Xtr = Xtr[ok_tr]; Ytr = cnts.to_numpy(np.float32)[ok_tr]
t = np.load(d + "test.npz", allow_pickle=True); pxl_te = t["pxl_raw"].astype(np.float64); Yte = t["truth"].astype(np.float32); xt = t["xy_um"].astype(np.float64); genes = [str(g) for g in t["eval_genes"]]
gi2, gj2, inb2 = grid_ij(pxl_te, scale, rf_i, rf_j, Hg, Wg); Xte = sample_stack(E, gi2, gj2, inb2, a.win); ok = np.isfinite(Xte).all(1)
col = {g: i for i, g in enumerate(cnts.columns.astype(str))}; kc = np.array([col[g] for g in genes]); Ytr = Ytr[:, kc]
pipe = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=min(a.pca, Xtr.shape[1], Xtr.shape[0] - 1), random_state=0))]).fit(Xtr); Ztr, Zte = pipe.transform(Xtr), pipe.transform(Xte[ok])
res = {"name": a.name, "n_train": int(ok_tr.sum()), "n_test_ok": int(ok.sum()), "n_test": int(len(Yte)), "hipt_dim": int(E.shape[0])}
W = build_operator(xt[ok]); band = lambda M: np.asarray(M - W @ M, np.float32); BT = band(Yte[ok])
for lab, alpha in [("ridge_hipt_official", 100.0 / (Ztr.shape[1] * Ytr.shape[1])), ("ridge_hipt_1e4", 1e4)]:
    P = Ridge(alpha=alpha).fit(Ztr, Ytr).predict(Zte).astype(np.float32)
    res[lab] = dict(pcc=float(np.nanmean(per_gene_pcc(P, Yte[ok]))), beta1=float(np.nanmean(per_gene_pcc(band(P), BT))), alpha=float(alpha))
os.makedirs(OUTD, exist_ok=True); json.dump(res, open(f"{OUTD}/{a.name}.json", "w"), indent=1)
print(f"[{a.name}] HIPT dim {E.shape[0]} train {ok_tr.sum()} test {ok.sum()}/{len(Yte)} | ridge_hipt(official α) {res['ridge_hipt_official']['pcc']:.3f}/{res['ridge_hipt_official']['beta1']:.3f} | ridge_hipt(1e4) {res['ridge_hipt_1e4']['pcc']:.3f}/{res['ridge_hipt_1e4']['beta1']:.3f}", flush=True)

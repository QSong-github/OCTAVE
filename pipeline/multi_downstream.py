# -*- coding: utf-8 -*-
"""复核意见 #5：OCTAVE 的最细带分数是否比标量更能预测真实编码器的下游效用。
对每个编码器（ridge 头，同 16×16 块 CV，同 50 HVG）算：标量 PCC、最细带 β1，以及 oracle_downstream.readouts 的六个下游读数
（SVG top-k Jaccard、SVG 秩相关、热点 Jaccard、边界位移 µm、共定位保持、热点召回选择性）。读数函数一律 import，不重写。"""
import argparse, json, os, sys, numpy as np, anndata as ad
from scipy import sparse
sys.path.insert(0, "/path/to/project/src")
from downstream2 import knn_graph, block_cv_predict
from oracle_downstream import readouts
from per_gene_xen import build_operator, per_gene_pcc
PREP = "/path/to/project/data/prepped_xen"; EMB = "/path/to/project/results/emb_xen"; OUTD = "/path/to/project/results/multi_ds"
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--tower", required=True); ap.add_argument("--hvg", type=int, default=50); a = ap.parse_args()
A = ad.read_h5ad(f"{PREP}/{a.name}_bin16.h5ad"); xy = np.asarray(A.obsm["pxl"], np.float64) / float(A.uns["px_per_um"])
Yall = np.log1p(np.asarray(sparse.csr_matrix(A.X).todense(), np.float32)); X = np.nan_to_num(np.load(f"{EMB}/emb_{a.tower}_{a.name}.npy").astype(np.float32))
gidx = np.argsort(-Yall.var(0))[:a.hvg]; Yt = Yall[:, gidx]
P = block_cv_predict(X, Yt, xy); ok = np.isfinite(P).all(1); xo, Yo, Po = xy[ok], Yt[ok], P[ok]
W = build_operator(xo); band = lambda M: np.asarray(M - W @ M, np.float32)
sc = dict(pcc=float(np.nanmean(per_gene_pcc(Po, Yo))), beta1=float(np.nanmean(per_gene_pcc(band(Po), band(Yo)))))
nbo, _ = knn_graph(xo); R = readouts(Yo, Po, xo, nbo, a.hvg)
out = dict(name=a.name, tower=a.tower, n_ok=int(ok.sum()), scores=sc, readouts=R)
os.makedirs(OUTD, exist_ok=True); json.dump(out, open(f"{OUTD}/{a.tower}_{a.name}.json", "w"), indent=1)
print(f"[{a.tower}/{a.name}] PCC {sc['pcc']:.3f} β1 {sc['beta1']:.3f} | " + " ".join(f"{k[:14]} {v:.3f}" for k, v in R.items() if v is not None), flush=True)

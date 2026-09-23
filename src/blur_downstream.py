#!/usr/bin/env python
"""第四步·第二次尝试：在同一批 bin 上直接模糊已训模型的预测。

避开前两次的硬伤：
  · 不用块预言机作对照 —— 它取实测均值，在依赖区域均值的读数上天然占便宜。
  · 不动分箱 —— 调细分箱同时改变预测分辨率与训练目标噪声，两者反向，轴不干净。
这里训练数据、真值、评测栅格全部不变，唯一变量是预测被 W^t 平滑的程度。

可判性：模糊让 PCC 掉一点、让 β₁ 掉很多。若下游随 β₁ 走，说明标量低估了
下游可见的损失；若随 PCC 走，说明 β₁ 夸大。两种结果都照报。

对照组：t=0 即未模糊的原预测。所有量对 t=0 归一，比较下降轨迹。
"""
import argparse, json, os, sys
import numpy as np
import anndata as ad
from scipy import sparse

S = "/path/to/project/src"
sys.path.insert(0, S)
from downstream2 import (gene_names, smooth, knn_graph, morans_i, block_cv_predict,
                         jaccard_topq, boundary_by_gradient, boundary_shift_um,
                         hotspot_recall_by_size, colocal_preserve)
from per_gene_xen import build_operator, per_gene_pcc
from scipy.stats import spearmanr

PREP = "/path/to/project/data/prepped_xen"
EMB = "/path/to/project/results/emb_xen"
RES = "/path/to/project/results"
TS = [0, 1, 2, 4, 8, 16]


def readouts(Yo, Po, xo, nbo, hvg):
    mi_p, mi_t = morans_i(Po, nbo), morans_i(Yo, nbo)
    k = max(5, hvg // 5)
    tt, tp = set(np.argsort(-mi_t)[:k].tolist()), set(np.argsort(-mi_p)[:k].tolist())
    Yos, Pos = smooth(Yo, nbo), smooth(Po, nbo)
    b_t = boundary_by_gradient(Yos, xo, nbo)
    b_p = boundary_by_gradient(Pos, xo, nbo)
    rec = hotspot_recall_by_size(Yo, Po, xo, nbo)
    ks = sorted(rec)
    g = lambda z: z[0] if isinstance(z, (list, tuple)) else z
    sel = float(g(rec[ks[-1]]) - g(rec[ks[0]])) if len(ks) >= 2 else None
    return dict(svg_top_jaccard=float(len(tt & tp) / len(tt | tp)),
                svg_rank_rho=float(spearmanr(mi_t, mi_p).statistic),
                hotspot_jaccard=float(jaccard_topq(Yo, Po)),
                boundary_shift_um=float(boundary_shift_um(b_t, b_p, xo)),
                coloc_preserve=float(colocal_preserve(Yo, Po)),
                hotspot_recall_selectivity=sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or os.path.join(RES, "blur_ds_%s.json" % a.name)

    A = ad.read_h5ad("%s/%s_bin16.h5ad" % (PREP, a.name))
    xy = np.asarray(A.obsm["pxl"], np.float64) / float(A.uns["px_per_um"])
    Yall = np.log1p(np.asarray(sparse.csr_matrix(A.X).todense(), np.float32))
    X = np.nan_to_num(np.load("%s/emb_hibou_l_%s.npy" % (EMB, a.name)).astype(np.float32))
    Yt = Yall[:, np.argsort(-Yall.var(0))[:a.hvg]]
    P0 = block_cv_predict(X, Yt, xy)
    ok = np.isfinite(P0).all(1)
    xo, Yo, Pm = xy[ok], Yt[ok], P0[ok]
    print("[%s] %d bin, 块 CV 覆盖 %d" % (a.name, len(xy), ok.sum()), flush=True)

    W = build_operator(xo)
    band = lambda M: np.asarray(M - W @ M, np.float32)
    BT = band(Yo)
    nbo, _ = knn_graph(xo)

    out = {"name": a.name, "n_ok": int(ok.sum()), "levels": {}}
    cur = Pm.copy()
    t_done = 0
    for t in TS:
        while t_done < t:
            cur = np.asarray(W @ cur, np.float32); t_done += 1
        P = cur
        pcc = float(np.nanmean(per_gene_pcc(P, Yo)))
        b1 = float(np.nanmean(per_gene_pcc(band(P), BT)))
        R = readouts(Yo, P, xo, nbo, a.hvg)
        out["levels"][str(t)] = dict(pcc=pcc, beta1=b1, **R)
        print("  t=%2d  PCC=%.4f  β1=%.4f  | SVGjac=%.3f hotJac=%.3f coloc=%.3f sel=%s shift=%.1f"
              % (t, pcc, b1, R["svg_top_jaccard"], R["hotspot_jaccard"], R["coloc_preserve"],
                 "n/a" if R["hotspot_recall_selectivity"] is None else "%.4f" % R["hotspot_recall_selectivity"],
                 R["boundary_shift_um"]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("-> %s" % a.out, flush=True)


if __name__ == "__main__":
    main()

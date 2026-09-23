#!/usr/bin/env python
"""第四步的决定性检验：标量与 β₁ 对同一对预测器给出差 3.6 倍的判断，下游站哪一边？

分箱那条轴不干净：把分箱调细同时降低预测的分辨率损失、又提高训练目标的计数噪声，
两者反向，所以下游读数在 16 µm 取到最优并不能区分两把尺子。

这里换一条干净的轴：**同一批 bin、同一个真值、同一套评测栅格**，比较
  · 已训模型  P_mod = 冻结特征 + 岭回归
  · 块预言机  P_blk = 图像划分内的实测均值（区域内零结构）
标量说这两者差 Δ_scalar≈22%，β₁ 说差 Δ_fine≈80%。六个下游读数的相对差距
落在哪一边，就是对两把尺子的独立裁决。

读数函数一律 import 自 downstream2.py，不重写；块预言机与 blocks_xen_bands.py 同构造
（PCA(50) → KMeans(K=20, n_init=4, random_state=0) → 域内实测均值）。
"""
import argparse, json, os, sys
import numpy as np
import anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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


def group_means(X, lab):
    out = np.empty_like(X)
    for c in np.unique(lab):
        m = lab == c
        out[m] = X[m].mean(0)
    return out


def readouts(Yo, Po, xo, nbo, hvg):
    mi_p, mi_t = morans_i(Po, nbo), morans_i(Yo, nbo)
    k = max(5, hvg // 5)
    tt, tp = set(np.argsort(-mi_t)[:k].tolist()), set(np.argsort(-mi_p)[:k].tolist())
    Yos, Pos = smooth(Yo, nbo), smooth(Po, nbo)
    b_t = boundary_by_gradient(Yos, xo, nbo)
    b_p = boundary_by_gradient(Pos, xo, nbo)
    rec = hotspot_recall_by_size(Yo, Po, xo, nbo)
    ks = sorted(rec)
    sel = None
    if len(ks) >= 2:
        g = lambda z: z[0] if isinstance(z, (list, tuple)) else z
        sel = float(g(rec[ks[-1]]) - g(rec[ks[0]]))
    return dict(svg_top_jaccard=len(tt & tp) / len(tt | tp),
                svg_rank_rho=float(spearmanr(mi_t, mi_p).statistic),
                hotspot_jaccard=float(jaccard_topq(Yo, Po)),
                boundary_shift_um=float(boundary_shift_um(b_t, b_p, xo)),
                coloc_preserve=float(colocal_preserve(Yo, Po)),
                hotspot_recall_selectivity=sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or os.path.join(RES, "oracle_ds_%s.json" % a.name)

    A = ad.read_h5ad("%s/%s_bin16.h5ad" % (PREP, a.name))
    xy = np.asarray(A.obsm["pxl"], np.float64) / float(A.uns["px_per_um"])
    Yall = np.log1p(np.asarray(sparse.csr_matrix(A.X).todense(), np.float32))
    X = np.nan_to_num(np.load("%s/emb_hibou_l_%s.npy" % (EMB, a.name)).astype(np.float32))
    gidx = np.argsort(-Yall.var(0))[:a.hvg]
    Yt = Yall[:, gidx]
    print("[%s] %d bin x %d 基因" % (a.name, len(xy), Yt.shape[1]), flush=True)

    P_mod = block_cv_predict(X, Yt, xy)
    ok = np.isfinite(P_mod).all(1)
    pc = PCA(n_components=50, random_state=0).fit_transform(X)
    lab = KMeans(n_clusters=a.K, n_init=4, random_state=0).fit_predict(pc)
    P_blk = group_means(Yt, lab)
    xo, Yo = xy[ok], Yt[ok]
    Pm, Pb = P_mod[ok], P_blk[ok]
    print("  块 CV 覆盖 %d/%d" % (ok.sum(), len(ok)), flush=True)

    W = build_operator(xo)
    band = lambda M: np.asarray(M - W @ M, np.float32)
    sc = {}
    for nm, P in (("model", Pm), ("oracle", Pb)):
        sc[nm] = dict(pcc=float(np.nanmean(per_gene_pcc(P, Yo))),
                      beta1=float(np.nanmean(per_gene_pcc(band(P), band(Yo)))))
    d_sc = 100 * (sc["model"]["pcc"] - sc["oracle"]["pcc"]) / sc["model"]["pcc"]
    d_fi = 100 * (sc["model"]["beta1"] - sc["oracle"]["beta1"]) / sc["model"]["beta1"]
    print("  标量落差 %.1f%%   最细带落差 %.1f%%   比值 %.2f" % (d_sc, d_fi, d_fi / d_sc), flush=True)

    nbo, _ = knn_graph(xo)
    R = {nm: readouts(Yo, P, xo, nbo, a.hvg) for nm, P in (("model", Pm), ("oracle", Pb))}
    # 每个读数的相对落差，方向统一为「模型优于预言机为正」
    WORSE_HIGH = {"boundary_shift_um"}      # 越大越差
    gaps = {}
    for k in R["model"]:
        vm, vb = R["model"][k], R["oracle"][k]
        if vm is None or vb is None:
            continue
        if k in WORSE_HIGH:
            base = max(abs(vb), 1e-9)
            gaps[k] = 100 * (vb - vm) / base          # 预言机位移更大 ⇒ 模型更好 ⇒ 正
        else:
            base = max(abs(vm), 1e-9)
            gaps[k] = 100 * (vm - vb) / base
        print("    %-28s 模型 %8.4f  预言机 %8.4f  相对落差 %+7.1f%%"
              % (k, vm, vb, gaps[k]), flush=True)
    out = dict(name=a.name, n_ok=int(ok.sum()), K=a.K, scores=sc,
               delta_scalar=d_sc, delta_fine=d_fi, ratio=d_fi / d_sc,
               readouts=R, readout_gaps=gaps)
    json.dump(out, open(a.out, "w"), indent=1)
    print("-> %s" % a.out, flush=True)


if __name__ == "__main__":
    main()

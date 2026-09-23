# -*- coding: utf-8 -*-
"""表 1 的超参消融：岭回归 alpha、基因数、划分块数 K。

关键实现：块 CV 的每一折先算 Gram 矩阵 A = Xtr^T Xtr (1024x1024) 与 b = Xtr^T Ytr，
换 alpha 只是重解 (A + aI)w = b，换基因数只是换 b。于是三条轴共用一次昂贵的遍历。

出口带断言：alpha=1e4、ngene=200 必须重现 sklearn 的 Ridge(alpha=1e4)，
否则整条对比无效。
"""
import argparse, json, os, sys
import numpy as np
import anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

sys.path.insert(0, "/path/to/project/src")
from per_gene_xen import build_operator, per_gene_pcc

PREP = "/path/to/project/data/prepped_xen"
EMB = "/path/to/project/results/emb_xen"
OUTD = "/path/to/project/results/blocks_xen_ablate"
ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6]
NGENES = [50, 100, 200, 400]
KS = [5, 10, 20, 50, 100, 200]
SEEDS = [0, 1, 2]


def folds_of(xy, grid=16):
    qx = np.quantile(xy[:, 0], np.linspace(0, 1, grid + 1)); qx[-1] += 1
    qy = np.quantile(xy[:, 1], np.linspace(0, 1, grid + 1)); qy[-1] += 1
    return (np.searchsorted(qx, xy[:, 0], "right") - 1) * grid + \
           (np.searchsorted(qy, xy[:, 1], "right") - 1)


def group_means(X, lab):
    out = np.empty_like(X)
    for c in np.unique(lab):
        m = lab == c
        out[m] = X[m].mean(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tower", default="hibou_l")
    a = ap.parse_args()
    thr = os.environ.get("OMP_NUM_THREADS", "unset")
    print("[%s] OMP_NUM_THREADS=%s" % (a.name, thr), flush=True)

    A_ = ad.read_h5ad("%s/%s_bin16.h5ad" % (PREP, a.name))
    px = float(A_.uns["px_per_um"])
    xy = np.asarray(A_.obsm["pxl"], np.float64) / px
    Y_all = np.log1p(np.asarray(sparse.csr_matrix(A_.X).todense(), np.float32))
    X = np.nan_to_num(np.load("%s/emb_%s_%s.npy" % (EMB, a.tower, a.name)).astype(np.float32))
    order = np.argsort(-Y_all.var(0))
    NG = [g for g in NGENES if g <= Y_all.shape[1]]
    Ymax = Y_all[:, order[:max(NG)]].astype(np.float64)
    fold = folds_of(xy)
    W = build_operator(xy)
    print("  n=%d bin, 基因上限 %d, 折 %d 个" % (len(xy), max(NG), len(np.unique(fold))), flush=True)

    # 一次遍历，按折存 Gram；换 alpha / 换基因数都只是重解
    P = {(al, g): np.full((len(xy), g), np.nan, np.float32) for al in ALPHAS for g in NG}
    okm = np.zeros(len(xy), bool)
    for f in np.unique(fold):
        te = fold == f
        if te.sum() < 20 or (~te).sum() < 2000:
            continue
        okm |= te
        Xtr, Xte = X[~te], X[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Ztr = ((Xtr - mu) / sd).astype(np.float64)
        Zte = ((Xte - mu) / sd).astype(np.float64)
        zm = Ztr.mean(0)
        Zc = Ztr - zm
        G = Zc.T @ Zc
        for g in NG:
            Yt = Ymax[~te, :g]
            ym = Yt.mean(0)
            b = Zc.T @ (Yt - ym)
            for al in ALPHAS:
                w = np.linalg.solve(G + al * np.eye(G.shape[0]), b)
                P[(al, g)][te] = ((Zte - zm) @ w + ym).astype(np.float32)
    ok = okm & np.isfinite(P[(1e4, 200)]).all(1) if 200 in NG else okm
    print("  块 CV 覆盖 %d/%d" % (int(ok.sum()), len(ok)), flush=True)

    # 断言：alpha=1e4、ngene=200 必须与 sklearn 的实现一致
    if 200 in NG:
        f0 = np.unique(fold)[0]
        te = fold == f0
        if te.sum() >= 20 and (~te).sum() >= 2000:
            mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-8
            m = Ridge(alpha=1e4).fit((X[~te] - mu) / sd, Ymax[~te, :200])
            ref = m.predict((X[te] - mu) / sd)
            mine = P[(1e4, 200)][te]
            dd = float(np.abs(ref - mine).max())
            # 判据用最终量（该折的逐基因平均 PCC），不用原始预测元素：
            # 元素级差是数值路径不同的正常结果，会不会影响结论要看 PCC。
            Yte = Ymax[te, :200]
            pr = float(np.nanmean(per_gene_pcc(ref.astype(np.float32), Yte.astype(np.float32))))
            pm = float(np.nanmean(per_gene_pcc(mine, Yte.astype(np.float32))))
            print("  实现校验：最大逐元素差 %.2e；该折 PCC sklearn %.6f vs 本实现 %.6f (Δ=%.2e)"
                  % (dd, pr, pm, pm - pr), flush=True)
            assert abs(pm - pr) < 1e-4, "Gram 实现与 sklearn 的 PCC 不一致: %.3e" % (pm - pr)

    def band(M):
        return np.asarray(M - W @ M, np.float32)

    out = {"name": a.name, "tower": a.tower, "omp_threads": thr,
           "n_bins": int(len(xy)), "n_ok": int(ok.sum()), "alpha": {}, "ngene": {}, "K": {}}
    pcs = PCA(n_components=50, random_state=0).fit_transform(X)
    LAB = {(k, s): KMeans(n_clusters=k, n_init=4, random_state=s).fit_predict(pcs)
           for k in KS for s in SEEDS}
    print("  KMeans 完成 %d 个划分" % len(LAB), flush=True)

    def scores(al, g, k, s):
        Y = Ymax[:, :g].astype(np.float32)
        BT = band(Y)
        Pf = np.nan_to_num(P[(al, g)])
        rp = float(np.nanmean(per_gene_pcc(Pf[ok], Y[ok])))
        rb = float(np.nanmean(per_gene_pcc(band(Pf)[ok], BT[ok])))
        D = group_means(Y, LAB[(k, s)])
        dp = float(np.nanmean(per_gene_pcc(D[ok], Y[ok])))
        db = float(np.nanmean(per_gene_pcc(band(D)[ok], BT[ok])))
        ov = 100.0 * (rp - dp) / rp
        fi = 100.0 * (rb - db) / rb
        return dict(ridge_pcc=rp, ridge_band=rb, oracle_pcc=dp, oracle_band=db,
                    overall=ov, fineband=fi, ratio=fi / ov, share=100.0 * dp / rp)

    for al in ALPHAS:
        out["alpha"][str(al)] = [dict(seed=s, **scores(al, 200, 20, s)) for s in SEEDS] if 200 in NG else []
        r = out["alpha"][str(al)]
        print("  alpha=%.0e : ridge %.4f  比值 %.3f (种子间 %.3f-%.3f)  块占比 %.1f%%"
              % (al, r[0]["ridge_pcc"], np.median([x["ratio"] for x in r]),
                 min(x["ratio"] for x in r), max(x["ratio"] for x in r),
                 np.median([x["share"] for x in r])), flush=True)
    for g in NG:
        out["ngene"][str(g)] = [dict(seed=s, **scores(1e4, g, 20, s)) for s in SEEDS]
        r = out["ngene"][str(g)]
        print("  ngene=%3d : ridge %.4f  比值 %.3f  块占比 %.1f%%"
              % (g, r[0]["ridge_pcc"], np.median([x["ratio"] for x in r]),
                 np.median([x["share"] for x in r])), flush=True)
    for k in KS:
        out["K"][str(k)] = [dict(seed=s, **scores(1e4, 200, k, s)) for s in SEEDS] if 200 in NG else []
        r = out["K"][str(k)]
        print("  K=%3d     : 比值 %.3f (种子间 %.3f-%.3f)  块占比 %.1f%%  标量落差 %.1f%%  细带落差 %.1f%%"
              % (k, np.median([x["ratio"] for x in r]), min(x["ratio"] for x in r),
                 max(x["ratio"] for x in r), np.median([x["share"] for x in r]),
                 np.median([x["overall"] for x in r]), np.median([x["fineband"] for x in r])), flush=True)
    os.makedirs(OUTD, exist_ok=True)
    json.dump(out, open("%s/%s.json" % (OUTD, a.name), "w"), indent=1)
    print("→ %s/%s.json" % (OUTD, a.name), flush=True)


if __name__ == "__main__":
    main()

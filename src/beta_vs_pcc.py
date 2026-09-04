#!/usr/bin/env python
"""第四步：用一个**独立于两把尺子本身**的判据，比较 PCC 与 β₁ 谁更贴合下游效用。

设计要点（陷阱在这里）：分箱尺寸同时驱动 PCC、β₁ 和全部下游读数，
若跨所有 (区域, 分箱) 直接求相关，三者都会显著相关，什么也证明不了。
因此一律**在区域内、跨四个分箱**比较两把尺子与下游读数的排序一致性，
再把区域聚到样本层级做配对符号检验。

本脚本只负责补算 β₁：重现 downstream2.py 的粗化预测（同一批 bin、同最近邻映射、
同 50 基因、同块 CV），并断言重算的 PCC 与已冻结的 downstream_*.json 一致——
对不上就说明我重现的不是同一批预测，整个比较无效。
"""
import argparse, json, os, sys
import numpy as np
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree

sys.path.insert(0, "/blue/qsong1/wang.qing/systema4ST/src")
from per_gene_xen import build_operator, per_gene_pcc, block_cv_predict, gene_names

PREP = "/blue/qsong1/wang.qing/systema4ST/data/prepped_xen"
EMB = "/blue/qsong1/wang.qing/systema4ST/results/emb_xen"
RES = "/blue/qsong1/wang.qing/systema4ST/results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or os.path.join(RES, "beta_ds_%s.json" % a.name)

    a16 = ad.read_h5ad("%s/%s_bin16.h5ad" % (PREP, a.name))
    xy = np.asarray(a16.obsm["pxl"], np.float64) / float(a16.uns["px_per_um"])
    Y16 = np.log1p(np.asarray(sparse.csr_matrix(a16.X).todense(), np.float32))
    gidx = np.argsort(-Y16.var(0))[:a.hvg]
    Yt = Y16[:, gidx]
    want = gene_names(a16)[gidx]
    print("[%s] %d bin x %d 基因" % (a.name, len(xy), Yt.shape[1]), flush=True)

    frozen = {}
    fp = os.path.join(RES, "downstream_%s.json" % a.name)
    if os.path.exists(fp):
        frozen = {k: v.get("pcc") for k, v in json.load(open(fp))["scales"].items()}

    out = {"name": a.name, "n_bin": int(len(xy)), "scales": {}}
    for b in (8, 16, 32, 64):
        f = "%s/%s_bin%d.h5ad" % (PREP, a.name, b)
        e = "%s/emb_hibou_l_%s%s.npy" % (EMB, a.name, "" if b == 16 else "_bin%d" % b)
        if not (os.path.exists(f) and os.path.exists(e)):
            print("  bin%d: 缺文件" % b, flush=True); continue
        ab = ad.read_h5ad(f)
        xyb = np.asarray(ab.obsm["pxl"], np.float64) / float(ab.uns["px_per_um"])
        Yb = np.log1p(np.asarray(sparse.csr_matrix(ab.X).todense(), np.float32))
        Xb = np.nan_to_num(np.load(e).astype(np.float32))
        gn_b = list(gene_names(ab))
        if Xb.shape[0] != Yb.shape[0] or any(g not in gn_b for g in want):
            print("  bin%d: 形状或基因不符" % b, flush=True); continue
        Pb = block_cv_predict(Xb, Yb[:, [gn_b.index(g) for g in want]], xyb)
        _, nn = cKDTree(xyb).query(xy, k=1)
        P = Pb[nn]
        ok = np.isfinite(P).all(1)
        if ok.sum() < 0.5 * len(xy):
            print("  bin%d: 覆盖不足" % b, flush=True); continue
        xo, Yo, Po = xy[ok], Yt[ok], P[ok]
        W = build_operator(xo)
        band = lambda M: np.asarray(M - W @ M, np.float32)
        pcc = float(np.nanmean(per_gene_pcc(Po, Yo)))
        beta1 = float(np.nanmean(per_gene_pcc(band(Po), band(Yo))))
        fz = frozen.get(str(b))
        d = None if fz is None else pcc - fz
        out["scales"][str(b)] = dict(pcc=pcc, beta1=beta1, n_ok=int(ok.sum()),
                                     frozen_pcc=fz, pcc_delta=d)
        print("  bin%2d  n_ok=%6d  PCC=%.6f  β1=%.6f   与冻结值 delta=%s"
              % (b, ok.sum(), pcc, beta1, "n/a" if d is None else "%+.2e" % d), flush=True)
        if d is not None and abs(d) > 1e-4:
            raise SystemExit("bin%d 的 PCC 与冻结值差 %.3e，说明重现的不是同一批预测，中止" % (b, d))
    json.dump(out, open(a.out, "w"), indent=1)
    print("-> %s" % a.out, flush=True)


if __name__ == "__main__":
    main()

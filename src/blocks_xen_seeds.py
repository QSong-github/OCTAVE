# -*- coding: utf-8 -*-
"""块预言机的划分是 KMeans 的一个局部最优。本脚本换种子重复，给出表 1 的误差棒。

确定的部分（算子、岭回归块 CV、真值带）只算一次；每个种子只重算
KMeans → 域均值 → 分数。seed 0 用与 blocks_xen_bands.py 相同的线程数跑，
应当重现冻结结果；对不上就报出来，不许掩盖。
"""
import argparse, json, os, sys
import numpy as np
import anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, "/path/to/systema4ST/src")
from per_gene_xen import build_operator, per_gene_pcc, block_cv_predict

PREP = "/path/to/systema4ST/data/prepped_xen"
EMB = "/path/to/systema4ST/results/emb_xen"
FROZEN = "/path/to/systema4ST/results/blocks_xen_bands"
OUTD = "/path/to/systema4ST/results/blocks_xen_seeds"


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
    ap.add_argument("--ngene", type=int, default=200)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--nseed", type=int, default=16)
    a = ap.parse_args()
    thr = os.environ.get("OMP_NUM_THREADS", "unset")
    print("[%s] OMP_NUM_THREADS=%s" % (a.name, thr), flush=True)

    A_ = ad.read_h5ad("%s/%s_bin16.h5ad" % (PREP, a.name))
    px = float(A_.uns["px_per_um"])
    xy = np.asarray(A_.obsm["pxl"], np.float64) / px
    Y_all = np.log1p(np.asarray(sparse.csr_matrix(A_.X).todense(), np.float32))
    X = np.nan_to_num(np.load("%s/emb_%s_%s.npy" % (EMB, a.tower, a.name)).astype(np.float32))
    assert X.shape[0] == Y_all.shape[0]
    gidx = np.argsort(-Y_all.var(0))[:a.ngene]
    Y = Y_all[:, gidx]
    print("  n=%d bin, %d 基因" % (len(xy), len(gidx)), flush=True)

    W = build_operator(xy)
    P = block_cv_predict(X, Y, xy)
    ok = np.isfinite(P).all(1)

    def band(M):
        return np.asarray(M - W @ M, np.float32)

    BT = band(Y)
    Pf = np.nan_to_num(P)
    r_pcc = float(np.nanmean(per_gene_pcc(Pf[ok], Y[ok])))
    r_band = float(np.nanmean(per_gene_pcc(band(Pf)[ok], BT[ok])))
    print("  ridge（确定）PCC %.6f  细带 %.6f" % (r_pcc, r_band), flush=True)

    pc = PCA(n_components=50, random_state=0).fit_transform(X)
    rows, lab0 = [], None
    for s in range(a.nseed):
        km = KMeans(n_clusters=a.k, n_init=4, random_state=s).fit(pc)
        lab = km.labels_
        if lab0 is None:
            lab0 = lab
        D = group_means(Y, lab)
        d_pcc = float(np.nanmean(per_gene_pcc(D[ok], Y[ok])))
        d_band = float(np.nanmean(per_gene_pcc(band(D)[ok], BT[ok])))
        ov = 100.0 * (r_pcc - d_pcc) / r_pcc
        fi = 100.0 * (r_band - d_band) / r_band
        ari = float(adjusted_rand_score(lab0, lab))
        rows.append(dict(seed=s, dom_pcc=d_pcc, dom_band_pcc=d_band, overall=ov,
                         fineband=fi, ratio=fi / ov, share=100.0 * d_pcc / r_pcc,
                         ari_vs_seed0=ari, inertia=float(km.inertia_),
                         n_clusters_used=int(len(np.unique(lab)))))
        print("  seed %2d: PCC %.6f 标量落差 %5.2f%% 细带落差 %5.2f%% 比值 %5.3f ARI(vs0) %.3f"
              % (s, d_pcc, ov, fi, fi / ov, ari), flush=True)

    chk = None
    fp = "%s/%s.json" % (FROZEN, a.name)
    if os.path.exists(fp):
        F = json.load(open(fp))["pred"]
        fz = F["dom20"]["pcc"]
        s0 = rows[0]["dom_pcc"]
        rz = F["ridge"]["pcc"]
        chk = dict(frozen_dom20_pcc=fz, seed0_dom20_pcc=s0, delta=s0 - fz,
                   frozen_ridge_pcc=rz, ridge_delta=r_pcc - rz,
                   reproduces=bool(abs(s0 - fz) < 1e-9))
        tag = "一致" if chk["reproduces"] else "不一致（多半是线程数不同，见 CLAIMS B7）"
        print("\n  锚点：seed 0 与冻结值 delta=%+.3e；岭回归 delta=%+.3e  %s"
              % (s0 - fz, r_pcc - rz, tag), flush=True)

    v = np.array([x["ratio"] for x in rows])
    ar = [x["ari_vs_seed0"] for x in rows[1:]]
    summ = dict(n_seeds=len(rows), ratio_median=float(np.median(v)), ratio_min=float(v.min()),
                ratio_max=float(v.max()), ratio_sd=float(v.std(ddof=1)),
                ratio_q25=float(np.percentile(v, 25)), ratio_q75=float(np.percentile(v, 75)),
                overall_median=float(np.median([x["overall"] for x in rows])),
                overall_sd=float(np.std([x["overall"] for x in rows], ddof=1)),
                fineband_median=float(np.median([x["fineband"] for x in rows])),
                fineband_sd=float(np.std([x["fineband"] for x in rows], ddof=1)),
                share_median=float(np.median([x["share"] for x in rows])),
                share_sd=float(np.std([x["share"] for x in rows], ddof=1)),
                ari_median=float(np.median(ar)) if ar else None,
                ari_min=float(np.min(ar)) if ar else None,
                fineband_gt_overall=int(sum(1 for x in rows if x["fineband"] > x["overall"])))
    print("\n  %d 个种子：比值中位 %.3f [%.3f, %.3f] sd %.3f；细带>标量 %d/%d；划分间 ARI 中位 %s"
          % (len(rows), summ["ratio_median"], summ["ratio_min"], summ["ratio_max"],
             summ["ratio_sd"], summ["fineband_gt_overall"], len(rows),
             ("%.3f" % summ["ari_median"]) if summ["ari_median"] is not None else "n/a"), flush=True)
    os.makedirs(OUTD, exist_ok=True)
    json.dump(dict(name=a.name, k=a.k, tower=a.tower, n_bins=int(len(xy)), n_ok=int(ok.sum()),
                   n_genes=int(len(gidx)), omp_threads=thr, ridge_pcc=r_pcc,
                   ridge_band_pcc=r_band, frozen_check=chk, seeds=rows, summary=summ),
              open("%s/%s.json" % (OUTD, a.name), "w"), indent=1)
    print("→ %s/%s.json" % (OUTD, a.name), flush=True)


if __name__ == "__main__":
    main()

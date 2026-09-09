# -*- coding: utf-8 -*-
"""块识别上界，跑在 Xenium 16 区域 / 8 独立样本上。

存在的理由：§39 的「纯块预测器拿到训练模型 88–94% 的分数」只有 Visium HD
2 张切片的 4 个折。n=2 时精确符号检验的下限就是 0.5 —— 永远不可能显著。
本脚本把同一构造搬到 16 个 Xenium 区域上，于是可在**样本层级**做检验
（n=8 → 最小可得 P=0.0078；与 Fig 3 口径一致的 7 样本子集 → 0.0156）。

协议与 per_gene_xen.py / downstream.py 完全一致：片内 16×16 空间块 CV、
固定 16 µm 栅格、top-200 HVG、Ridge(alpha=1e4)。

可检验的命题（每区域一个数）：
    rel_gap_overall  = (pcc_ridge - pcc_blocks) / pcc_ridge
    rel_gap_fineband = (band_ridge - band_blocks) / band_ridge
「标量低估了两者在细尺度上的差别」⇔ rel_gap_fineband > rel_gap_overall。
注意不测「σ_blocks > σ_ridge」：一折内 σ 由 PCC 单调决定，那等于测「ridge 赢」，无信息。
"""
import argparse, json, os, sys
import numpy as np
import anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sys.path.insert(0, "/path/to/systema4ST/src")
from per_gene_xen import (build_operator, calibrate_sigma, per_gene_pcc,
                          block_cv_predict, gene_names)

PREP = "/path/to/systema4ST/data/prepped_xen"
EMB = "/path/to/systema4ST/results/emb_xen"
OUTD = "/path/to/systema4ST/results/blocks_xen"


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
    ap.add_argument("--tmax", type=int, default=2048)
    a = ap.parse_args()
    cps = [1]
    while cps[-1] < a.tmax:
        cps.append(cps[-1] * 2)

    A_ = ad.read_h5ad(f"{PREP}/{a.name}_bin16.h5ad")
    px = float(A_.uns["px_per_um"])
    xy = np.asarray(A_.obsm["pxl"], np.float64) / px
    Y_all = np.log1p(np.asarray(sparse.csr_matrix(A_.X).todense(), np.float32))
    X = np.nan_to_num(np.load(f"{EMB}/emb_{a.tower}_{a.name}.npy").astype(np.float32))
    assert X.shape[0] == Y_all.shape[0]
    gidx = np.argsort(-Y_all.var(0))[:a.ngene]
    Y = Y_all[:, gidx]
    print(f"[{a.name}] n={len(xy)} bin, {len(gidx)} 基因", flush=True)

    W = build_operator(xy)
    sig = calibrate_sigma(W, xy, cps)
    print(f"  σ: {sig[cps[0]]:.1f} → {sig[cps[-1]]:.0f} µm", flush=True)

    P = block_cv_predict(X, Y, xy)
    ok = np.isfinite(P).all(1)
    print(f"  块 CV 覆盖 {int(ok.sum())}/{len(ok)}", flush=True)

    # A_coarse 阶梯：全图扩散（物理正确），在 ok bin 上评估（与方法 PCC 同口径）
    ladder, cur, t = {}, Y.copy(), 0
    S1 = None
    for cp in cps:
        while t < cp:
            cur = W @ cur; t += 1
        if cp == 1:
            S1 = cur.copy()
        ladder[cp] = float(np.nanmean(per_gene_pcc(cur[ok], Y[ok])))
    print("  阶梯完成", flush=True)

    # 域 oracle：图像嵌入 PCA(50) → KMeans → 取**实测**域均值（域内零结构）
    pc = PCA(n_components=50, random_state=0).fit_transform(X)
    DOM = {}
    for k in (20, 200):
        lab = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(pc)
        DOM[k] = group_means(Y, lab)
        print(f"  KMeans k={k} 完成", flush=True)

    def eqA(v):
        pts = sorted(((sig[c], ladder[c]) for c in cps), key=lambda z: z[0])
        if not np.isfinite(v):
            return None, "nan"
        if v >= pts[0][1]:
            return float(pts[0][0]), "left"
        for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
            if v0 >= v >= v1:
                w = (v0 - v) / max(v0 - v1, 1e-12)
                return float(np.exp(np.log(s0) + w * (np.log(s1) - np.log(s0)))), "ok"
        return None, "right"

    # 最细带：B = F - W F，在 ok bin 上评估（W 为全图算子，物理一致）
    def band(M):
        return np.asarray(M - W @ M, np.float32)
    BT = band(Y)
    preds = {"ridge": P, "dom20": DOM[20], "dom200": DOM[200]}
    R = {}
    for nm, M in preds.items():
        Mf = np.nan_to_num(M)
        p = float(np.nanmean(per_gene_pcc(Mf[ok], Y[ok])))
        e, fl = eqA(p)
        BM = band(Mf)
        b = float(np.nanmean(per_gene_pcc(BM[ok], BT[ok])))
        vs = float(np.mean(BM[ok].var(0) / (Mf[ok].var(0) + 1e-12)))
        R[nm] = {"pcc": p, "sigma_um": e, "sigma_flag": fl, "band_pcc": b,
                 "fine_var_share": vs}
        print(f"  {nm:<8s} PCC {p:.4f}  σ {('%.1f' % e) if e else 'n/a'}{fl[:1]}  "
              f"带 PCC {b:.4f}  细带方差占比 {vs:.4f}", flush=True)
    R["truth"] = {"fine_var_share": float(np.mean(BT[ok].var(0) / (Y[ok].var(0) + 1e-12)))}

    out = {"name": a.name, "n_bins": int(len(xy)), "n_ok": int(ok.sum()),
           "n_genes": int(len(gidx)), "tower": a.tower,
           "sigma_um": {str(c): sig[c] for c in cps},
           "ladder": {str(c): ladder[c] for c in cps}, "pred": R}
    for k in (20, 200):
        nm = f"dom{k}"
        out.setdefault("rel_gap", {})[nm] = {
            "overall": (R["ridge"]["pcc"] - R[nm]["pcc"]) / R["ridge"]["pcc"],
            "fineband": (R["ridge"]["band_pcc"] - R[nm]["band_pcc"]) / R["ridge"]["band_pcc"],
            "score_ratio": R[nm]["pcc"] / R["ridge"]["pcc"],
            "sigma_ratio": (R[nm]["sigma_um"] / R["ridge"]["sigma_um"])
            if (R[nm]["sigma_um"] and R["ridge"]["sigma_um"]) else None}
        g = out["rel_gap"][nm]
        print(f"  → {nm}: 整体相对差距 {100*g['overall']:.1f}%  "
              f"细带相对差距 {100*g['fineband']:.1f}%  "
              f"(比 {g['fineband']/g['overall']:.2f}×)" if g["overall"] else "", flush=True)
    os.makedirs(OUTD, exist_ok=True)
    json.dump(out, open(f"{OUTD}/{a.name}.json", "w"), indent=1)
    print(f"→ {OUTD}/{a.name}.json", flush=True)


if __name__ == "__main__":
    main()

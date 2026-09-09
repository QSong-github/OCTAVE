# -*- coding: utf-8 -*-
"""逐基因分解，跑在 Xenium 15 区域 / 7 样本队列上。

存在的理由：`per_gene.py` 只跑了 Visium HD 的 2 张切片，于是全篇最强的那条论证
——「per-gene PCC 被基因自身的 Moran's I 解释掉 70%，而等效分辨率免疫」——
只有 n=2 的描述性证据，做不了任何检验。

本脚本在与 Fig 3/Fig 6 完全相同的协议下（片内 16×16 空间块 CV、固定 16 µm 栅格）
逐片输出每个基因的 {pcc, eq, moran, moran_pred, ceil, mean, zero}，
于是可以逐片算 r(PCC, Moran) 与 r(log eq, Moran)，再在**样本层级**做符号检验。

注意：15 个区域来自 7 个独立样本（乳腺 S1–S4 各含 Top/Mid/Bot），
本脚本只负责逐片产出，样本层级聚合交给下游分析，避免在这里就把伪重复固化。

σ 用估计量 A（阶梯匹配），与 per_gene.py 一致；**不得**与估计量 B 的跨度并列比较。
"""
import argparse, json, os, sys
import numpy as np
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PREP = "/path/to/systema4ST/data/prepped_xen"
EMB = "/path/to/systema4ST/results/emb_xen"
OUTD = "/path/to/systema4ST/results/per_gene_xen"


def gene_names(a):
    for c in ("gene", "gene_name", "feature_name", "symbol"):
        if c in a.var.columns:
            return np.asarray(a.var[c]).astype(str)
    return np.asarray(a.var_names).astype(str)


def build_operator(xy_um, k=8, cut_um=29.0, lazy=0.5):
    """惰性随机游走 W = lazy·I + (1-lazy)·D^-1 A（默认 1/2），邻域半径按 16 µm 栅格设定（同 effres.py）。"""
    n = len(xy_um)
    d, idx = cKDTree(xy_um).query(xy_um, k=k + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    m = d <= cut_um
    rows = np.repeat(np.arange(n), k)[m.ravel()]
    cols = idx.ravel()[m.ravel()]
    A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1.0
    P = sparse.diags(1.0 / deg) @ A
    return (lazy * sparse.identity(n, format="csr", dtype=np.float32) + (1.0 - lazy) * P).astype(np.float32)


def calibrate_sigma(W, xy_um, cps, n_seed=128, seed=0):
    """δ 种子扩散的二阶矩 → 每个检查点的实测 σ（µm）。"""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    seeds = rng.choice(n, size=min(n_seed, n), replace=False)
    S = np.zeros((n, len(seeds)), np.float32)
    S[seeds, np.arange(len(seeds))] = 1.0
    out, t = {}, 0
    for cp in cps:
        while t < cp:
            S = W @ S
            t += 1
        v = []
        for j, s0 in enumerate(seeds):
            w = S[:, j]
            m = w > 1e-8
            if m.sum() < 3:
                continue
            ww = w[m] / w[m].sum()
            dd = xy_um[m] - xy_um[s0]
            v.append(float((ww * (dd ** 2).sum(1)).sum()))
        out[cp] = float(np.sqrt(np.mean(v) / 2.0)) if v else np.nan
    return out


def per_gene_pcc(A, B):
    Ac = A - A.mean(0); Bc = B - B.mean(0)
    num = (Ac * Bc).sum(0)
    den = np.sqrt((Ac ** 2).sum(0) * (Bc ** 2).sum(0)) + 1e-12
    return num / den


def morans_pergene(Adj, X):
    Xc = X - X.mean(0)
    return (X.shape[0] / Adj.sum()) * (Xc * (Adj @ Xc)).sum(0) / ((Xc ** 2).sum(0) + 1e-12)


def block_cv_predict(X, Y, xy, grid=16):
    """与 downstream.py 完全同协议的片内空间块 CV —— 口径必须一致，否则不能与 Fig 3/6 并读。"""
    qx = np.quantile(xy[:, 0], np.linspace(0, 1, grid + 1)); qx[-1] += 1
    qy = np.quantile(xy[:, 1], np.linspace(0, 1, grid + 1)); qy[-1] += 1
    fold = (np.searchsorted(qx, xy[:, 0], "right") - 1) * grid + \
           (np.searchsorted(qy, xy[:, 1], "right") - 1)
    P = np.full_like(Y, np.nan)
    for f in np.unique(fold):
        te = fold == f
        if te.sum() < 20 or (~te).sum() < 2000:
            continue
        mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-8
        m = Ridge(alpha=1e4).fit((X[~te] - mu) / sd, Y[~te])
        P[te] = m.predict((X[te] - mu) / sd).astype(np.float32)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tower", default="hibou_l")
    ap.add_argument("--ngene", type=int, default=200)
    ap.add_argument("--tmax", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=3)
    a_ = ap.parse_args()

    cps = [1]
    while cps[-1] < a_.tmax:
        cps.append(cps[-1] * 2)

    A_ = ad.read_h5ad(f"{PREP}/{a_.name}_bin16.h5ad")
    px = float(A_.uns["px_per_um"])
    xy = np.asarray(A_.obsm["pxl"], np.float64) / px
    C = sparse.csr_matrix(A_.X)                      # 原始计数，拆半用
    Y_all = np.log1p(np.asarray(C.todense(), np.float32))
    gn_all = gene_names(A_)
    X = np.nan_to_num(np.load(f"{EMB}/emb_{a_.tower}_{a_.name}.npy").astype(np.float32))
    assert X.shape[0] == Y_all.shape[0], f"嵌入 {X.shape[0]} vs 表达 {Y_all.shape[0]}"

    gidx = np.argsort(-Y_all.var(0))[:a_.ngene]
    Y = Y_all[:, gidx]; gn = gn_all[gidx]
    print(f"[{a_.name}] n={len(xy)} bin, {len(gidx)} 基因", flush=True)

    W = build_operator(xy)
    Adj = (W > 0).astype(np.float32); Adj.setdiag(0); Adj.eliminate_zeros()
    sig = calibrate_sigma(W, xy, cps)
    print(f"  σ 标定: {sig[cps[0]]:.1f} → {sig[cps[-1]]:.0f} µm", flush=True)

    # 真值阶梯：逐基因，用于把每个基因的 PCC 换算成等效 σ
    ladder, cur, t = {}, Y.copy(), 0
    for cp in cps:
        while t < cp:
            cur = W @ cur; t += 1
        ladder[cp] = per_gene_pcc(cur, Y)
    print("  阶梯完成", flush=True)

    P = block_cv_predict(X, Y, xy)
    ok = np.isfinite(P).all(1)
    print(f"  块 CV 覆盖 {ok.sum()}/{len(ok)}", flush=True)
    pcc_g = per_gene_pcc(P[ok], Y[ok])
    moran_g = morans_pergene(Adj, Y)
    moran_p = morans_pergene(Adj, np.nan_to_num(P))

    # 逐基因噪声天花板：二项拆半 + Spearman-Brown
    rng = np.random.default_rng(0)
    Csub = C[:, gidx]
    ch = []
    for _ in range(a_.reps):
        h = rng.binomial(Csub.data.astype(np.int64), 0.5).astype(np.float32)
        H1 = sparse.csr_matrix((h, Csub.indices, Csub.indptr), shape=Csub.shape)
        H2 = sparse.csr_matrix((Csub.data - h, Csub.indices, Csub.indptr), shape=Csub.shape)
        f = lambda M: np.log1p(np.asarray(M.todense(), np.float32))
        ch.append(per_gene_pcc(f(H1), f(H2)))
    ch = np.mean(ch, 0)
    ceil_g = 2 * ch / (1 + ch)

    def eq_of(j, v):
        """该基因的 PCC 落在真值阶梯的哪个 σ 上（log σ 线性插值）。"""
        pts = sorted(((sig[c], ladder[c][j]) for c in cps), key=lambda z: z[0])
        if not np.isfinite(v):
            return np.nan
        if v >= pts[0][1]:
            return pts[0][0]                       # 左删失：优于最细带
        for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
            if v0 >= v >= v1:
                w = (v0 - v) / max(v0 - v1, 1e-12)
                return float(np.exp(np.log(s0) + w * (np.log(s1) - np.log(s0))))
        return np.nan                              # 右删失：比最粗带还差
    eq_g = np.array([eq_of(j, pcc_g[j]) for j in range(len(gn))])

    res = {
        "name": a_.name, "tower": a_.tower, "n_bin": int(len(xy)),
        "n_ok": int(ok.sum()), "sigma_um": {str(c): sig[c] for c in cps},
        "censored_right": int(np.sum(~np.isfinite(eq_g) & np.isfinite(pcc_g))),
        "genes": {str(g): {"pcc": float(pcc_g[j]), "eq": float(eq_g[j]),
                           "moran": float(moran_g[j]), "moran_pred": float(moran_p[j]),
                           "ceil": float(ceil_g[j]),
                           "mean": float(Y[:, j].mean()),
                           "zero": float((Y[:, j] == 0).mean())}
                  for j, g in enumerate(gn)},
    }
    os.makedirs(OUTD, exist_ok=True)
    json.dump(res, open(f"{OUTD}/{a_.name}.json", "w"), indent=1, ensure_ascii=False)

    m = np.isfinite(eq_g) & np.isfinite(moran_g)
    from scipy.stats import pearsonr
    print(f"  r(PCC, Moran)      = {pearsonr(pcc_g, moran_g)[0]:+.3f}  (n={len(gn)})")
    print(f"  r(log eq, Moran)   = {pearsonr(np.log10(eq_g[m]), moran_g[m])[0]:+.3f}  (n={m.sum()})")
    print(f"  右删失基因 {res['censored_right']}/{len(gn)}")
    print(f"已存 {OUTD}/{a_.name}.json")


if __name__ == "__main__":
    main()

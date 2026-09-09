# -*- coding: utf-8 -*-
"""
空间可变基因(SVG)的假发现 —— 把"Moran 膨胀"变成可核查的下游损害。

逐基因分析(38978855)显示: 真实空间信号最弱的基因, 其 Moran's I 被膨胀 5.78×。
"膨胀 N 倍"仍是抽象的指标语言。本脚本直接跑领域实际会跑的分析:
在预测 ST 上做 SVG 检验, 与在真实 ST 上做同一检验对照, 报告经验假发现率。

检验: Moran's I + 行置换零分布(每基因保留自身边际分布), 用置换矩/方差算 z,
      BH 控制 FDR 0.05 —— 与 moran.test(randomisation=TRUE) 同口径。
输出: 真实 SVG 集 vs 预测 SVG 集的 假发现率 / 召回 / 清单长度膨胀 / top-K 重合。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import norm
import evaluate as E, retrieval as R
from baselines import ridge_predict
from effres import build_operator, PX_PER_UM, SLIDES, SEB

ap = argparse.ArgumentParser()
ap.add_argument("--tower", default="hibou_l"); ap.add_argument("--ngene", type=int, default=2000)
ap.add_argument("--nperm", type=int, default=200); ap.add_argument("--fdr", type=float, default=0.05)
a_ = ap.parse_args()

a = ad.read_h5ad(SEB.H5AD)
expr = np.nan_to_num(np.asarray(a.X, np.float32))
slide = a.obs["slide_id"].astype(str).values
pxl = np.asarray(a.obsm["pxl"], np.float64)
img = np.nan_to_num(np.concatenate([
    np.load(os.path.join(SEB.EMBDIR, f"emb_{a_.tower}_P2.npy")),
    np.load(os.path.join(SEB.EMBDIR, f"emb_{a_.tower}_P5.npy"))]).astype(np.float32))

def moran(A, X, s0):
    Xc = X - X.mean(0)
    return (X.shape[0]/s0) * (Xc*(A@Xc)).sum(0) / ((Xc**2).sum(0)+1e-12)

def bh(p, q):
    o = np.argsort(p); m = len(p)
    thr = p[o] <= q*np.arange(1, m+1)/m
    k = np.where(thr)[0]
    out = np.zeros(m, bool)
    if len(k): out[o[:k[-1]+1]] = True
    return out

def svg_call(A, X, s0, rng, nperm):
    I = moran(A, X, s0)
    null = np.empty((nperm, X.shape[1]), np.float32)
    for r in range(nperm):
        null[r] = moran(A, X[rng.permutation(X.shape[0])], s0)
    z = (I - null.mean(0)) / (null.std(0) + 1e-12)
    return I, bh(norm.sf(z), a_.fdr)

res = {}
for s in SLIDES:
    te = slide == s; tr = ~te
    gidx = E.topk_hvg(expr[tr], a_.ngene)
    y = expr[te][:, gidx]; Ytr = expr[tr]
    xy = pxl[te] / PX_PER_UM[s]
    W = build_operator(xy, k=8, cut_um=29.0)
    A = (W > 0).astype(np.float32); A.setdiag(0); A.eliminate_zeros(); s0 = A.sum()
    print(f"\n[{s}] n={int(te.sum())} 基因={len(gidx)} 置换={a_.nperm}", flush=True)

    rng = np.random.default_rng(0)
    I_t, S_t = svg_call(A, y, s0, rng, a_.nperm)
    print(f"  真实 SVG: {S_t.sum()}/{len(gidx)}", flush=True)

    P = {"Ridge_HEST": ridge_predict(img[tr], Ytr, img[te], 1e4)[:, gidx],
         "imageKNN": R.image_floor(img[te], img[tr], Ytr, k=800)[:, gidx]}
    res[s] = {"n_gene": len(gidx), "n_true_svg": int(S_t.sum())}
    for nm, pr in P.items():
        I_p, S_p = svg_call(A, pr, s0, np.random.default_rng(0), a_.nperm)
        tp = int((S_p & S_t).sum()); fp = int((S_p & ~S_t).sum())
        K = int(S_t.sum())
        topK = len(set(np.argsort(-I_p)[:K]) & set(np.argsort(-I_t)[:K])) / max(K, 1)
        res[s][nm] = dict(n_pred_svg=int(S_p.sum()), tp=tp, fp=fp,
                          fdr=fp/max(S_p.sum(), 1), recall=tp/max(S_t.sum(), 1),
                          inflate=float(S_p.sum())/max(S_t.sum(), 1), topK=topK,
                          moran_med=float(np.median(I_p)), moran_med_true=float(np.median(I_t)))
        d = res[s][nm]
        print(f"  {nm:12s} 预测SVG={d['n_pred_svg']:5d} 假发现={fp:5d} "
              f"FDR={d['fdr']:.3f} 召回={d['recall']:.3f} 清单膨胀={d['inflate']:.2f}× "
              f"top-K重合={topK:.3f}", flush=True)

print(f"\n{'='*76}"); print(f"两片合计 (FDR 目标 {a_.fdr})"); print("="*76)
print(f"{'方法':14s}{'真实SVG':>9s}{'预测SVG':>9s}{'假发现':>9s}{'经验FDR':>10s}{'召回':>9s}{'top-K重合':>11s}")
for nm in ("Ridge_HEST", "imageKNN"):
    t = sum(res[s]["n_true_svg"] for s in SLIDES)
    p = sum(res[s][nm]["n_pred_svg"] for s in SLIDES)
    f = sum(res[s][nm]["fp"] for s in SLIDES)
    tp = sum(res[s][nm]["tp"] for s in SLIDES)
    print(f"{nm:14s}{t:>9d}{p:>9d}{f:>9d}{f/max(p,1):>10.3f}{tp/max(t,1):>9.3f}"
          f"{np.mean([res[s][nm]['topK'] for s in SLIDES]):>11.3f}")
print(f"\n⇒ 名义 FDR 控制在 {a_.fdr}, 真实假发现率见上。差距 = 用预测 ST 做 SVG 分析的代价。")
json.dump(res, open("/path/to/systema4ST/results/svg_fdr.json", "w"),
          indent=2, ensure_ascii=False)

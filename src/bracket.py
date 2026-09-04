# -*- coding: utf-8 -*-
"""
三方夹逼 —— 区分"模态极限"与"指标掩盖的差距"。

现有: A_coarse 阶梯(结构 oracle) + 真实方法。缺一个上界: H&E 本身最多支撑多细?
构造【图像 oracle】: 在测试片自己上拟合 Ridge(故意用测试片真值), 得到图像特征
所能表达的极限 —— 任何基于该特征的方法都不可能超过它。

判读(二值, 两种都必须如实报):
  图像 oracle σ ≈ 真实方法 σ  ⇒ 极限在【模态】, H&E 不携带更细信息, 批评指标无意义
  图像 oracle σ ≪ 真实方法 σ  ⇒ 方法离模态极限还很远, 而指标把这个差距藏起来了

另加两个中间档以定位差距来源:
  oracle_cv : 测试片内 5 折交叉(仍是同片, 但不看自己) → 隔离"跨片泛化"这一项
  ridge_xs  : 常规跨片 Ridge(即真实协议)
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import evaluate as E, retrieval as R
from baselines import ridge_predict
from effres import build_operator, calibrate_sigma, PX_PER_UM, SLIDES, SEB

ap = argparse.ArgumentParser()
ap.add_argument("--tower", default="hibou_l")
ap.add_argument("--hvg", type=int, default=50)
ap.add_argument("--tmax", type=int, default=2048)
ap.add_argument("--pca", type=int, default=256)
a_ = ap.parse_args()
cps = [1]
while cps[-1] < a_.tmax: cps.append(cps[-1]*2)

a = ad.read_h5ad(SEB.H5AD)
expr = np.nan_to_num(np.asarray(a.X, np.float32))
slide = a.obs["slide_id"].astype(str).values
pxl = np.asarray(a.obsm["pxl"], np.float64)
img = np.nan_to_num(np.concatenate([
    np.load(os.path.join(SEB.EMBDIR, f"emb_{a_.tower}_P2.npy")),
    np.load(os.path.join(SEB.EMBDIR, f"emb_{a_.tower}_P5.npy"))]).astype(np.float32))

out = {}
for s in SLIDES:
    te = slide == s; tr = ~te
    gidx = E.topk_hvg(expr[tr], a_.hvg)
    y = expr[te][:, gidx]; Xte = img[te]; Xtr = img[tr]; Ytr = expr[tr][:, gidx]
    xy = pxl[te] / PX_PER_UM[s]
    W = build_operator(xy, k=8, cut_um=29.0)
    sig = calibrate_sigma(W, xy, cps)
    print(f"\n[{s}] n={int(te.sum())}", flush=True)

    def fit_pred(Xa, Ya, Xb):
        p = Pipeline([("sc", StandardScaler()),
                      ("pca", PCA(n_components=min(a_.pca, Xa.shape[1], Xa.shape[0]-1),
                                  random_state=0))]).fit(Xa)
        alpha = 100.0 / (min(a_.pca, Xa.shape[1]) * Ya.shape[1])   # HEST 官方 alpha 规则
        return Ridge(solver="lsqr", alpha=alpha, fit_intercept=False,
                     max_iter=1000).fit(p.transform(Xa), Ya).predict(p.transform(Xb))

    preds = {}
    preds["oracle_self"] = fit_pred(Xte, y, Xte)              # 上界: 在自己身上拟合
    cvp = np.empty_like(y)                                     # 中间: 片内 5 折
    for a_i, b_i in KFold(5, shuffle=True, random_state=0).split(Xte):
        cvp[b_i] = fit_pred(Xte[a_i], y[a_i], Xte[b_i])
    preds["oracle_cv"] = cvp
    preds["ridge_xs"] = fit_pred(Xtr, Ytr, Xte)               # 真实协议: 跨片
    preds["imageKNN"] = R.image_floor(Xte, Xtr, expr[tr], k=800)[:, gidx]

    ladder, cur, t = {}, y.copy(), 0
    for cp in cps:
        while t < cp: cur = W @ cur; t += 1
        ladder[cp] = float(E.per_gene_pcc(cur, y).mean())

    def eqs(v):
        pts = sorted(((sig[c], ladder[c]) for c in cps), key=lambda z: z[0])
        if v >= pts[0][1]: return pts[0][0]
        for (s0,v0),(s1,v1) in zip(pts, pts[1:]):
            if v0 >= v >= v1: return s0 + (v0-v)/max(v0-v1,1e-12)*(s1-s0)
        return float("nan")

    rec = {"sigma_um": {str(c): sig[c] for c in cps}, "ladder": {str(c): ladder[c] for c in cps}}
    for nm, p in preds.items():
        v = float(E.per_gene_pcc(p, y).mean())
        rec[nm] = {"pcc": v, "eq_sigma": eqs(v)}
        print(f"  {nm:12s} pcc={v:.4f}  等价σ={rec[nm]['eq_sigma']:.0f}µm", flush=True)
    out[s] = rec

print(f"\n=== 三方夹逼 (塔={a_.tower}, 两片平均) ===")
print(f"{'':14s}{'PCC':>9s}{'等价σ(µm)':>12s}   含义")
meaning = {"oracle_self": "图像特征的表达上界(用测试片真值拟合)",
           "oracle_cv":   "片内 5 折(不看自己, 但同片)",
           "ridge_xs":    "真实协议: 跨片泛化",
           "imageKNN":    "纯形态学下界"}
for nm in ("oracle_self", "oracle_cv", "ridge_xs", "imageKNN"):
    p = np.mean([out[s][nm]["pcc"] for s in SLIDES])
    q = np.nanmean([out[s][nm]["eq_sigma"] for s in SLIDES])
    print(f"{nm:14s}{p:>9.4f}{q:>12.0f}   {meaning[nm]}")
os.makedirs("results", exist_ok=True)
json.dump(out, open(f"results/bracket_{a_.tower}.json","w"), indent=2, ensure_ascii=False)
print(f"\n判读: oracle_self 与 ridge_xs 的差距 = 指标掩盖的可改进空间;"
      f"\n      oracle_self 本身的 σ = H&E 这个模态的分辨率天花板。")

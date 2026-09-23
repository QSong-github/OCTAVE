# -*- coding: utf-8 -*-
"""
Visium HD 线的下游指标 —— 与 HEST 广度线口径对齐。

hest_downstream.py 已在 72 个 HEST 样本(100µm 栅格)上做了 pcc/ret@1/dom1/ari/Moran,
但深度线(Visium HD, 16µm)还只有 pcc。两条线指标不对齐就没法互相印证。本脚本补齐。

Visium HD 能做而 HEST 做不到的: 最细频带到 ~10µm, 所以 Moran's I 膨胀和检索准确率
可以一直测到接近单细胞尺度。

方法只用两个合法锚点(imageKNN 纯形态学下界 / Ridge = HEST 参照协议), 不含任何
上游项目自研方法, 也不含把第三方编码器塞进检索管线的消融。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import evaluate as E, retrieval as R
from baselines import ridge_predict
from effres import build_operator, calibrate_sigma, PX_PER_UM, SLIDES, SEB

MAXRET = 4000


def zs(A):
    A = A - A.mean(0); s = A.std(0); s[s < 1e-8] = 1.0
    return A / s


def ret_at_k(pred, truth, seed=0, ks=(1, 10)):
    rng = np.random.default_rng(seed)
    idx = rng.choice(pred.shape[0], min(MAXRET, pred.shape[0]), replace=False)
    P, T = zs(pred[idx]), zs(truth[idx])
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-8
    S = P @ T.T
    rank = (S > np.diag(S)[:, None]).sum(1)
    return {f"ret@{k}": float((rank < k).mean()) for k in ks}, len(idx)


def dom_at1(pred, truth, lab):
    us = np.unique(lab)
    C = np.stack([truth[lab == u].mean(0) for u in us])
    d = np.empty((pred.shape[0], len(us)), np.float32)
    for j in range(len(us)):
        d[:, j] = ((pred - C[j]) ** 2).sum(1)
    return float((us[d.argmin(1)] == lab).mean())


def morans_I(A, X):
    Xc = X - X.mean(0)
    return float(np.mean((X.shape[0] / A.sum()) * (Xc * (A @ Xc)).sum(0) / ((Xc ** 2).sum(0) + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", default="hibou_l")
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--k", type=int, default=800)
    ap.add_argument("--tmax", type=int, default=2048)
    ap.add_argument("--k_dom", type=int, default=10)
    ap.add_argument("--out", default="results/vhd_downstream.json")
    a_ = ap.parse_args()
    cps = [1]
    while cps[-1] < a_.tmax:
        cps.append(cps[-1] * 2)

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
        Ytr, y = expr[tr], expr[te][:, gidx]
        xy = pxl[te] / PX_PER_UM[s]
        print(f"\n[{s}] test={int(te.sum())}", flush=True)
        W = build_operator(xy, k=8, cut_um=29.0)   # effres 版只返回 W(与 hest_ladder 版签名不同)
        A = (W > 0).astype(np.float32); A.setdiag(0); A.eliminate_zeros()
        sig = calibrate_sigma(W, xy, cps)
        lab = KMeans(n_clusters=a_.k_dom, n_init=4, random_state=0).fit_predict(zs(y))
        I_true = morans_I(A, y)

        def metrics(P):
            r, n = ret_at_k(P, y)
            lp = KMeans(n_clusters=a_.k_dom, n_init=4, random_state=0).fit_predict(zs(P))
            return dict(pcc=float(E.per_gene_pcc(P, y).mean()), **r, dom1=dom_at1(P, y, lab),
                        ari=float(adjusted_rand_score(lab, lp)), moran=morans_I(A, P), n_ret=n)

        preds = {"imageKNN": R.image_floor(img[te], img[tr], Ytr, k=a_.k)[:, gidx],
                 "Ridge_HEST": ridge_predict(img[tr], Ytr, img[te], 1e4)[:, gidx]}
        M = {nm: metrics(p) for nm, p in preds.items()}
        for nm, m in M.items():
            print(f"  {nm:12s} pcc={m['pcc']:.3f} ret@1={m['ret@1']:.4f}(chance {1/m['n_ret']:.1e}) "
                  f"dom1={m['dom1']:.3f} ari={m['ari']:.3f} Moran 真={I_true:.3f} 预测={m['moran']:.3f}",
                  flush=True)
        ladder, cur, t = {}, y.copy(), 0
        for cp in cps:
            while t < cp:
                cur = W @ cur; t += 1
            ladder[str(cp)] = metrics(cur)
            print(f"  A_coarse σ≈{sig[cp]:>4.0f}µm  pcc={ladder[str(cp)]['pcc']:.3f} "
                  f"ret@1={ladder[str(cp)]['ret@1']:.4f} ari={ladder[str(cp)]['ari']:.3f} "
                  f"Moran={ladder[str(cp)]['moran']:.3f}", flush=True)
        out[s] = dict(sigma_um={str(c): sig[c] for c in cps}, moran_true=I_true,
                      methods=M, ladder=ladder, chance_dom=1.0 / a_.k_dom)

    os.makedirs("results", exist_ok=True)
    json.dump(out, open(a_.out, "w"), indent=2, ensure_ascii=False)

    def eqs(d, mname, key):
        v = d["methods"][mname][key]
        pts = sorted(((d["sigma_um"][c], d["ladder"][c][key]) for c in d["ladder"]), key=lambda t: t[0])
        if v >= pts[0][1]:
            return pts[0][0]
        for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
            if v0 >= v >= v1:
                return s0 + (v0 - v) / max(v0 - v1, 1e-12) * (s1 - s0)
        return np.nan

    print(f"\n=== Visium HD: 四个指标各自给出的等价 σ (µm, 两片平均) ===")
    print(f"{'方法':12s}" + "".join(f"{k:>10s}" for k in ("pcc", "ret@1", "dom1", "ari")))
    for nm in ("Ridge_HEST", "imageKNN"):
        row = [np.nanmean([eqs(out[s], nm, k) for s in SLIDES]) for k in ("pcc", "ret@1", "dom1", "ari")]
        print(f"{nm:12s}" + "".join(f"{v:>10.0f}" for v in row))
    print(f"\n=== Moran's I 膨胀 ===")
    for s in SLIDES:
        d = out[s]
        print(f"  {s[-2:]}: 真值={d['moran_true']:.3f} " +
              " ".join(f"{nm}={d['methods'][nm]['moran']:.3f}({d['methods'][nm]['moran']/d['moran_true']:.2f}×)"
                       for nm in d["methods"]))
    print(f"\n已存 {a_.out}")


if __name__ == "__main__":
    main()

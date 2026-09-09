# -*- coding: utf-8 -*-
"""把核心论断做成影像图：预测 ≈ 把实测模糊到 σ。

四格并排（同一块组织）：H&E ｜ 实测表达 ｜ 模型预测 ｜ 实测模糊到 σ
三张表达图用**同一个 PC1 载荷**投影，保证是同一个量、同一个色标，可直接比。

σ 的取法：估计量 A —— 在真值的模糊阶梯上找与该方法 PCC 相等的那一档。
图像只能显示离散档位，故取最接近的检查点并把它的实测 σ 标在图上，不外推。

坐标：he-raw.jpg 像素 = h5ad obsm['pxl'] / 4.0（已实测验证：中位偏差 0.414 px，
100% 在 1 px 内，见 level-downsample.txt = 4.000053）。
"""
import os, sys, json
import numpy as np
import anndata as ad
from scipy.spatial import cKDTree
from scipy import sparse
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, "/path/to/systema4ST/src")

PARENT = "/path/to/spatial2exp/he2st_align"
H5AD = os.path.join(PARENT, "data/binned_16um.h5ad")
EMB = os.path.join(PARENT, "results")
HE = os.path.join(PARENT, "istar_run/P2_checker/he-raw.jpg")
DOWN = 4.000053157559005          # he-raw 像素 = pxl / DOWN
UMPX_HE = 1.095925                # he-raw 每像素微米
PX_PER_UM = {"Visium_HD_Human_Colon_Cancer_P2": 3.6499,
             "Visium_HD_Human_Colon_Cancer_P5": 3.6526}
OUT = "/path/to/systema4ST/figures"
TOWER = "hibou_l"
NG = 200
TMAX = 2048

P = {"blue": "#0F4D92", "red": "#B64342", "grey_m": "#767676",
     "grey_d": "#4D4D4D", "black": "#272727", "grey_l": "#CFCECE"}
MM = 1 / 25.4
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.8,
                     "legend.frameon": False, "axes.titlesize": 7})


def per_gene_pcc(A, B):
    Ac = A - A.mean(0); Bc = B - B.mean(0)
    return (Ac * Bc).sum(0) / (np.sqrt((Ac ** 2).sum(0) * (Bc ** 2).sum(0)) + 1e-12)


def build_W(xy, k=8, cut=29.0):
    n = len(xy)
    d, idx = cKDTree(xy).query(xy, k=k + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    m = d <= cut
    rows = np.repeat(np.arange(n), k)[m.ravel()]; cols = idx.ravel()[m.ravel()]
    A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1.0
    return (sparse.identity(n, format="csr", dtype=np.float32)
            + sparse.diags(1.0 / deg) @ A).astype(np.float32) * 0.5


def calib_sigma(W, xy, cps, n_seed=128, seed=0):
    rng = np.random.default_rng(seed); n = W.shape[0]
    seeds = rng.choice(n, size=min(n_seed, n), replace=False)
    S = np.zeros((n, len(seeds)), np.float32); S[seeds, np.arange(len(seeds))] = 1.0
    out, t = {}, 0
    for cp in cps:
        while t < cp:
            S = W @ S; t += 1
        v = []
        for j, s0 in enumerate(seeds):
            w = S[:, j]; mm = w > 1e-8
            if mm.sum() < 3:
                continue
            ww = w[mm] / w[mm].sum(); dd = xy[mm] - xy[s0]
            v.append(float((ww * (dd ** 2).sum(1)).sum()))
        out[cp] = float(np.sqrt(np.mean(v) / 2.0)) if v else np.nan
    return out


def main():
    cps = [1]
    while cps[-1] < TMAX:
        cps.append(cps[-1] * 2)

    a = ad.read_h5ad(H5AD)
    expr = np.nan_to_num(np.asarray(a.X, np.float32))
    slide = a.obs["slide_id"].astype(str).values
    pxl = np.asarray(a.obsm["pxl"], np.float64)
    img = np.nan_to_num(np.concatenate([
        np.load(os.path.join(EMB, f"emb_{TOWER}_P2.npy")),
        np.load(os.path.join(EMB, f"emb_{TOWER}_P5.npy"))]).astype(np.float32))

    te = slide == "Visium_HD_Human_Colon_Cancer_P2"; tr = ~te
    gidx = np.argsort(-expr[tr].var(0))[:NG]          # 训练片选基因，无泄漏
    Y = expr[te][:, gidx]
    xy = pxl[te] / PX_PER_UM["Visium_HD_Human_Colon_Cancer_P2"]
    print(f"P2 bins={te.sum()}  genes={NG}", flush=True)

    mu, sd = img[tr].mean(0), img[tr].std(0) + 1e-8
    R = Ridge(alpha=1e4).fit((img[tr] - mu) / sd, expr[tr][:, gidx])
    PRED = R.predict((img[te] - mu) / sd).astype(np.float32)
    pcc = float(np.nanmean(per_gene_pcc(PRED, Y)))
    print(f"cross-slide ridge PCC = {pcc:.4f}", flush=True)

    W = build_W(xy)
    sig = calib_sigma(W, xy, cps)
    print("sigma ladder:", {c: round(sig[c], 1) for c in cps}, flush=True)

    # 真值阶梯 + 保留每一档的模糊场，供挑选
    blur, cur, t = {}, Y.copy(), 0
    lad = {}
    for cp in cps:
        while t < cp:
            cur = W @ cur; t += 1
        lad[cp] = float(np.nanmean(per_gene_pcc(cur, Y)))
        blur[cp] = cur.copy()
    print("ladder pcc:", {c: round(lad[c], 3) for c in cps}, flush=True)

    # 找与方法 PCC 最接近的那一档
    best = min(cps, key=lambda c: abs(lad[c] - pcc))
    print(f"matched checkpoint t={best}  sigma={sig[best]:.0f} um  ladder={lad[best]:.4f}",
          flush=True)

    # 同一个 PC1 载荷投影三张场
    Yc = Y - Y.mean(0)
    U, S_, Vt = np.linalg.svd(Yc, full_matrices=False)
    load = Vt[0]
    if (Yc @ load).mean() < 0:
        load = -load
    proj = lambda M: (M - Y.mean(0)) @ load
    F_true, F_pred, F_blur = proj(Y), proj(PRED), proj(blur[best])

    np.savez_compressed(
        "/path/to/systema4ST/results/he_panel_P2.npz",
        xy=xy, F_true=F_true, F_pred=F_pred, F_blur=F_blur,
        pxl=pxl[te], pcc=pcc, sigma=sig[best], t=best,
        ladder_pcc=lad[best],
        sig_all=np.array([sig[c] for c in cps]),
        lad_all=np.array([lad[c] for c in cps]), cps=np.array(cps))
    print("已存 results/he_panel_P2.npz", flush=True)


if __name__ == "__main__":
    main()

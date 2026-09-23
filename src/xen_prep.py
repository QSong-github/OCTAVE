# -*- coding: utf-8 -*-
"""
Xenium → 与 Visium HD 同构的 16µm 分箱 h5ad。

为什么要分箱而不用单细胞：本文的等价 σ 是拿方法得分去插值一把「把真值模糊到 σ」的
阶梯。要和 9 张 Visium HD 可比，必须用同样的 16µm 网格。单细胞层级留作后续把阶梯
往下压的实验（Xenium 的价值正在于此）。

坐标链（每一步都必须对，否则 patch 取错位置而不会报错）：
  细胞质心 (µm)  --÷ pixel_size-->  Xenium 像素  --inv(affine)-->  H&E 像素
其中 affine 来自 <SAMPLE>_he_imagealignment.csv，官方定义是 H&E px → Xenium px，
故取逆。pixel_size 从 experiment.xenium 读，不写死。

内置校验：把分箱投影回 H&E，检查 ① 全部落在图像范围内 ② patch 不是一片空白背景。
两项任一不过就报错退出 —— 宁可不出结果，也不要出一个坐标错位却看不出来的结果。
"""
import os, sys, json, glob, argparse, numpy as np, anndata as ad, scanpy as sc
from scipy import sparse

ROOT = "/path/to/project/data/xenium"
OUT  = "/path/to/project/data/prepped_xen"

def load_affine(d, name):
    f = os.path.join(d, f"{name}_he_imagealignment.csv")
    M = np.loadtxt(f, delimiter=",")
    assert M.shape == (3, 3), f"对齐矩阵形状异常 {M.shape}"
    return M

def prep(name, bin_um=16.0, min_cells=1):
    d = os.path.join(ROOT, name)
    exp = json.load(open(os.path.join(d, "experiment.xenium")))
    px = float(exp.get("pixel_size", 0.2125))
    import pyarrow.parquet as pq
    C = pq.read_table(os.path.join(d, "cells.parquet")).to_pandas()
    A = sc.read_10x_h5(os.path.join(d, "cell_feature_matrix.h5"))
    A.var_names_make_unique()
    cid = C["cell_id"].astype(str).values
    order = {c: i for i, c in enumerate(map(str, A.obs_names))}
    keep = np.array([order[c] for c in cid if c in order])
    ok = np.array([c in order for c in cid])
    C = C[ok].reset_index(drop=True); X = sparse.csr_matrix(A.X)[keep]
    xy = np.stack([C["x_centroid"].to_numpy(), C["y_centroid"].to_numpy()], 1).astype(np.float64)

    bx = np.floor((xy[:, 0] - xy[:, 0].min()) / bin_um).astype(np.int64)
    by = np.floor((xy[:, 1] - xy[:, 1].min()) / bin_um).astype(np.int64)
    key = bx * (by.max() + 2) + by
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    sel = cnt >= min_cells
    Agg = sparse.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uk), len(inv)))
    Xb = (Agg @ X).tocsr()[sel]
    xyb = np.stack([(Agg @ xy[:, 0]) / cnt, (Agg @ xy[:, 1]) / cnt], 1)[sel]   # 成员细胞质心均值
    ncell = cnt[sel]

    # µm → Xenium px → H&E px
    M = load_affine(d, name)
    xen_px = xyb / px
    inv_M = np.linalg.inv(M)
    hom = np.concatenate([xen_px, np.ones((len(xen_px), 1))], 1)
    he = (inv_M @ hom.T).T
    he_px = he[:, :2] / he[:, [2]]

    tif = os.path.join(d, f"{name}_PYRAMIDAL.tif")
    if not os.path.exists(tif): tif = os.path.join(d, f"{name}_he_image.ome.tif")
    import openslide
    S = openslide.OpenSlide(tif); W, H = S.dimensions

    inb = (he_px[:, 0] >= 112) & (he_px[:, 0] < W-112) & (he_px[:, 1] >= 112) & (he_px[:, 1] < H-112)
    frac = inb.mean()
    print(f"[{name}] 细胞={len(xy)} 分箱={len(xyb)} px={px} 图像={W}×{H} 落在图内={frac:.3f}", flush=True)
    assert frac > 0.80, f"{name}: 仅 {frac:.1%} 的分箱落在 H&E 图内 —— 坐标链有误，拒绝输出"

    rng = np.random.default_rng(0)
    s = rng.choice(np.where(inb)[0], min(40, inb.sum()), replace=False)
    mu = [np.asarray(S.read_region((int(x-112), int(y-112)), 0, (224, 224)).convert("RGB")).mean()
          for x, y in he_px[s]]
    blank = float(np.mean(np.array(mu) > 235))
    print(f"[{name}] 抽检 patch 亮度中位={np.median(mu):.1f} 近全白占比={blank:.2f}", flush=True)
    assert blank < 0.5, f"{name}: {blank:.0%} 的 patch 是空白背景 —— 坐标链有误，拒绝输出"

    a = ad.AnnData(X=Xb[inb], var=A.var.copy())
    a.obs["slide_id"] = name; a.obs["n_cells"] = ncell[inb]
    a.obsm["pxl"] = he_px[inb]                      # 与 Visium HD 同名同义：H&E 全分辨率像素
    a.obsm["xy_um"] = xyb[inb]
    a.uns["px_per_um"] = float(1.0 / (px * abs(np.linalg.det(M[:2, :2])) ** 0.5))
    a.uns["bin_um"] = bin_um; a.uns["platform"] = "Xenium"
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{name}_bin{int(bin_um)}.h5ad")
    a.write_h5ad(p)
    print(f"[{name}] ✔ {a.n_obs} bin × {a.n_vars} 基因  px/µm={a.uns['px_per_um']:.4f}  "
          f"细胞/bin 中位={np.median(ncell[inb]):.1f}  已存", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None); ap.add_argument("--bin_um", type=float, default=16.0)
    a_ = ap.parse_args()
    names = [a_.name] if a_.name else sorted(os.listdir(ROOT))
    for n in names:
        if not os.path.isdir(os.path.join(ROOT, n)): continue
        try: prep(n, a_.bin_um)
        except Exception as e: print(f"[{n}] ✘ {type(e).__name__}: {e}", flush=True)

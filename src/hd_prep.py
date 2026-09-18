# -*- coding: utf-8 -*-
"""
把新下载的 Visium HD 样本整理成与 P2/P5 完全同构的 h5ad。

深度线原本只有 P2/P5（同一研究、同一实验室、同一组织类型），最致命的反驳是
「σ≈110µm 可能只是结直肠肿瘤的组织结构尺度」。本脚本接入 7 张新片：
  人 CRC P1（第三个病人）· 人癌旁正常 P3 · 人胰腺 · 鼠脑/肾/胚胎/小肠
覆盖跨病人 / 癌-正常 / 跨器官 / 跨物种四个方向。

输出与母项目 binned_16um.h5ad 同构:
  X          = 原始计数（稀疏）
  obs        = slide_id, array_row, array_col, in_tissue
  obsm['pxl']= [x=pxl_col_in_fullres, y=pxl_row_in_fullres]  ← openslide read_region 的 (x,y) 顺序
  uns        = px_per_um（由 scalefactors 的 microns_per_pixel 取倒数）
坐标一律用 pxl（母项目的 obs['x_um'] 与 coord.npy 被 NaN 均值污染，见 内部记录 第 6 节）。
"""
import os, sys, json, glob, argparse, numpy as np, anndata as ad, scanpy as sc
from scipy import sparse

ROOT = "/path/to/systema4ST/data/visiumhd"
OUT = "/path/to/systema4ST/data/prepped"

def prep(name):
    d = os.path.join(ROOT, name)
    sq = glob.glob(os.path.join(d, "**", "square_016um"), recursive=True)
    assert sq, f"{name}: 找不到 square_016um"
    sq = sq[0]
    h5 = os.path.join(sq, "filtered_feature_bc_matrix.h5")
    a = sc.read_10x_h5(h5)
    a.var_names_make_unique()

    tp = glob.glob(os.path.join(sq, "spatial", "tissue_positions*"))
    assert tp, f"{name}: 找不到 tissue_positions"
    if tp[0].endswith(".parquet"):
        import pyarrow.parquet as pq
        P = pq.read_table(tp[0]).to_pandas()
    else:
        import pandas as pd
        P = pd.read_csv(tp[0])
    P = P.set_index("barcode")
    keep = [b for b in a.obs_names if b in P.index]
    a = a[keep].copy()
    P = P.loc[a.obs_names]

    sf = json.load(open(os.path.join(sq, "spatial", "scalefactors_json.json")))
    mpp = sf.get("microns_per_pixel")
    assert mpp, f"{name}: scalefactors 无 microns_per_pixel: {list(sf)}"

    a.obsm["pxl"] = np.stack([P["pxl_col_in_fullres"].to_numpy(),      # x
                              P["pxl_row_in_fullres"].to_numpy()], 1).astype(np.float64)
    a.obs["array_row"] = P["array_row"].to_numpy()
    a.obs["array_col"] = P["array_col"].to_numpy()
    a.obs["slide_id"] = name
    a.uns["px_per_um"] = float(1.0 / mpp)
    a.uns["microns_per_pixel"] = float(mpp)
    a.X = sparse.csr_matrix(a.X)

    xy = a.obsm["pxl"] / a.uns["px_per_um"]
    span = xy.max(0) - xy.min(0)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{name}_16um.h5ad")
    a.write_h5ad(p)
    print(f"[{name}] n_obs={a.n_obs} n_vars={a.n_vars} px/µm={a.uns['px_per_um']:.4f} "
          f"组织跨度={span[0]/1000:.1f}×{span[1]/1000:.1f} mm  计数中位={np.median(np.asarray(a.X.sum(1))):.0f}", flush=True)
    return p

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", default=None)
    a_ = ap.parse_args()
    names = [a_.name] if a_.name else sorted(os.listdir(ROOT))
    for n in names:
        if not os.path.isdir(os.path.join(ROOT, n)): continue
        try: prep(n)
        except Exception as e: print(f"[{n}] ✘ {type(e).__name__}: {e}", flush=True)

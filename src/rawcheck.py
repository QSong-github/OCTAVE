# -*- coding: utf-8 -*-
"""
诊断: st_bench/data/<slide>/adata_16um.h5ad(全基因原始 counts)能否与 binned_16um.h5ad 对齐。

用途 —— 做【噪声天花板】: 对原始 counts 做二项拆半, 两半各自走同一套带通分解再互相求相关,
得到每个空间尺度上"可达的最高相关" c(σ)。没有它, 最细带的 r=0.177 无法解读
(16µm bin 有 57.7% 零元, 散粒噪声本身就压着上限)。

前提是两个文件的行能对上。本脚本检查: 行数、obs 键、坐标是否一致、基因是否覆盖那 200 个。
只读。
"""
import numpy as np, anndata as ad, h5py, os

P = "/blue/qsong1/wang.qing/spatial2exp/he2st_align"
SLIDES = ["Visium_HD_Human_Colon_Cancer_P2", "Visium_HD_Human_Colon_Cancer_P5"]

b = ad.read_h5ad(os.path.join(P, "data/binned_16um.h5ad"))
bslide = b.obs["slide_id"].astype(str).values
bgene = np.asarray(b.var["gene"]).astype(str)
bpxl = np.asarray(b.obsm["pxl"], np.float64)
print(f"binned_16um: {b.shape}, 每片 " +
      ", ".join(f"{s}={int((bslide==s).sum())}" for s in SLIDES))

for s in SLIDES:
    f = os.path.join(P, "st_bench/data", s, "adata_16um.h5ad")
    print(f"\n=== {s} ===")
    with h5py.File(f, "r") as h:
        print("  顶层:", list(h.keys()))
        print("  obs 键:", list(h["obs"].keys()))
        print("  obsm 键:", list(h["obsm"].keys()) if "obsm" in h else "无")
        xg = h["X"]
        if isinstance(xg, h5py.Group):
            print(f"  X: 稀疏 {dict(xg.attrs)} indptr={xg['indptr'].shape}")
            n_raw = xg["indptr"].shape[0] - 1
            dt = xg["data"].dtype
        else:
            print(f"  X: 稠密 {xg.shape} {xg.dtype}")
            n_raw = xg.shape[0]
            dt = xg.dtype
        print(f"  n_obs={n_raw}  X dtype={dt} "
              f"{'✅ 整数 counts' if np.issubdtype(dt, np.integer) else '⚠️ 非整数, 可能已归一化'}")
        # 前若干 data 值, 判断是不是原始 counts
        d0 = xg["data"][:2000] if isinstance(xg, h5py.Group) else xg[0, :2000]
        print(f"  data 样本: min={d0.min()} max={d0.max()} 是否整数={bool(np.all(d0 == np.round(d0)))}")
        # 基因名
        vg = h["var"]
        vk = [k for k in vg.keys() if k in ("_index", "gene_ids", "feature_name")]
        gname = None
        for k in ["_index"] + vk:
            if k in vg:
                try:
                    gname = np.array(vg[k][:]).astype(str); break
                except Exception:
                    pass
        if gname is not None:
            hit = np.isin(bgene, gname).sum()
            print(f"  var 基因 {len(gname)} 个, 覆盖 binned 的 200 个 HVG: {hit}/200 "
                  f"{'✅' if hit == 200 else '⚠️'}")
        # 坐标
        ob = h["obs"]
        cand = [k for k in ob.keys() if "pxl" in k or "array" in k]
        print(f"  obs 里的坐标列: {cand}")

    nb = int((bslide == s).sum())
    print(f"  行数对比: raw={n_raw}  binned={nb}  "
          f"{'✅ 一致' if n_raw == nb else '⚠️ 不一致, 需按坐标做匹配'}")

print("\n判读: 若行数一致且 counts 为整数, 可直接二项拆半;"
      "\n      若行数不一致, 用 pxl/array 坐标做最近邻匹配后再拆。")

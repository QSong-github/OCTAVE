# -*- coding: utf-8 -*-
"""
诊断: binned_16um.h5ad 的空间网格结构。

目的 —— 决定"带通分解 / 有效分辨率"这一步能否用 2D 栅格滤波(scipy.ndimage)实现。
若 bin 落在规则网格上, 可以把每个基因栅格化成一张二维图, 用高斯滤波 O(n) 拿到
多尺度平滑 S_sigma, 再作差得到带通分量 B_sigma = S_sigma - S_2sigma。
组织外的空洞用归一化卷积处理: smooth(value*mask) / smooth(mask)。

只读源目录, 不写任何东西。
"""
import numpy as np, anndata as ad

H5AD = "/path/to/spatial2exp/he2st_align/data/binned_16um.h5ad"

a = ad.read_h5ad(H5AD)
print(f"n_obs={a.n_obs}  n_vars={a.n_vars}", flush=True)
print("obs 列:", list(a.obs.columns))
print("obsm:", {k: tuple(np.shape(a.obsm[k])) for k in a.obsm})
print("layers:", list(a.layers))
print("var 前10:", list(a.var_names[:10]))

X = np.nan_to_num(np.asarray(a.X, np.float32))
print(f"X: dtype={X.dtype} min={X.min():.3f} max={X.max():.3f} mean={X.mean():.4f} "
      f"零元比例={(X == 0).mean() * 100:.1f}%")

slide = a.obs["slide_id"].astype(str).values
sp = np.asarray(a.obsm["spatial"])
print(f"\nspatial dtype={sp.dtype} shape={sp.shape}")

for s in sorted(set(slide)):
    m = slide == s
    c = sp[m].astype(np.int64)
    print(f"\n=== {s}  n={m.sum()} ===")
    steps = []
    for j in range(2):
        v = np.unique(c[:, j])
        d = np.unique(np.diff(v))
        step = int(d.min()) if len(d) else 1
        steps.append(step)
        print(f"  dim{j}: min={v.min()} max={v.max()} n_unique={len(v)} "
              f"相邻差集合(前8)={d[:8]}")
    # 规则性检验: 用最小步长离散化后看是否唯一
    g = np.stack([(c[:, j] - c[:, j].min()) // steps[j] for j in range(2)], 1)
    H, W = int(g[:, 0].max()) + 1, int(g[:, 1].max()) + 1
    n_uniq = len(np.unique(g[:, 0] * (W + 1) + g[:, 1]))
    print(f"  步长={steps} → 栅格 {H}×{W}={H * W} 格, 占用 {m.sum()} 个 "
          f"(填充率 {m.sum() / (H * W) * 100:.1f}%), 去重后 {n_uniq} 个 "
          f"{'✅ 无碰撞, 规则网格' if n_uniq == m.sum() else '⚠️ 有碰撞, 非规则网格'}")
    # 物理尺度: obsm['spatial'] 存的是 2µm bin 索引 (within_bench.py 里 UM=2.0)
    print(f"  → 物理: 每格 {steps[0] * 2}µm × {steps[1] * 2}µm, "
          f"视野约 {H * steps[0] * 2 / 1000:.1f}mm × {W * steps[1] * 2 / 1000:.1f}mm")

print("\n判读: 若两片都是'规则网格 + 无碰撞', 带通分解可以直接用 scipy.ndimage.gaussian_filter"
      "\n      配合归一化卷积处理组织外空洞, 264k×200 的开销可忽略。")

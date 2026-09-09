# -*- coding: utf-8 -*-
"""
诊断 v2 —— 修正 v1 的错误假设。

v1 用"相邻唯一值的最小间隔"当步长, 得到步长=1, 于是判成非规则网格。这是错的:
prep_bin.py 里 obsm['spatial'] = 各 super-bin 【成员 2µm bin 坐标的均值】,
因组织掩膜导致成员数不等, 质心会在 8×8 块内游走 → 取值离散但不落在 8 的倍数上。

真正的格点索引应当是 floor(spatial / 8): 质心必落在其所属块 [8b, 8b+7] 内,
所以 floor 能无损还原块索引。本脚本验证这一点, 并给出真实的格点间距与填充率。

同时确认基因符号在 var['gene'] 里(v1 打印的 var_names 是 '0'..'199' 的占位索引)。
只读, 不写。
"""
import numpy as np, anndata as ad
from scipy.spatial import cKDTree

H5AD = "/path/to/spatial2exp/he2st_align/data/binned_16um.h5ad"
FACTOR = 8          # prep_bin.py: 8×2µm = 16µm super-bin
UM_PER_UNIT = 2.0   # within_bench.py: obsm['spatial'] 以 2µm bin 为单位

a = ad.read_h5ad(H5AD)
print(f"n_obs={a.n_obs} n_vars={a.n_vars}")
print("var 列:", list(a.var.columns))
if "gene" in a.var:
    g = np.asarray(a.var["gene"]).astype(str)
    print(f"✅ 基因符号在 var['gene'] 里, 前12个: {list(g[:12])}")
    print(f"   唯一符号数 {len(set(g))}/{len(g)}")
else:
    print("⚠️ 没有 var['gene'], 基因符号丢失")

slide = a.obs["slide_id"].astype(str).values
sp = np.asarray(a.obsm["spatial"], np.float64)
pxl = np.asarray(a.obsm["pxl"], np.float64)

for s in sorted(set(slide)):
    m = slide == s
    c = sp[m]
    n = int(m.sum())
    print(f"\n=== {s}  n={n} ===")

    # (1) 真实最近邻间距 —— 不依赖任何假设
    d, _ = cKDTree(c).query(c, k=2)
    nn = d[:, 1]
    print(f"  最近邻间距(单位): 中位={np.median(nn):.2f} 均值={nn.mean():.2f} "
          f"p5={np.percentile(nn,5):.2f} p95={np.percentile(nn,95):.2f}")
    print(f"  → 物理间距 中位 {np.median(nn)*UM_PER_UNIT:.1f}µm "
          f"({'符合 16µm super-bin' if 12 < np.median(nn)*UM_PER_UNIT < 20 else '⚠️ 与 16µm 不符'})")

    # (2) floor(spatial/8) 能否无损还原格点索引
    gi = np.floor(c / FACTOR).astype(np.int64)
    gi -= gi.min(0)
    H, W = int(gi[:, 0].max()) + 1, int(gi[:, 1].max()) + 1
    key = gi[:, 0] * (W + 1) + gi[:, 1]
    nu = len(np.unique(key))
    ok = nu == n
    print(f"  floor(spatial/{FACTOR}) → 栅格 {H}×{W}={H*W}, 占用 {n}, 去重 {nu} "
          f"{'✅ 零碰撞, 可无损栅格化' if ok else f'⚠️ 碰撞 {n-nu} 个 ({(n-nu)/n*100:.2f}%)'}")
    print(f"  填充率 {n/(H*W)*100:.1f}% (其余是组织外空洞 → 归一化卷积处理)")
    print(f"  视野 {H*FACTOR*UM_PER_UNIT/1000:.2f}mm × {W*FACTOR*UM_PER_UNIT/1000:.2f}mm")

    # (3) 质心相对块原点的偏移分布 —— 验证"质心在块内游走"的解释
    off = c - np.floor(c / FACTOR) * FACTOR
    print(f"  块内偏移: dim0 中位={np.median(off[:,0]):.2f} dim1 中位={np.median(off[:,1]):.2f} "
          f"(满员块应为 3.5)")

    # (4) spatial 与 pxl 的比例 —— 交叉验证单位
    sc = []
    for j in range(2):
        rs, rp = np.ptp(c[:, j]), np.ptp(pxl[:, j])
        sc.append(rp / rs if rs > 0 else np.nan)
    print(f"  pxl/spatial 跨度比: {sc[0]:.3f}, {sc[1]:.3f} (全分辨率像素/2µm单位)")

print("\n判读: 若两片都是'零碰撞', 带通分解直接在 floor(spatial/8) 栅格上做 —— "
      "\n      每个基因摊成 H×W 图, scipy.ndimage.gaussian_filter 多尺度平滑, "
      "\n      空洞用 smooth(v*mask)/smooth(mask) 归一化卷积。")

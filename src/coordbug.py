# -*- coding: utf-8 -*-
"""
量化 obsm['spatial'] 的坐标污染。

prep_bin.py:
    coordb = (A @ np.nan_to_num(coord[sp])) / cnt[:, None]
coord.npy 含 NaN, nan_to_num 把它变成 0, 却仍除以【全部】成员数 cnt ⇒ 有 NaN 成员的
super-bin 质心被拉向原点。这解释了 v2 的 20~29% 格点碰撞, 也解释了 P2/P5 碰撞率不同。

obsm['pxl'] 走的是另一条路(源自 obs 的 pxl_*_in_fullres, 无 NaN), v3 已验证其最近邻
间距恰为 16.00µm ⇒ pxl 可信。

本脚本把两者都换算成 µm 并逐点比较位移, 回答: 用 spatial 做空间划分/平滑的现有结果
(within_bench.py, smooth_exp.py) 受多大影响。只读。
"""
import numpy as np, h5py, glob, os, anndata as ad

SRC = "/blue/qsong1/wang.qing/spatial2exp/he2st_align/data/virtualST"
H5AD = "/blue/qsong1/wang.qing/spatial2exp/he2st_align/data/binned_16um.h5ad"
SLIDES = ["Visium_HD_Human_Colon_Cancer_P2", "Visium_HD_Human_Colon_Cancer_P5"]
PX_PER_UM = {SLIDES[0]: 3.6499, SLIDES[1]: 3.6526}      # v3 标定
UM = 2.0                                                 # within_bench.py 的假设

# ---- 1. coord.npy 里 NaN 到底占多少
for s in SLIDES:
    p = glob.glob(os.path.join(SRC, s, "*coord.npy"))[0]
    co = np.load(p, mmap_mode="r")
    n = co.shape[0]
    bad = 0
    for i in range(0, n, 1000000):
        bad += int(np.isnan(np.asarray(co[i:i + 1000000])).any(1).sum())
    print(f"{s}: coord.npy {n} 行, 含 NaN {bad} ({bad/n*100:.1f}%)")

# ---- 2. spatial 与 pxl 的逐点位移
a = ad.read_h5ad(H5AD)
slide = a.obs["slide_id"].astype(str).values
sp = np.asarray(a.obsm["spatial"], np.float64)
pxl = np.asarray(a.obsm["pxl"], np.float64)

for s in SLIDES:
    m = slide == s
    A_um = sp[m] * UM                       # within_bench/smooth_exp 眼中的坐标
    B_um = pxl[m] / PX_PER_UM[s]            # 可信坐标
    # 两者可能差一个刚体变换(轴序/翻转/旋转/平移), 先用最小二乘拟合仿射再看残差
    # 残差 = 无法用统一变换解释的部分 = 真正的污染
    D = np.hstack([A_um, np.ones((A_um.shape[0], 1))])
    T, *_ = np.linalg.lstsq(D, B_um, rcond=None)
    resid = B_um - D @ T
    r = np.linalg.norm(resid, axis=1)
    print(f"\n=== {s} n={m.sum()} ===")
    print(f"  仿射拟合后残差(µm): 中位={np.median(r):.2f} 均值={r.mean():.2f} "
          f"p90={np.percentile(r,90):.2f} p99={np.percentile(r,99):.2f} max={r.max():.1f}")
    for th in (8, 16, 32, 64, 160):
        print(f"    位移 > {th:3d}µm ({th/16:.0f}个bin): {(r>th).mean()*100:6.2f}%")

print("\n判读: 若 p90 残差远小于 within_bench 的隔离带(64µm)与棋盘块(320µm), 则既有结果基本安全;"
      "\n      若尾部有大量点位移 >64µm, 则片内划分的'空间分离'没有真正做到, 需用 pxl 重跑。")

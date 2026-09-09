# -*- coding: utf-8 -*-
"""
诊断 v3 —— 只解决一个问题: **拿到可靠的物理尺度(µm)**。

v2 证伪了 floor(spatial/8) 能还原格点: 碰撞 20~29%。说明 prep_bin.py 里存进
obsm['spatial'] 的 coord.npy 并不是 (array_row, array_col) 本身。继续反推它的坐标
约定没有价值 —— 因为带通分解改用**图扩散**实现后, 根本不需要规则格点, 只需要一个
可信的物理距离。obsm['pxl'] 是全分辨率像素中心, 物理、各向同性, 正合适。

本脚本做三件事:
  1. 从源 adata.h5ad 直接标定 像素/µm  —— 相邻 array_col 差 1 ⇔ 2µm, 量它对应多少像素
  2. 用该标定把 binned h5ad 的 pxl 换算成 µm, 看最近邻间距是否 ≈16µm
  3. 顺带确认 coord.npy 到底是什么(与 array_row/col 的关系), 存档备查

只读源目录。
"""
import numpy as np, h5py, glob, os, anndata as ad
from scipy.spatial import cKDTree

SRC = "/path/to/spatial2exp/he2st_align/data/virtualST"
H5AD = "/path/to/spatial2exp/he2st_align/data/binned_16um.h5ad"
SLIDES = ["Visium_HD_Human_Colon_Cancer_P2", "Visium_HD_Human_Colon_Cancer_P5"]

px_per_um = {}
for s in SLIDES:
    d = os.path.join(SRC, s)
    print(f"\n########## {s} ##########", flush=True)
    with h5py.File(os.path.join(d, "adata.h5ad"), "r") as h:
        ob = h["obs"]
        print("  源 obs 键:", list(ob.keys()))
        row = ob["array_row"][:].astype(np.int64)
        col = ob["array_col"][:].astype(np.int64)
        pr = ob["pxl_row_in_fullres"][:].astype(np.float64)
        pc = ob["pxl_col_in_fullres"][:].astype(np.float64)
    n = len(row)
    print(f"  源 2µm bin: {n}  array_row∈[{row.min()},{row.max()}] "
          f"array_col∈[{col.min()},{col.max()}]")

    # ---- 1. 标定: array_col 差 1 ⇔ 2µm, 对应多少像素? 用最小二乘斜率, 稳健且不受缺格影响
    def slope(a, b):
        A = np.stack([a.astype(np.float64), np.ones_like(a, np.float64)], 1)
        return float(np.linalg.lstsq(A, b, rcond=None)[0][0])
    scr, scc = slope(row, pr), slope(col, pc)
    cross = (abs(slope(row, pc)), abs(slope(col, pr)))
    print(f"  像素/单位: d(pxl_row)/d(array_row)={scr:.4f}  d(pxl_col)/d(array_col)={scc:.4f}")
    print(f"  交叉项(应≈0, 否则坐标系有旋转): {cross[0]:.4f}, {cross[1]:.4f}")
    pxum = (abs(scr) + abs(scc)) / 2 / 2.0                # 每 µm 多少像素 (1 单位 = 2µm)
    px_per_um[s] = pxum
    print(f"  ⇒ 标定 {pxum:.4f} px/µm  ({1/pxum:.4f} µm/px)")

    # ---- 3. coord.npy 到底是什么
    cp = glob.glob(os.path.join(d, "*coord.npy"))
    if cp:
        co = np.load(cp[0], mmap_mode="r")
        co = np.asarray(co[:min(n, 2000000)], np.float64)
        rr, cc = row[:len(co)], col[:len(co)]
        print(f"  coord.npy {os.path.basename(cp[0])} shape={np.shape(np.load(cp[0], mmap_mode='r'))}")
        for j in range(2):
            k_row, k_col = slope(rr, co[:, j]), slope(cc, co[:, j])
            print(f"    coord[:,{j}] vs array_row 斜率={k_row:.4f} | vs array_col 斜率={k_col:.4f} "
                  f"| 范围[{co[:,j].min():.1f},{co[:,j].max():.1f}]")
    else:
        print("  未找到 coord.npy")

# ---- 2. binned h5ad: pxl 换算成 µm 后的最近邻间距
a = ad.read_h5ad(H5AD)
slide = a.obs["slide_id"].astype(str).values
pxl = np.asarray(a.obsm["pxl"], np.float64)
print("\n########## binned 16µm: pxl → µm 后的间距 ##########")
for s in SLIDES:
    m = slide == s
    c = pxl[m] / px_per_um[s]                     # → µm
    d, _ = cKDTree(c).query(c, k=2)
    nn = d[:, 1]
    hist, edges = np.histogram(nn, bins=np.arange(0, 40, 1.0))
    mode = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    print(f"  {s}: n={m.sum()}  最近邻 中位={np.median(nn):.2f}µm 众数≈{mode:.1f}µm "
          f"p95={np.percentile(nn,95):.2f}µm")
    print(f"    直方图(µm): " + " ".join(f"{int(edges[i])}:{hist[i]}" for i in range(len(hist)) if hist[i] > 0)[:400])
    print(f"    视野 {np.ptp(c[:,0])/1000:.2f}mm × {np.ptp(c[:,1])/1000:.2f}mm")

print("\n判读: 众数应 ≈16µm。确认后, 带通分解用 pxl(µm) 建 kNN 图 + 图扩散(热核),"
      "\n      不需要规则格点, σ 轴可直接标微米。")

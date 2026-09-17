# -*- coding: utf-8 -*-
"""HEST 基准的 patches/{id}.h5 里 coords 存的是 (y, x)（patch 中心，全分辨率像素）；用 WSI 反向裁剪比对验证：按 (x=coords[:,1], y=coords[:,0]) 取 409 px
再缩到 224 与 h5 里的 img 相关 0.97–1.00，按原顺序则 0.1–0.5 且大量越界。TRIPLEX 的 H5TileDataset 按 (x, y) 读 WSI，
故只在我们给 TRIPLEX 的拷贝里交换两列（幂等：写 attrs['coords_swapped']=1）。作者代码不动。"""
import sys, glob, h5py, numpy as np
d = sys.argv[1]; n = 0
for f in sorted(glob.glob(f"{d}/*.h5")):
    with h5py.File(f, "r+") as h:
        if h["coords"].attrs.get("coords_swapped", 0) == 1: continue
        c = h["coords"][:]; h["coords"][...] = c[:, ::-1]; h["coords"].attrs["coords_swapped"] = 1; n += 1
print(f"{d}: 交换了 {n} 个文件的 coords 列")

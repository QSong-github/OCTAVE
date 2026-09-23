# -*- coding: utf-8 -*-
"""为个别切片直接从 WSI 切邻居块（几何与 HEST dump_patches 一致：目标 1120 px @0.5 µm/px，源窗口 = 目标 patch 源尺寸 ×5，居中于同一 spot）。
用于 TENX111：它在 HEST 原始 st/ 里的条码与坐标系与基准 patch 不一致（同一问题也曾使其表达文件为空），dump_patches 把全部窗口判为图外。
基准 patch 的坐标经反查裁剪验证为左上角 (x, y)（相关 1.00），故按此切。越界部分以白色填充，与 openslide 读出图外区域的处理一致。"""
import sys, os, h5py, numpy as np, openslide
from PIL import Image
C, sid = sys.argv[1], sys.argv[2]
d = f"/path/to/project/data/triplex/{C}"; N = 5
p = f"{d}/patches/{sid}.h5"
with h5py.File(p, "r") as h:
    co = h["coords"][:].astype(np.int64); bc = np.asarray(h["barcode"][:]).flatten(); at = dict(h["img"].attrs)
    swapped = int(h["coords"].attrs.get("coords_swapped", 0))
if swapped: co = co[:, ::-1]                                    # 还原为文件原始列序（该片原始即 (x, y) 左上角）
tgt = int(at.get("patch_size", 224)); fac = float(at.get("factor", 1.0))
src_224 = int(round(224 * fac)) if tgt == 224 else tgt          # 224 目标对应的源窗口边长
src_nbr = src_224 * N                                           # 邻居窗口源边长
s = openslide.OpenSlide(f"/path/to/project/data/hest_wsis/wsis/{sid}.tif")
off = (src_nbr - src_224) // 2
out = f"{d}/patches/neighbor/{sid}.h5"; os.makedirs(os.path.dirname(out), exist_ok=True)
if os.path.exists(out): os.remove(out)
with h5py.File(out, "w") as h:
    img = h.create_dataset("img", shape=(len(co), 224 * N, 224 * N, 3), dtype=np.uint8, chunks=(1, 224 * N, 224 * N, 3), compression=None)
    for i, (x, y) in enumerate(co.tolist()):
        t = s.read_region((int(x) - off, int(y) - off), 0, (src_nbr, src_nbr)).convert("RGB").resize((224 * N, 224 * N), Image.BILINEAR)
        img[i] = np.asarray(t, np.uint8)
        if i % 500 == 0: print(f"  {i}/{len(co)}", flush=True)
    h.create_dataset("coords", data=co); h.create_dataset("barcode", data=bc)
    h.attrs["matched_to_target"] = True; h.attrs["from_wsi_direct"] = True
print(f"{C}/{sid}: 邻居块 {len(co)} 片，源窗口 {src_nbr}px → {224*N}px（目标 patch 源 {src_224}px）")

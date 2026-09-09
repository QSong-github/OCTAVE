# -*- coding: utf-8 -*-
"""Path Foundation 两段式第一段（hest 环境）：按 obsm['pxl'] 在 openslide level 0 切 ctx_px 方块，缩放到 224，写 gzip HDF5 分片（uint8）。
按 chunk（256 块）攒满后整块写入：逐块写入会让每次写都重压缩整个 chunk，一片 2 万块要几小时；整块写入只需几分钟。"""
import argparse, numpy as np, h5py, anndata as ad, openslide, time
from PIL import Image
a = argparse.ArgumentParser(); a.add_argument("--h5ad"); a.add_argument("--tiff"); a.add_argument("--start", type=int); a.add_argument("--end", type=int); a.add_argument("--out"); a = a.parse_args()
A = ad.read_h5ad(a.h5ad, backed="r"); pxl = np.asarray(A.obsm["pxl"], np.float64); ctx = int(round(61.4 * float(A.uns["px_per_um"]))); half = ctx // 2
sl = openslide.OpenSlide(a.tiff); idx = list(range(a.start, min(a.end, len(pxl)))); CH = 256; t0 = time.time()
with h5py.File(a.out, "w") as h:
    d = h.create_dataset("img", (len(idx), 224, 224, 3), dtype="uint8", chunks=(CH, 224, 224, 3), compression="gzip", compression_opts=3)
    buf = np.empty((CH, 224, 224, 3), np.uint8); n = 0; k0 = 0
    for i in idx:
        x, y = pxl[i]; t = sl.read_region((int(x - half), int(y - half)), 0, (ctx, ctx)).convert("RGB")
        if ctx != 224: t = t.resize((224, 224), Image.BILINEAR)
        buf[n] = np.asarray(t, dtype=np.uint8); n += 1
        if n == CH: d[k0:k0 + n] = buf[:n]; k0 += n; n = 0
    if n: d[k0:k0 + n] = buf[:n]
print(f"cut {a.start}-{min(a.end, len(pxl))} of {len(pxl)} ctx={ctx} -> {a.out}  ({time.time() - t0:.0f} s)", flush=True)

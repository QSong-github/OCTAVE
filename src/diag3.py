# -*- coding: utf-8 -*-
"""1px 步长的平移搜索 —— 上一轮步长 4px，1–2px 的偏移会被漏掉。"""
import glob, numpy as np, openslide, h5py
PA = "/path/to/align_workspace"
A = glob.glob(f"{PA}/data/virtualST/*P2/*PYRAMIDAL*.tif*")[0]
L = "/path/to/systema4ST/data/visiumhd/Visium_HD_Human_Colon_Cancer_P2/Visium_HD_Human_Colon_Cancer_P2_LOSSLESS.tif"
with h5py.File(f"{PA}/data/binned_16um.h5ad", "r") as h:
    sid = h["obs"]["slide_id"]
    cats = [c.decode() if isinstance(c, bytes) else str(c) for c in sid["categories"][:]]
    names = np.array(cats)[sid["codes"][:]]; pxl = h["obsm"]["pxl"][:]
pxl = pxl[np.char.endswith(names.astype(str), "P2")]
sa, sl = openslide.OpenSlide(A), openslide.OpenSlide(L)
rng = np.random.default_rng(2); R = 6
print(f"{'idx':>7}  最佳(dx,dy)  最佳相关   (0,0)相关   MAE@最佳  MAE@(0,0)")
for i in rng.choice(len(pxl), 6, replace=False):
    x, y = int(pxl[i][0]), int(pxl[i][1])
    ta = np.asarray(sa.read_region((x-112, y-112), 0, (224, 224)).convert("RGB"), np.float32)
    big = np.asarray(sl.read_region((x-112-R, y-112-R), 0, (224+2*R, 224+2*R)).convert("RGB"), np.float32)
    best, bd, bm = -2, None, None
    for dy in range(2*R+1):
        for dx in range(2*R+1):
            w = big[dy:dy+224, dx:dx+224]
            c = np.corrcoef(ta.ravel(), w.ravel())[0, 1]
            if c > best: best, bd, bm = c, (dx-R, dy-R), np.abs(ta-w).mean()
    z = big[R:R+224, R:R+224]
    print(f"{i:>7}  {str(bd):>10}  {best:>8.4f}   {np.corrcoef(ta.ravel(), z.ravel())[0,1]:>9.4f}"
          f"   {bm:>8.2f}  {np.abs(ta-z).mean():>9.2f}")

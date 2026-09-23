# -*- coding: utf-8 -*-
"""诊断: 上游项目 P2 金字塔 vs 我 vips 转的 P2 金字塔，为何嵌入不等价。"""
import glob, numpy as np, openslide, h5py
PA = "/path/to/upstream_align"
A = glob.glob(f"{PA}/data/virtualST/*P2/*PYRAMIDAL*.tif*")[0]
B = "/path/to/project/data/visiumhd/Visium_HD_Human_Colon_Cancer_P2/Visium_HD_Human_Colon_Cancer_P2_PYRAMIDAL.tif"
sa, sb = openslide.OpenSlide(A), openslide.OpenSlide(B)
print(f"上游项目 {A.split('/')[-1]}\n  尺寸={sa.dimensions} 层={sa.level_count}")
for k in ("openslide.vendor", "tiff.ImageDescription", "tiff.PhotometricInterpretation",
          "tiff.ResolutionUnit", "tiff.XResolution", "openslide.mpp-x"):
    print(f"    {k} = {str(sa.properties.get(k))[:110]}")
print(f"我的  {B.split('/')[-1]}\n  尺寸={sb.dimensions} 层={sb.level_count}")
for k in ("openslide.vendor", "tiff.ImageDescription", "tiff.PhotometricInterpretation",
          "tiff.ResolutionUnit", "tiff.XResolution", "openslide.mpp-x"):
    print(f"    {k} = {str(sb.properties.get(k))[:110]}")

with h5py.File(f"{PA}/data/binned_16um.h5ad", "r") as h:
    sid = h["obs"]["slide_id"]
    cats = [c.decode() if isinstance(c, bytes) else str(c) for c in sid["categories"][:]]
    names = np.array(cats)[sid["codes"][:]]
    pxl = h["obsm"]["pxl"][:]
pxl = pxl[np.char.endswith(names.astype(str), "P2")]
print(f"\npxl 范围: x[{pxl[:,0].min():.0f},{pxl[:,0].max():.0f}] y[{pxl[:,1].min():.0f},{pxl[:,1].max():.0f}]  n={len(pxl)}")

print(f"\n{'idx':>6}{'像素MAE':>10}{'相关':>9}{'A均值':>9}{'B均值':>9}  最佳平移(dx,dy)")
rng = np.random.default_rng(0)
for i in rng.choice(len(pxl), 6, replace=False):
    x, y = int(pxl[i][0]), int(pxl[i][1])
    ta = np.asarray(sa.read_region((x-112, y-112), 0, (224, 224)).convert("RGB"), np.float32)
    tb = np.asarray(sb.read_region((x-112, y-112), 0, (224, 224)).convert("RGB"), np.float32)
    mae = np.abs(ta-tb).mean(); r = np.corrcoef(ta.ravel(), tb.ravel())[0,1]
    # 在 ±24px 内搜最佳平移，判断是否只是坐标偏移
    best, bd = -2, None
    big = np.asarray(sb.read_region((x-112-24, y-112-24), 0, (272, 272)).convert("RGB"), np.float32)
    for dy in range(0, 49, 4):
        for dx in range(0, 49, 4):
            c = np.corrcoef(ta.ravel(), big[dy:dy+224, dx:dx+224].ravel())[0,1]
            if c > best: best, bd = c, (dx-24, dy-24)
    print(f"{i:>6}{mae:>10.2f}{r:>9.4f}{ta.mean():>9.1f}{tb.mean():>9.1f}   r={best:.4f} @ {bd}")

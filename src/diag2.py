# -*- coding: utf-8 -*-
"""
判定差异来源：母项目的图 vs 10x 原图。
 ① 我的 JPEG 版 与 我的无损版 的嵌入是否一致 → 压缩到底影不影响
 ② 无损版 与 母项目图 的像素差 → 若仍 ~1.2%，证明母项目那张图内容本身不同
"""
import glob, numpy as np, openslide, h5py, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hd_embed import build, embed
PA = "/blue/qsong1/wang.qing/spatial2exp/he2st_align"
D = "/blue/qsong1/wang.qing/systema4ST/data/visiumhd/Visium_HD_Human_Colon_Cancer_P2"
A = glob.glob(f"{PA}/data/virtualST/*P2/*PYRAMIDAL*.tif*")[0]
J = f"{D}/Visium_HD_Human_Colon_Cancer_P2_PYRAMIDAL.tif"
L = f"{D}/Visium_HD_Human_Colon_Cancer_P2_LOSSLESS.tif"

with h5py.File(f"{PA}/data/binned_16um.h5ad", "r") as h:
    sid = h["obs"]["slide_id"]
    cats = [c.decode() if isinstance(c, bytes) else str(c) for c in sid["categories"][:]]
    names = np.array(cats)[sid["codes"][:]]; pxl = h["obsm"]["pxl"][:]
pxl = pxl[np.char.endswith(names.astype(str), "P2")][:2000]

print("=== ① 压缩的影响：我的 JPEG 版 vs 我的无损版 ===")
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, tf, prec = build("hibou_l", dev)
ej = embed(model, tf, prec, J, pxl, 224, dev)
el = embed(model, tf, prec, L, pxl, 224, dev)
cos = (ej*el).sum(1)/(np.linalg.norm(ej,axis=1)*np.linalg.norm(el,axis=1)+1e-9)
print(f"  余弦 中位={np.median(cos):.6f} 最小={cos.min():.6f}  最大绝对差={np.abs(ej-el).max():.6f}")
print(f"  ⇒ {'压缩无影响，JPEG Q95 可用' if np.median(cos)>0.999 else '压缩确有影响，须用无损'}")

print("\n=== ② 无损版 vs 母项目图 的像素差 ===")
sa, sl = openslide.OpenSlide(A), openslide.OpenSlide(L)
rng = np.random.default_rng(1)
maes, cors = [], []
for i in rng.choice(len(pxl), 8, replace=False):
    x, y = int(pxl[i][0]), int(pxl[i][1])
    ta = np.asarray(sa.read_region((x-112,y-112),0,(224,224)).convert("RGB"), np.float32)
    tl = np.asarray(sl.read_region((x-112,y-112),0,(224,224)).convert("RGB"), np.float32)
    maes.append(np.abs(ta-tl).mean()); cors.append(np.corrcoef(ta.ravel(), tl.ravel())[0,1])
print(f"  像素 MAE 中位={np.median(maes):.2f}/255  相关 中位={np.median(cors):.4f}")
print(f"  ⇒ {'母项目那张图的内容本身与 10x 原图不同（处理过，来源不可考）' if np.median(maes)>1 else '像素一致，差异另有来源'}")

print("\n=== ③ 通道级差异（判断是否做过色彩变换）===")
x, y = int(pxl[0][0]), int(pxl[0][1])
ta = np.asarray(sa.read_region((x-112,y-112),0,(224,224)).convert("RGB"), np.float32)
tl = np.asarray(sl.read_region((x-112,y-112),0,(224,224)).convert("RGB"), np.float32)
for c, nm in enumerate("RGB"):
    d = ta[...,c]-tl[...,c]
    print(f"  {nm}: 母项目均值={ta[...,c].mean():7.2f}  10x均值={tl[...,c].mean():7.2f}  差均值={d.mean():+6.2f} 差std={d.std():5.2f}")

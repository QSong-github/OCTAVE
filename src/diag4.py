# -*- coding: utf-8 -*-
"""
高频内容对比：上游项目的 P2 图 是否相对 10x 原图偏模糊？

已排除：空间偏移（1px 步长搜索全为 (0,0)）、色彩变换（三通道均值一致）、
        有损压缩（我的 JPEG vs 无损，嵌入余弦 0.9999）。
剩下的可能：高频（纹理）内容不同 —— 低频吻合(相关 0.997)但嵌入差 0.888 正符合此特征。

若上游项目的图确实偏模糊，则其等价 σ（108–120µm）含有图像预处理造成的系统性高估，
这将直接影响本文头条数字，必须查实。
"""
import glob, numpy as np, openslide, h5py
PA = "/path/to/upstream_align"
A = glob.glob(f"{PA}/data/virtualST/*P2/*PYRAMIDAL*.tif*")[0]
L = "/path/to/project/data/visiumhd/Visium_HD_Human_Colon_Cancer_P2/Visium_HD_Human_Colon_Cancer_P2_LOSSLESS.tif"
J = "/path/to/project/data/visiumhd/Visium_HD_Human_Colon_Cancer_P2/Visium_HD_Human_Colon_Cancer_P2_PYRAMIDAL.tif"

with h5py.File(f"{PA}/data/binned_16um.h5ad", "r") as h:
    sid = h["obs"]["slide_id"]
    cats = [c.decode() if isinstance(c, bytes) else str(c) for c in sid["categories"][:]]
    names = np.array(cats)[sid["codes"][:]]; pxl = h["obsm"]["pxl"][:]
pxl = pxl[np.char.endswith(names.astype(str), "P2")]
S = {"上游项目": openslide.OpenSlide(A), "10x无损": openslide.OpenSlide(L), "10x-JPEG": openslide.OpenSlide(J)}

def stats(t):
    g = t.mean(2)
    gx = np.diff(g, axis=1); gy = np.diff(g, axis=0)
    grad = np.sqrt(gx[:-1]**2 + gy[:, :-1]**2).mean()
    lap = (g[1:-1,1:-1]*4 - g[:-2,1:-1] - g[2:,1:-1] - g[1:-1,:-2] - g[1:-1,2:]).var()
    F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))**2
    n = F.shape[0]; c = n//2
    yy, xx = np.mgrid[:n, :n]; r = np.sqrt((yy-c)**2 + (xx-c)**2)
    lo = F[r <= n*0.10].sum(); hi = F[r >= n*0.25].sum()
    return grad, lap, hi/max(lo, 1e-9)

rng = np.random.default_rng(3)
idx = rng.choice(len(pxl), 40, replace=False)
acc = {k: [] for k in S}
for i in idx:
    x, y = int(pxl[i][0]), int(pxl[i][1])
    for k, s in S.items():
        t = np.asarray(s.read_region((x-112, y-112), 0, (224, 224)).convert("RGB"), np.float32)
        acc[k].append(stats(t))
print(f"{'图源':>10}{'梯度幅值':>11}{'Laplacian方差':>15}{'高频/低频能量比':>17}")
for k in S:
    a = np.array(acc[k])
    print(f"{k:>10}{a[:,0].mean():>11.3f}{a[:,1].mean():>15.2f}{a[:,2].mean():>17.5f}")
b = np.array(acc["上游项目"]); o = np.array(acc["10x无损"])
print(f"\n上游项目 / 10x 原图 的比值：梯度 {b[:,0].mean()/o[:,0].mean():.4f}  "
      f"Laplacian {b[:,1].mean()/o[:,1].mean():.4f}  高频占比 {b[:,2].mean()/o[:,2].mean():.4f}")
print("⇒ " + ("上游项目的图更模糊 —— 其等价 σ 含图像预处理造成的高估"
              if b[:,1].mean() < o[:,1].mean()*0.95 else
              "上游项目的图更锐 —— 可能做过锐化/去噪" if b[:,1].mean() > o[:,1].mean()*1.05 else
              "高频能量相当 —— 模糊假设不成立，差异另有来源"))

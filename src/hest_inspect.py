# -*- coding: utf-8 -*-
"""
探清 HEST-benchmark 的 72 个样本结构 —— 为跨样本的 η / A_coarse 阶梯做准备。

目标实验(对应 Systema Fig 3c): 逐样本测"粗尺度结构能拿多少分", 再与该样本上
HEST-benchmark 官方协议报告的 PCC 求相关。Systema 在 10 个数据集上测到 0.91–0.95;
我们有 72 个样本、10 个器官, 广度更大。

本脚本只读 + 打印结构: obs/obsm 键、坐标单位、表达是否已归一化、50 基因清单、划分文件。
"""
import os, glob, json, numpy as np, anndata as ad

B = "/blue/qsong1/wang.qing/he2st/HEST/eval/bench_data"
cohorts = sorted(os.listdir(B))
print("队列:", cohorts)

tot = 0
for c in cohorts:
    ads = sorted(glob.glob(os.path.join(B, c, "adata", "*.h5ad")))
    tot += len(ads)
    sp = sorted(glob.glob(os.path.join(B, c, "splits", "*.csv")))
    vg = os.path.join(B, c, "var_50genes.json")
    ng = len(json.load(open(vg))["genes"]) if os.path.exists(vg) else "?"
    print(f"\n=== {c}: {len(ads)} 样本, {len(sp)} 个划分文件, 评测基因 {ng} ===")
    if not ads:
        continue
    a = ad.read_h5ad(ads[0])
    print(f"  样本 {os.path.basename(ads[0])}: shape={a.shape}")
    print(f"  obs: {list(a.obs.columns)[:8]}")
    print(f"  obsm: { {k: tuple(np.shape(a.obsm[k])) for k in a.obsm} }")
    print(f"  layers: {list(a.layers)}  var前5: {list(a.var_names[:5])}")
    X = np.asarray(a.X.todense() if hasattr(a.X, 'todense') else a.X, np.float32)
    print(f"  X: min={X.min():.2f} max={X.max():.2f} mean={X.mean():.3f} "
          f"整数={bool(np.all(X == np.round(X)))} 零元={(X == 0).mean()*100:.1f}%")
    if "spatial" in a.obsm:
        s = np.asarray(a.obsm["spatial"], np.float64)
        from scipy.spatial import cKDTree
        d, _ = cKDTree(s).query(s, k=min(2, len(s)))
        nn = d[:, 1] if d.ndim > 1 and d.shape[1] > 1 else d
        print(f"  spatial: 范围 x[{s[:,0].min():.0f},{s[:,0].max():.0f}] "
              f"y[{s[:,1].min():.0f},{s[:,1].max():.0f}] 最近邻中位={np.median(nn):.1f}")

print(f"\n总样本 {tot}")
print("\n判读: 需确认 (a) spatial 的单位(像素还是µm) (b) X 是否已 log 归一化"
      "\n      (c) 每样本 spot 数是否足够建图(<500 的样本要排除)。")

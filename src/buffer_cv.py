# -*- coding: utf-8 -*-
"""
空间泄漏：逐 spot 随机划分 vs 空间块划分 + 缓冲带。

首轮只做了 4×4 块划分, 缓冲带 0→800µm 仅掉 2% —— 我原本预期的急落
未发生。但那不是"无泄漏", 是**对照选错了**: 13 万 spot 的切片切 4×4, 每块约
8000 spot、跨度 1–2mm, d=0 时绝大多数测试点本就远离训练集。
真正泄漏的协议是**逐 spot 随机划分**(把紧邻 spot 放进训练集), 而那恰是不少
H&E→ST 论文在用的。本轮加上它, 并加密网格(4×4 / 8×8 / 16×16)看块尺寸的作用。

参照系: 跨片 0.5785（三方夹逼实测, 同口径 top-50 HVG）。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E
from baselines import ridge_predict
from effres import PX_PER_UM, SLIDES, SEB

ap = argparse.ArgumentParser()
ap.add_argument("--tower", default="hibou_l"); ap.add_argument("--hvg", type=int, default=50)
ap.add_argument("--grids", default="4,8,16")
ap.add_argument("--buf", default="0,25,50,100,200,400,800")
ap.add_argument("--nrand", type=int, default=8)
a_ = ap.parse_args()
BUF = [float(x) for x in a_.buf.split(",")]
GRIDS = [int(x) for x in a_.grids.split(",")]

a = ad.read_h5ad(SEB.H5AD)
expr = np.nan_to_num(np.asarray(a.X, np.float32))
slide = a.obs["slide_id"].astype(str).values
pxl = np.asarray(a.obsm["pxl"], np.float64)
img = np.nan_to_num(np.concatenate([
    np.load(os.path.join(SEB.EMBDIR, f"emb_{a_.tower}_P2.npy")),
    np.load(os.path.join(SEB.EMBDIR, f"emb_{a_.tower}_P5.npy"))]).astype(np.float32))

out = {}
for s in SLIDES:
    m = slide == s
    xy = pxl[m] / PX_PER_UM[s]; X = img[m]; Y = expr[m]
    gidx = E.topk_hvg(Y, a_.hvg); n = len(Y)
    fit = lambda tr, te: float(E.per_gene_pcc(
        ridge_predict(X[tr], Y[tr], X[te], 1e4)[:, gidx], Y[te][:, gidx]).mean())
    out[s] = {}
    print(f"\n[{s}] n={n}", flush=True)

    rng = np.random.default_rng(0); acc = []
    for r in range(a_.nrand):                      # 逐 spot 随机划分, 测试比例 1/16
        te = np.zeros(n, bool); te[rng.choice(n, n // 16, replace=False)] = True
        acc.append(fit(~te, te))
    out[s]["random_spot"] = float(np.mean(acc))
    print(f"  逐 spot 随机划分      PCC={np.mean(acc):.4f}  (±{np.std(acc):.4f}, {a_.nrand} 次)", flush=True)

    for g in GRIDS:
        qx = np.quantile(xy[:, 0], np.linspace(0, 1, g + 1))
        qy = np.quantile(xy[:, 1], np.linspace(0, 1, g + 1))
        for d in BUF:
            accs, kept = [], []
            for i in range(g):
                for j in range(g):
                    x0, x1, y0, y1 = qx[i], qx[i+1], qy[j], qy[j+1]
                    te = ((xy[:, 0] >= x0) & (xy[:, 0] <= x1) &
                          (xy[:, 1] >= y0) & (xy[:, 1] <= y1))
                    if te.sum() < 200: continue
                    tr = ~((xy[:, 0] >= x0-d) & (xy[:, 0] <= x1+d) &
                           (xy[:, 1] >= y0-d) & (xy[:, 1] <= y1+d))
                    if tr.sum() < 500: continue
                    accs.append(fit(tr, te)); kept.append(float(tr.sum()) / (~te).sum())
            if not accs: continue
            out[s][f"g{g}_d{d:g}"] = dict(pcc=float(np.mean(accs)), n_block=len(accs),
                                          train_frac=float(np.mean(kept)))
            print(f"  网格{g:2d}×{g:<2d} 缓冲{d:5.0f}µm  PCC={np.mean(accs):.4f}  "
                  f"块={len(accs):3d}  训练保留={np.mean(kept)*100:3.0f}%", flush=True)

print(f"\n{'='*70}"); print("两片均值 —— 划分协议对分数的影响"); print("="*70)
mean = lambda k: np.mean([out[s][k]["pcc"] if isinstance(out[s][k], dict) else out[s][k]
                          for s in SLIDES if k in out[s]])
base = mean("random_spot")
print(f"{'协议':>30s}{'PCC':>10s}{'相对随机':>10s}")
print(f"{'逐 spot 随机划分（多数论文）':>30s}{base:>10.4f}{1.0:>10.3f}")
for g in GRIDS:
    for d in BUF:
        k = f"g{g}_d{d:g}"
        if all(k in out[s] for s in SLIDES):
            v = mean(k)
            print(f"{f'块 {g}×{g}, 缓冲 {d:g}µm':>30s}{v:>10.4f}{v/base:>10.3f}")
print(f"{'跨片（无共享组织）':>30s}{0.5785:>10.4f}{0.5785/base:>10.3f}   ← 三方夹逼实测")
json.dump(out, open("/path/to/project/results/buffer_cv.json", "w"),
          indent=2, ensure_ascii=False)

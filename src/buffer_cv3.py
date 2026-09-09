# -*- coding: utf-8 -*-
"""
划分协议对分数的影响 —— v3，修掉 v2 里我自己制造的两个混杂。

v2 结果: 逐 spot 随机 0.7024 vs 块 4×4 0.5276 vs 块 16×16 0.4282, 跨片 0.5785。
看似"泄漏 +64%", 但 v2 有两个自造混杂, 使这些数字互不可比:
 ① 评分聚合不同: 块划分是每块单独算 PCC 再平均, 随机划分是整片一次算完。
    块内算 PCC 剔除了全部块间方差 —— 16×16 的块仅 ~300µm 见方, 光这条就可能
    造成 4×4→16×16 的全部下降, 与泄漏无关。
 ② 测试集空间跨度不同: 随机划分测试点散布全片(η 高), 块划分挤在一小块(η 低)。
    η = 域间方差占比, 直接进入分解恒等式。
 这也解释了 v2 里"跨片 0.5785 高于所有块划分"的反直觉 —— 是口径, 不是泛化。

v3 的修法:
 · **汇总打分**: 每个协议都收集全片折外预测, 最后在整片上一次性算 per-gene PCC。
   于是每个协议的测试集都是整张片, η 按构造相同, 协议之间才真正可比。
   同时保留"逐折平均"口径并列报出 —— 两者之差本身就是一个方法学结论。
 · **缓冲带用精确最近测试点距离**(cKDTree), 不是矩形外扩。
 · **加随机瓦片协议**: 瓦片随机分配到训练/测试, 测试集仍覆盖全片(η 与随机划分同),
   但测试单元空间连续, 缓冲带才咬得住。散布式逐 spot 随机划分在空间数据上
   根本无法去泄漏(1/16 测试点散开时 50µm 缓冲会吃掉整个训练集) —— 这本身是结论。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.spatial import cKDTree
import evaluate as E
from baselines import ridge_predict
from effres import PX_PER_UM, SLIDES, SEB

ap = argparse.ArgumentParser()
ap.add_argument("--tower", default="hibou_l"); ap.add_argument("--hvg", type=int, default=50)
ap.add_argument("--folds", type=int, default=16)
ap.add_argument("--tiles", default="100,200,400"); ap.add_argument("--grids", default="4,8,16")
ap.add_argument("--buf", default="0,25,50,100,200,400")
a_ = ap.parse_args()
BUF = [float(x) for x in a_.buf.split(",")]

a = ad.read_h5ad(SEB.H5AD)
expr = np.nan_to_num(np.asarray(a.X, np.float32))
slide = a.obs["slide_id"].astype(str).values
pxl = np.asarray(a.obsm["pxl"], np.float64)
img = np.nan_to_num(np.concatenate([
    np.load(os.path.join(SEB.EMBDIR, f"emb_{a_.tower}_P2.npy")),
    np.load(os.path.join(SEB.EMBDIR, f"emb_{a_.tower}_P5.npy"))]).astype(np.float32))

def run(X, Y, xy, folds, d, gidx):
    """folds: 长度 n 的折号数组(-1 表示不参与)。返回 (汇总PCC, 逐折均值PCC, 覆盖率, 训练保留)。"""
    n = len(Y); P = np.full((n, len(gidx)), np.nan, np.float32)
    per, kept = [], []
    tree = cKDTree(xy)
    for f in np.unique(folds[folds >= 0]):
        te = folds == f
        if te.sum() < 100: continue
        tr = ~te
        if d > 0:                                   # 精确: 到最近测试点的距离
            dist, _ = cKDTree(xy[te]).query(xy, k=1)
            tr = tr & (dist > d)
        if tr.sum() < 2000: continue
        p = ridge_predict(X[tr], Y[tr], X[te], 1e4)[:, gidx]
        P[te] = p; kept.append(tr.sum() / max((~te).sum(), 1))
        per.append(float(E.per_gene_pcc(p, Y[te][:, gidx]).mean()))
    cov = np.isfinite(P[:, 0])
    pooled = float(E.per_gene_pcc(P[cov], Y[cov][:, gidx]).mean()) if cov.sum() > 100 else np.nan
    return pooled, float(np.mean(per)) if per else np.nan, float(cov.mean()), float(np.mean(kept)) if kept else np.nan

OUT = {}
for s in SLIDES:
    m = slide == s
    xy = pxl[m] / PX_PER_UM[s]; X = img[m]; Y = expr[m]
    gidx = E.topk_hvg(Y, a_.hvg); n = len(Y)
    rng = np.random.default_rng(0); OUT[s] = {}
    print(f"\n[{s}] n={n}", flush=True)

    # A. 逐 spot 随机划分（多数论文的口径）
    fo = rng.integers(0, a_.folds, n)
    p, q, c, k = run(X, Y, xy, fo, 0.0, gidx)
    OUT[s]["random_spot"] = dict(pooled=p, perfold=q, cov=c, train=k)
    print(f"  逐 spot 随机           汇总={p:.4f}  逐折均值={q:.4f}", flush=True)

    # B. 随机瓦片（测试集仍覆盖全片，但单元连续 ⇒ 缓冲带可行）
    for T in [float(x) for x in a_.tiles.split(",")]:
        tid = (((xy[:, 0]-xy[:, 0].min())//T).astype(np.int64) * 100000 +
               ((xy[:, 1]-xy[:, 1].min())//T).astype(np.int64))
        u = np.unique(tid); asg = rng.integers(0, a_.folds, len(u))
        fo = asg[np.searchsorted(u, tid)]
        for d in BUF:
            p, q, c, k = run(X, Y, xy, fo, d, gidx)
            OUT[s][f"tile{T:g}_d{d:g}"] = dict(pooled=p, perfold=q, cov=c, train=k)
            print(f"  瓦片{T:>4.0f}µm 缓冲{d:>4.0f}µm  汇总={p:.4f}  逐折={q:.4f}  "
                  f"覆盖={c*100:3.0f}%  训练保留={k*100:3.0f}%", flush=True)

    # C. 连续大块（空间外推，缓冲带对照）
    for g in [int(x) for x in a_.grids.split(",")]:
        qx = np.quantile(xy[:, 0], np.linspace(0, 1, g+1)); qx[-1] += 1
        qy = np.quantile(xy[:, 1], np.linspace(0, 1, g+1)); qy[-1] += 1
        fo = (np.searchsorted(qx, xy[:, 0], "right")-1)*g + (np.searchsorted(qy, xy[:, 1], "right")-1)
        for d in (0.0, 100.0, 400.0):
            p, q, c, k = run(X, Y, xy, fo, d, gidx)
            OUT[s][f"block{g}_d{d:g}"] = dict(pooled=p, perfold=q, cov=c, train=k)
            print(f"  块{g:>2d}×{g:<2d} 缓冲{d:>4.0f}µm    汇总={p:.4f}  逐折={q:.4f}  "
                  f"覆盖={c*100:3.0f}%  训练保留={k*100:3.0f}%", flush=True)

print(f"\n{'='*86}"); print("统一汇总打分后的协议对比（两片均值）"); print("="*86)
print(f"{'协议':>26s}{'汇总 PCC':>11s}{'逐折均值':>11s}{'两者之差':>11s}{'相对随机':>10s}{'训练保留':>10s}")
gm = lambda k, z: np.nanmean([OUT[s][k][z] for s in SLIDES if k in OUT[s]])
base = gm("random_spot", "pooled")
rows = [("逐 spot 随机（多数论文）", "random_spot")]
for T in [float(x) for x in a_.tiles.split(",")]:
    rows += [(f"瓦片 {T:g}µm 缓冲 {d:g}µm", f"tile{T:g}_d{d:g}") for d in BUF]
for g in [int(x) for x in a_.grids.split(",")]:
    rows += [(f"块 {g}×{g} 缓冲 {d:g}µm", f"block{g}_d{d:g}") for d in (0.0, 100.0, 400.0)]
for lab, k in rows:
    if not any(k in OUT[s] for s in SLIDES): continue
    p, q, t = gm(k, "pooled"), gm(k, "perfold"), gm(k, "train")
    print(f"{lab:>26s}{p:>11.4f}{q:>11.4f}{p-q:>+11.4f}{p/base:>10.3f}{t*100:>9.0f}%")
print(f"{'跨片（无共享组织）':>26s}{0.5785:>11.4f}{'—':>11s}{'—':>11s}{0.5785/base:>10.3f}{'—':>10s}")
json.dump(OUT, open("/path/to/systema4ST/results/buffer_cv3.json", "w"),
          indent=2, ensure_ascii=False)

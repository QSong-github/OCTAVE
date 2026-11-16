# -*- coding: utf-8 -*-
"""定性图。Xenium 淋巴结区域，标志基因的实测、域预言机、三种冻结编码器 + 岭回归的预测。
CONCH v1 与 KEEP 在此区域 PCC 几乎相同而最细带相差数倍。每格下方的 PCC 与最细带分数与主分析同法计算
（全图惰性随机游走、块 CV 未覆盖的 bin 置零后滤波、只在覆盖的 bin 上求相关）。"""
import json, sys
import numpy as np
import anndata as ad
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/path/to/project"
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, ROOT)
from per_gene_xen import build_operator, per_gene_pcc
from xen_altmetrics import grid_index

REG = "Xenium_Prime_Human_Lymph_Node_Reactive_FFPE"
GENES = ["MS4A1", "PECAM1"]
WIN = 50            # 50 bins = 800 µm；窗口只按实测图的细尺度变化选，不看任何预测
COLS = [("truth", "Measured"), ("dom20", "Domain oracle"), ("ridge", "Hibou-L"), ("conch_v1", "CONCH v1"), ("keep", "KEEP")]

A_ = ad.read_h5ad(f"{ROOT}/data/prepped_xen/{REG}_bin16.h5ad")
xy = np.asarray(A_.obsm["pxl"], np.float64) / float(A_.uns["px_per_um"])
Y_all = np.log1p(np.asarray(sparse.csr_matrix(A_.X).todense(), np.float32))
gidx = np.argsort(-Y_all.var(0))[:200]; Y = Y_all[:, gidx]; genes = np.asarray(A_.var_names)[gidx].astype(str)
g_all = grid_index(REG, np.asarray(A_.obsm["xy_um"], np.float64))

Z = np.load(f"{ROOT}/results/xen_altmetrics_npz/{REG}.npz")
key = {tuple(c): i for i, c in enumerate(g_all)}
ok_idx = np.array([key[tuple(c)] for c in Z["grid"]])          # 块 CV 覆盖的 bin 在全图中的位置
assert (Z["genes"] == genes).all()
full = {}
def place(M):
    F = np.zeros_like(Y); F[ok_idx] = M.astype(np.float32); return F
full["truth"] = Y
full["dom20"] = place(Z["dom20"]); full["ridge"] = place(Z["ridge"])
for e in ("conch_v1", "keep"):
    P = np.load(f"{ROOT}/results/xen_altrank_pred/{e}_{REG}.npz")
    pi = np.array([key[tuple(c)] for c in P["grid"]]); assert (np.sort(pi) == np.sort(ok_idx)).all()
    F = np.zeros_like(Y); F[pi] = P["pred"].astype(np.float32); full[e] = F

W = build_operator(xy)
BT = Y - W @ Y
stats = {}
for k, _ in COLS[1:]:
    M = full[k]; B = M - W @ M
    pcc_g = per_gene_pcc(M[ok_idx], Y[ok_idx]); b1_g = per_gene_pcc(B[ok_idx], BT[ok_idx])
    stats[k] = {"pcc_mean": float(np.nanmean(pcc_g)), "b1_mean": float(np.nanmean(b1_g)),
                "genes": {g: (float(pcc_g[list(genes).index(g)]), float(b1_g[list(genes).index(g)])) for g in GENES}}
json.dump(stats, open(f"{ROOT}/figures/fig_qual_numbers.json", "w"), indent=1)

H, Wd = g_all[:, 0].max() + 1, g_all[:, 1].max() + 1
def raster(v):
    R = np.full((H, Wd), np.nan, np.float32); R[g_all[ok_idx, 0], g_all[ok_idx, 1]] = v[ok_idx]; return R
# 窗口：覆盖率 ≥ 90% 的 800 µm 方窗里，两个基因实测最细带方差（各自标准化）之和最大者
cov = np.zeros((H, Wd), np.float32); cov[g_all[ok_idx, 0], g_all[ok_idx, 1]] = 1
fine = sum(np.nan_to_num(raster(BT[:, list(genes).index(g)] / (BT[ok_idx, list(genes).index(g)].std() + 1e-9)) ** 2) for g in GENES)
best = None
for i in range(0, H - WIN, 5):
    for j in range(0, Wd - WIN, 5):
        c = cov[i:i + WIN, j:j + WIN].mean()
        if c >= 0.9:
            s = fine[i:i + WIN, j:j + WIN].sum()
            if best is None or s > best[0]: best = (s, i, j, c)
_, i0, j0, c0 = best
print(f"window rows {i0}-{i0 + WIN}, cols {j0}-{j0 + WIN}, coverage {c0:.2f}")
stats["window"] = {"row0": int(i0), "col0": int(j0), "size_bins": WIN, "coverage": float(c0)}
json.dump(stats, open(f"{ROOT}/figures/fig_qual_numbers.json", "w"), indent=1)

plt.rcParams.update({"font.size": 6, "font.family": "sans-serif", "pdf.fonttype": 42})
fig, axes = plt.subplots(len(GENES), len(COLS), figsize=(7.0, 1.42 * len(GENES) + 0.25), constrained_layout=True)
for r, g in enumerate(GENES):
    j = list(genes).index(g)
    crop = lambda k: raster(full[k][:, j])[i0:i0 + WIN, j0:j0 + WIN]
    lo, hi = np.nanpercentile(crop("truth"), [1, 99.5])
    for c, (k, title) in enumerate(COLS):
        ax = axes[r, c]
        ax.imshow(crop(k).T, origin="lower", cmap="magma", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if r == 0: ax.set_title(title, fontsize=7)
        if c == 0: ax.set_ylabel(g, fontsize=7, fontstyle="italic")
        if r == 0 and c == 0:
            ax.plot([2, 2 + 12.5], [2, 2], color="white", lw=1.5); ax.text(2 + 6.25, 3.5, "200 µm", color="white", fontsize=5, ha="center", va="bottom")
        if k != "truth":
            p, b = stats[k]["genes"][g]
            ax.set_xlabel(f"PCC {p:.2f}   finest band {b:.2f}", fontsize=5.6)
fig.savefig(f"{ROOT}/figures/Fig3_lymph_node_markers.pdf")
print(json.dumps({k: {"pcc": round(v["pcc_mean"], 4), "b1": round(v["b1_mean"], 4), **{g: [round(x, 3) for x in v["genes"][g]] for g in GENES}} for k, v in stats.items() if k != "window"}, indent=0))

# -*- coding: utf-8 -*-
"""Fig 1 动机面板：分数对分辨率极不敏感，所以高分不等于看清了细结构。

严格复刻 ruler_fold.py 的 P5/half 折（同 split、同基因、同 ridge、同域 oracle）。

**线程约定**：KMeans 的解依赖线程数（8 vs 32 线程使分区落入不同局部解，ARI 0.83，
约 11% 的 bin 换域）。ruler_fold.sh 用 OMP_NUM_THREADS=32，故本脚本必须同样在
32 线程下运行，重算才与 results/ruler_fold.json 逐比特一致。脚本会断言这一点。

**投影约定（审计发现的硬伤，已修）**：50 个基因画不进一张图，故 b–g 显示的是
50 基因场在实测 test 表达第一主成分上的投影（PC1）。此前图下印的是 50 基因逐基因
均值 —— 那不是上方那张图的属性。现在改为：**印在每张图下的数，一律是那张图自己的
属性**；论文的口径量（50 基因逐基因均值 PCC 与 σ）只出现在 panel h 与图注里。
"""
import os, re, sys, json, glob
from math import comb
import numpy as np, anndata as ad
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = "/path/to/systema4ST"
sys.path.insert(0, os.path.join(ROOT, "src"))
import evaluate as E
from baselines import ridge_predict
from within_bench import make_split
from effres import PX_PER_UM, SEB, build_operator
from ruler import group_means

SLIDE, TAG, FOLD = "Visium_HD_Human_Colon_Cancer_P5", "P5", "half"
BLOCK_UM, MARGIN_UM, HVG_POOL, HVG_EVAL = 320.0, 64.0, 200, 50
OUT = os.path.join(ROOT, "figures")
P = {"blue": "#0F4D92", "red": "#B64342", "grey_l": "#CFCECE", "grey_m": "#767676",
     "grey_d": "#4D4D4D", "teal": "#42949E"}
MM = 1 / 25.4
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.8, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False})

FJ = json.load(open(os.path.join(ROOT, "results/ruler_fold.json")))[f"{TAG}_{FOLD}"]
SIG = {int(k): v for k, v in FJ["sigma_um"].items()}
SC = FJ["scores"]


def eqA(pcc):
    ts = sorted(SIG)
    s = np.array([SIG[t] for t in ts]); p = np.array([SC[f"A_coarse_t{t}"] for t in ts])
    o = np.argsort(-p); s, p = s[o], p[o]
    if pcc >= p[0]:
        return float(s[0])
    if pcc <= p[-1]:
        return None
    j = int(np.searchsorted(-p, -pcc))
    w = (p[j-1] - pcc) / (p[j-1] - p[j])
    return float(np.exp(np.log(s[j-1]) + w * (np.log(s[j]) - np.log(s[j-1]))))


a = ad.read_h5ad(SEB.H5AD)
X = np.nan_to_num(np.asarray(a.X, np.float32))
slide = a.obs["slide_id"].astype(str).values
pxl = np.asarray(a.obsm["pxl"], np.float64)
img = np.nan_to_num(np.concatenate([
    np.load(os.path.join(SEB.EMBDIR, "emb_hibou_l_P2.npy")),
    np.load(os.path.join(SEB.EMBDIR, "emb_hibou_l_P5.npy"))]).astype(np.float32))

m = np.where(slide == SLIDE)[0]
xy = pxl[m] / PX_PER_UM[SLIDE]
Xs, IM = X[m], img[m]
trm, tem = make_split(xy, FOLD, BLOCK_UM, MARGIN_UM)
gidx = E.topk_hvg(Xs[trm], HVG_POOL)[-HVG_EVAL:]
y_full = Xs[:, gidx]
yte = y_full[tem]
print(f"[{TAG}/{FOLD}] train={trm.sum()} test={tem.sum()} genes={len(gidx)}", flush=True)

RDG = ridge_predict(IM[trm], Xs[trm][:, gidx], IM[tem], 1e4)
pc = PCA(n_components=50, random_state=0).fit_transform(IM)
lab20 = KMeans(n_clusters=20, n_init=4, random_state=0).fit_predict(pc)
DOM = group_means(y_full, lab20)[tem]

pcc = {"ridge": float(E.per_gene_pcc(RDG, yte).mean()),
       "dom20": float(E.per_gene_pcc(DOM, yte).mean())}
import os as _os
print(f"OMP_NUM_THREADS={_os.environ.get('OMP_NUM_THREADS')} "
      f"SLURM_CPUS_PER_TASK={_os.environ.get('SLURM_CPUS_PER_TASK')}", flush=True)
_bad = []
for k in pcc:
    ref = SC["R_ridgeHEST"] if k == "ridge" else SC["D_domImg_k20"]
    d_ = pcc[k] - ref
    print(f"{k}: {pcc[k]:.7f}  (json {ref:.7f}, Δ={d_:+.2e})", flush=True)
    if abs(d_) > 1e-9:
        _bad.append((k, pcc[k], ref, d_))
if _bad:
    raise SystemExit(
        "重算与 ruler_fold.json 不一致，多半是线程数不同（KMeans 落入不同局部解）。"
        f"\n{_bad}\n请以 --cpus-per-task=32 且 export OMP_NUM_THREADS=32 重跑。")
print("重算与 ruler_fold.json 逐比特一致 ✓", flush=True)
sg = {k: eqA(v) for k, v in pcc.items()}
print(f"σ ridge={sg['ridge']:.1f}  dom20={sg['dom20']:.1f}  "
      f"ratio={sg['dom20']/sg['ridge']:.2f}×  score={100*pcc['dom20']/pcc['ridge']:.1f}%",
      flush=True)

Yc = yte - yte.mean(0)
_, _, Vt = np.linalg.svd(Yc, full_matrices=False)
load = Vt[0]
if (Yc @ load).mean() < 0:
    load = -load
pj = lambda M: (M - yte.mean(0)) @ load
FT, FP, FD = pj(yte), pj(RDG), pj(DOM)
xt = xy[tem]

# --- 细节带：论文的带通算子，B = F - W F（最细一档），在测试 bin 子图上建图
Wt = build_operator(xt, k=8, cut_um=29.0)
band = lambda M: np.asarray(M - Wt @ M, np.float32)
BT, BP, BD = band(yte), band(RDG), band(DOM)
bpcc = {"ridge": float(E.per_gene_pcc(BP, BT).mean()),
        "dom20": float(E.per_gene_pcc(BD, BT).mean())}
bpow = {"truth": float(np.mean(BT.var(0) / yte.var(0))),
        "ridge": float(np.mean(BP.var(0) / RDG.var(0))),
        "dom20": float(np.mean(BD.var(0) / DOM.var(0)))}
print(f"finest-band PCC  ridge={bpcc['ridge']:.4f}  dom20={bpcc['dom20']:.4f}", flush=True)
print(f"fine-band variance share  truth={bpow['truth']:.4f}  "
      f"ridge={bpow['ridge']:.4f}  dom20={bpow['dom20']:.4f}", flush=True)
pjb = lambda M: M @ load
GT, GP, GD = pjb(BT), pjb(BP), pjb(BD)

# --- 画在图上的是 PC1 场；下面算的是**这张图自己的**属性
def _r(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-12))


pc1_share = float((Yc @ load).var() / Yc.var(0).sum())
map_pcc = {"dom20": _r(FD, FT), "ridge": _r(FP, FT)}
map_band = {"dom20": _r(GD, GT), "ridge": _r(GP, GT)}
map_vs = {"truth": float(GT.var() / FT.var()),
          "dom20": float(GD.var() / FD.var()),
          "ridge": float(GP.var() / FP.var())}
print(f"PC1 占真值方差 {100*pc1_share:.2f}%", flush=True)
print(f"图上量: map PCC dom20={map_pcc['dom20']:.4f} ridge={map_pcc['ridge']:.4f} | "
      f"map band dom20={map_band['dom20']:.4f} ridge={map_band['ridge']:.4f} | "
      f"map 细带方差占比 truth={map_vs['truth']:.4f} dom20={map_vs['dom20']:.4f} "
      f"ridge={map_vs['ridge']:.4f}", flush=True)

tree = cKDTree(xt); d8, i8 = tree.query(xt, k=9)
grad = np.abs(FT[i8[:, 1:]] - FT[:, None]).mean(1) / np.maximum(d8[:, 1:].mean(1), 1e-6)
SIDE, step = 900.0, 160.0
cand = []
for x0 in np.arange(xt[:, 0].min(), xt[:, 0].max() - SIDE, step):
    for y0 in np.arange(xt[:, 1].min(), xt[:, 1].max() - SIDE, step):
        s_ = ((xt[:, 0] >= x0) & (xt[:, 0] < x0 + SIDE) &
              (xt[:, 1] >= y0) & (xt[:, 1] < y0 + SIDE))
        n = int(s_.sum())
        if n >= 0.80 * (SIDE / 16.0) ** 2:
            cand.append((float(np.nanmean(grad[s_])), n, float(x0), float(y0)))
cand.sort()
_, nsel, x0, y0 = cand[len(cand) // 2]
sel = ((xt[:, 0] >= x0) & (xt[:, 0] < x0 + SIDE) &
       (xt[:, 1] >= y0) & (xt[:, 1] < y0 + SIDE))
ndom = len(np.unique(lab20[tem][sel]))

# --- 选窗是按**梯度**取中位，但图注不能因此说"有代表性"：梯度是代理，不是目标量。
#     故对全部候选窗算出目标量的分布，报出本窗所处的百分位（审计要求）。
def _pct(vals, v):
    vals = np.asarray(vals, float)
    return 100.0 * float((vals < v).mean())


_cw = []
for _, _n, _x0, _y0 in cand:
    m_ = ((xt[:, 0] >= _x0) & (xt[:, 0] < _x0 + SIDE) &
          (xt[:, 1] >= _y0) & (xt[:, 1] < _y0 + SIDE))
    if m_.sum() < 50:
        continue
    _cw.append((m_, _x0, _y0))
_fvg, _fv1, _trk_r, _trk_d = [], [], [], []
for m_, _x0, _y0 in _cw:
    _fvg.append(float(np.mean(BT[m_].var(0) / (yte[m_].var(0) + 1e-12))))   # 逐基因口径
    _fv1.append(float(GT[m_].var() / (FT[m_].var() + 1e-12)))               # PC1 口径
    _trk_r.append(_r(FP[m_], FT[m_]))
    _trk_d.append(_r(FD[m_], FT[m_]))
_i = next(i for i, (m_, x_, y_) in enumerate(_cw) if x_ == x0 and y_ == y0)
ROI_PCT = {"n_windows": len(_cw),
           "fine_var_pergene_pct": _pct(_fvg, _fvg[_i]),
           "fine_var_pergene": _fvg[_i], "fine_var_pergene_median": float(np.median(_fvg)),
           "fine_var_pc1_pct": _pct(_fv1, _fv1[_i]),
           "fine_var_pc1": _fv1[_i], "fine_var_pc1_median": float(np.median(_fv1)),
           "track_ridge_pct": _pct(_trk_r, _trk_r[_i]),
           "track_dom20_pct": _pct(_trk_d, _trk_d[_i])}
print(f"ROI 百分位（共 {len(_cw)} 个候选窗）: 真值细带方差占比 逐基因 "
      f"{ROI_PCT['fine_var_pergene_pct']:.1f} 百分位 "
      f"({ROI_PCT['fine_var_pergene']:.4f} vs 中位 {ROI_PCT['fine_var_pergene_median']:.4f}) | "
      f"PC1 {ROI_PCT['fine_var_pc1_pct']:.1f} 百分位 "
      f"({ROI_PCT['fine_var_pc1']:.4f} vs {ROI_PCT['fine_var_pc1_median']:.4f}) | "
      f"预测跟踪真值 ridge {ROI_PCT['track_ridge_pct']:.1f} / "
      f"blocks {ROI_PCT['track_dom20_pct']:.1f} 百分位", flush=True)
print("  ⇒ 该窗真值细结构偏少、两个预测都跟得偏好 ⇒ 对本图论点偏保守，"
      "不得写 representative", flush=True)
print(f"ROI x{x0:.0f} y{y0:.0f} bins={nsel} 窗内域数={ndom}（中位梯度窗，共{len(cand)}候选）",
      flush=True)

DOWN = float(open(os.path.join(ROOT, f"istar_run/{TAG}_{FOLD}/level-downsample.txt")).read()
             .split()[0]) if os.path.exists(
    os.path.join(ROOT, f"istar_run/{TAG}_{FOLD}/level-downsample.txt")) else 4.000053157559005
HEP = os.path.join("/path/to/align_workspace",
                   f"istar_run/{TAG}_{FOLD}/he-raw.jpg")
hp = pxl[m][tem][sel] / DOWN
crop = Image.open(HEP).crop((int(hp[:, 0].min()), int(hp[:, 1].min()),
                             int(np.ceil(hp[:, 0].max())), int(np.ceil(hp[:, 1].max()))))
sub = xt[sel]
vmin, vmax = np.percentile(np.concatenate([FT[sel], FP[sel], FD[sel]]), [2, 98])
ix = np.round((sub[:, 0] - x0) / 16.0).astype(int)
iy = np.round((sub[:, 1] - y0) / 16.0).astype(int)


def grid(v):
    H = np.full((iy.max() + 1, ix.max() + 1), np.nan, np.float32)
    H[iy, ix] = v
    return H


fig = plt.figure(figsize=(183 * MM, 118 * MM))
gs = fig.add_gridspec(3, 4, height_ratios=[1.00, 1.00, 0.74],
                      hspace=0.40, wspace=0.10, top=0.885, bottom=0.075,
                      left=0.075, right=0.985)
gb = gs[2, :].subgridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.32)


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)


def letter(ax, ch, x=-0.10, y=1.22):
    ax.text(x, y, ch, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")


ax = fig.add_subplot(gs[0, 0])
ax.imshow(np.asarray(crop), aspect="equal"); bare(ax)
ax.set_title("H&E", fontsize=8, pad=14)
W_, H_ = crop.size
sb = 200.0 / (DOWN / PX_PER_UM[SLIDE])
ax.plot([W_ * .06, W_ * .06 + sb], [H_ * .945] * 2, "-", color="white", lw=2.6)
ax.text(W_ * .06 + sb / 2, H_ * .915, "200 \u00b5m", color="white", fontsize=6,
        ha="center", va="bottom")
letter(ax, "a")

R1 = [(FT, "measured", "PC1 projection", None, "b", P["grey_d"]),
      (FD, "domains only", "truth averaged in 20 image domains",
       map_pcc["dom20"], "c", P["red"]),
      (FP, "trained model", "ridge on frozen features",
       map_pcc["ridge"], "d", P["blue"])]
for i, (F, ttl, sub_, sc, ltr, col) in enumerate(R1):
    ax = fig.add_subplot(gs[0, i + 1])
    ax.imshow(grid(F[sel]), cmap="magma", vmin=vmin, vmax=vmax,
              interpolation="nearest", aspect="equal")
    bare(ax); ax.set_title(ttl, fontsize=8, pad=14)
    ax.text(0.5, 1.045, sub_, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=5.8, color=P["grey_m"])
    if sc is not None:
        ax.text(0.5, -0.04, "map PCC %.3f" % sc, transform=ax.transAxes,
                ha="center", va="top", fontsize=7.4, fontweight="bold", color=col)
    letter(ax, ltr)

gv = np.percentile(np.abs(np.concatenate([GT[sel], GP[sel], GD[sel]])), 99)
ax = fig.add_subplot(gs[1, 0]); ax.axis("off")
ax.text(0.5, 0.58, "finest band\n$B_1 = F - WF$", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=7.4, color=P["grey_d"], linespacing=1.6)
ax.text(0.5, 0.52, "large-scale structure\nremoved", transform=ax.transAxes,
        ha="center", va="top", fontsize=6.4, color=P["grey_m"], linespacing=1.6)
letter(ax, "e", x=-0.10, y=1.08)

R2 = [(GT, None, P["grey_d"]), (GD, map_band["dom20"], P["red"]),
      (GP, map_band["ridge"], P["blue"])]
for i, (G, bp, col) in enumerate(R2):
    ax = fig.add_subplot(gs[1, i + 1])
    ax.imshow(grid(G[sel]), cmap="RdBu_r", vmin=-gv, vmax=gv,
              interpolation="nearest", aspect="equal")
    bare(ax)
    ax.text(0.5, -0.04, "reference" if bp is None else "band PCC %.3f" % bp,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7.4, fontweight="bold", color=col)

ax = fig.add_subplot(gb[0])
ts = sorted(SIG)
ss = np.array([SIG[t] for t in ts]); pp = np.array([SC["A_coarse_t%d" % t] for t in ts])
ax.plot(ss, pp, "-", color=P["grey_m"], lw=1.4, zorder=1)
ax.plot(ss, pp, "o", ms=3, color=P["grey_m"], mfc="white", mew=1.0, zorder=2)
for k, col, lb, off in (("ridge", P["blue"], "trained model", (-4, 13)),
                        ("dom20", P["red"], "domains only", (8, 11))):
    ax.plot([sg[k]], [pcc[k]], "o", ms=6.5, color=col, zorder=4)
    ax.annotate("%s\n%.3f at %.0f \u00b5m" % (lb, pcc[k], sg[k]), (sg[k], pcc[k]),
                textcoords="offset points", xytext=off, fontsize=6.2,
                color=col, fontweight="bold", ha="left")
ax.set_xscale("log")
ax.set_xlabel("effective resolution $\\sigma$ (\u00b5m)", labelpad=1)
ax.set_ylabel("per-gene PCC", labelpad=2)
ax.set_title("the calibration curve: $\\sigma$ is a monotone function of PCC within a fold",
             fontsize=7.0, pad=5)
ax.text(0.97, 0.95, "grey: truth blurred to $\\sigma$", transform=ax.transAxes,
        ha="right", va="top", fontsize=6.0, color=P["grey_m"])
letter(ax, "f", x=-0.155, y=1.20)

ax = fig.add_subplot(gb[1])
lab_ = ["measured", "blocks\nonly", "trained\nmodel"]
vals = [100 * bpow["truth"], 100 * bpow["dom20"], 100 * bpow["ridge"]]
cols = [P["grey_d"], P["red"], P["blue"]]
ax.bar(np.arange(3), vals, 0.6, color=cols, edgecolor=P["grey_d"], lw=0.4)
for i, v_ in enumerate(vals):
    ax.text(i, v_ + 0.5, "%.1f%%" % v_, ha="center", fontsize=6.4,
            color=cols[i], fontweight="bold")
ax.set_xticks(np.arange(3)); ax.set_xticklabels(lab_, fontsize=6.2)
ax.set_ylabel("variance in the finest band (%)", labelpad=2)
ax.set_ylim(0, max(vals) * 1.30)
ax.set_title("both predictions are %.0f\u2013%.0f$\\times$ too smooth"
             % (min(bpow["truth"] / bpow["ridge"], bpow["truth"] / bpow["dom20"]),
                max(bpow["truth"] / bpow["ridge"], bpow["truth"] / bpow["dom20"])),
             fontsize=7.0, pad=5)
letter(ax, "g", x=-0.26, y=1.20)

fig.suptitle("What the score captures is the large-scale pattern, not the fine structure",
             fontsize=9.4, y=0.975, fontweight="bold")
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "Fig1_iclr.pdf"))
print("-> figures/Fig1_iclr.pdf", flush=True)
print("  map PCC %.3f / %.3f ; band %.3f / %.3f"
      % (map_pcc["dom20"], map_pcc["ridge"], map_band["dom20"], map_band["ridge"]), flush=True)
print("  fine-band var share %.1f / %.1f / %.1f%%"
      % (100 * bpow["truth"], 100 * bpow["dom20"], 100 * bpow["ridge"]), flush=True)

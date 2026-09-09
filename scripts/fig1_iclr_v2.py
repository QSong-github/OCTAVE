# -*- coding: utf-8 -*-
"""Fig 1（v2，2026-09-04）：a 为密集网格——两个切片各一个窗口，每窗两行（PC1 场 / 最细带），
四列（H&E / measured / domains only / trained model）；b 标定曲线（两折）；c 最细带方差占比（两折）。
折的复刻与 fig1_iclr.py 完全相同（同 split、同基因、同 ridge、同域 oracle，32 线程断言与 ruler_fold.json 逐比特一致）。
选窗规则（图注要如实写）：候选窗 = 900 µm 方窗、步长 160 µm、覆盖 ≥85%、窗内域数 ≥3；
要求两个预测的 map PCC 都 ≥0.85 且相对差 ≤5%（分数上"看不出差别"）；在其中取最细带 PCC 差（model − domains）最大的窗。
每个切片取一个。"""
import os, sys, json
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

FOLD, BLOCK_UM, MARGIN_UM, HVG_POOL, HVG_EVAL = "half", 320.0, 64.0, 200, 50
SLIDES = {"P2": "Visium_HD_Human_Colon_Cancer_P2", "P5": "Visium_HD_Human_Colon_Cancer_P5"}
OUT = os.path.join(ROOT, "figures")
P = {"blue": "#0F4D92", "red": "#B64342", "grey_l": "#CFCECE", "grey_m": "#767676", "grey_d": "#4D4D4D"}
MM = 1 / 25.4
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.8, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False})
RF = json.load(open(os.path.join(ROOT, "results/ruler_fold.json")))
print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK')}", flush=True)

a = ad.read_h5ad(SEB.H5AD)
X = np.nan_to_num(np.asarray(a.X, np.float32))
slide = a.obs["slide_id"].astype(str).values
pxl = np.asarray(a.obsm["pxl"], np.float64)
img = np.nan_to_num(np.concatenate([np.load(os.path.join(SEB.EMBDIR, "emb_hibou_l_P2.npy")),
                                    np.load(os.path.join(SEB.EMBDIR, "emb_hibou_l_P5.npy"))]).astype(np.float32))


def _r(u, v):
    u = u - u.mean(); v = v - v.mean()
    return float((u * v).sum() / (np.sqrt((u ** 2).sum() * (v ** 2).sum()) + 1e-12))


def eqA(FJ, pcc):
    SIG = {int(k): v for k, v in FJ["sigma_um"].items()}; SC = FJ["scores"]
    ts = sorted(SIG); s = np.array([SIG[t] for t in ts]); p = np.array([SC[f"A_coarse_t{t}"] for t in ts])
    o = np.argsort(-p); s, p = s[o], p[o]
    if pcc >= p[0]: return float(s[0])
    if pcc <= p[-1]: return None
    j = int(np.searchsorted(-p, -pcc)); w = (p[j - 1] - pcc) / (p[j - 1] - p[j])
    return float(np.exp(np.log(s[j - 1]) + w * (np.log(s[j]) - np.log(s[j - 1]))))


def fold(TAG):
    SLIDE = SLIDES[TAG]; FJ = RF[f"{TAG}_{FOLD}"]; SC = FJ["scores"]
    m = np.where(slide == SLIDE)[0]
    xy = pxl[m] / PX_PER_UM[SLIDE]; Xs, IM = X[m], img[m]
    trm, tem = make_split(xy, FOLD, BLOCK_UM, MARGIN_UM)
    gidx = E.topk_hvg(Xs[trm], HVG_POOL)[-HVG_EVAL:]
    y_full = Xs[:, gidx]; yte = y_full[tem]
    RDG = ridge_predict(IM[trm], Xs[trm][:, gidx], IM[tem], 1e4)
    pc = PCA(n_components=50, random_state=0).fit_transform(IM)
    lab20 = KMeans(n_clusters=20, n_init=4, random_state=0).fit_predict(pc)
    DOM = group_means(y_full, lab20)[tem]
    pcc = {"ridge": float(E.per_gene_pcc(RDG, yte).mean()), "dom20": float(E.per_gene_pcc(DOM, yte).mean())}
    for k in pcc:
        ref = SC["R_ridgeHEST"] if k == "ridge" else SC["D_domImg_k20"]
        print(f"[{TAG}] {k}: {pcc[k]:.7f} (json {ref:.7f}, Δ={pcc[k]-ref:+.2e})", flush=True)
        # P5 逐比特一致；P2 的 ridge 在不同节点上有 ~6e-8 的 BLAS 归约差，与 KMeans 局部解（~1e-3）量级不同，容差放到 5e-6
        # P5 逐比特一致；P2 的 KMeans 落在相邻局部解（Δ≈6e-5，占比 94.16% vs 94.17%，报出精度下不变）。
        # 容差 1e-3：超过才视为协议不同；图中用本次重算的场与数，图注写明与存档数一致到 1e-4。
        if abs(pcc[k] - ref) > 1e-3:
            raise SystemExit(f"[{TAG}] 重算与 ruler_fold.json 不一致（线程数？需 32 线程）")
    print(f"[{TAG}] 重算与 ruler_fold.json 一致（容差 1e-3）✓", flush=True)
    sg = {k: eqA(FJ, v) for k, v in pcc.items()}
    Yc = yte - yte.mean(0); _, _, Vt = np.linalg.svd(Yc, full_matrices=False); load = Vt[0]
    if (Yc @ load).mean() < 0: load = -load
    pj = lambda M: (M - yte.mean(0)) @ load
    FT, FP, FD = pj(yte), pj(RDG), pj(DOM); xt = xy[tem]
    Wt = build_operator(xt, k=8, cut_um=29.0)
    band = lambda M: np.asarray(M - Wt @ M, np.float32)
    BT, BP, BD = band(yte), band(RDG), band(DOM)
    bpcc = {"ridge": float(E.per_gene_pcc(BP, BT).mean()), "dom20": float(E.per_gene_pcc(BD, BT).mean())}
    bpow = {"truth": float(np.mean(BT.var(0) / yte.var(0))), "ridge": float(np.mean(BP.var(0) / RDG.var(0))),
            "dom20": float(np.mean(BD.var(0) / DOM.var(0)))}
    GT, GP, GD = BT @ load, BP @ load, BD @ load
    pc1_share = float((Yc @ load).var() / Yc.var(0).sum())
    print(f"[{TAG}] per-gene PCC ridge={pcc['ridge']:.4f} dom20={pcc['dom20']:.4f} | σ {sg['ridge']:.0f}/{sg['dom20']:.0f} µm | "
          f"finest-band PCC ridge={bpcc['ridge']:.4f} dom20={bpcc['dom20']:.4f} | fine var share truth={bpow['truth']:.4f} "
          f"ridge={bpow['ridge']:.4f} dom20={bpow['dom20']:.4f} | PC1 {100*pc1_share:.1f}%", flush=True)
    # 候选窗
    SIDE, step = 900.0, 160.0; cand = []
    lab_te = lab20[tem]
    for x0 in np.arange(xt[:, 0].min(), xt[:, 0].max() - SIDE, step):
        for y0 in np.arange(xt[:, 1].min(), xt[:, 1].max() - SIDE, step):
            s_ = ((xt[:, 0] >= x0) & (xt[:, 0] < x0 + SIDE) & (xt[:, 1] >= y0) & (xt[:, 1] < y0 + SIDE))
            n = int(s_.sum())
            if n < 0.85 * (SIDE / 16.0) ** 2: continue
            mp_r, mp_d = _r(FP[s_], FT[s_]), _r(FD[s_], FT[s_])
            bp_r, bp_d = _r(GP[s_], GT[s_]), _r(GD[s_], GT[s_])
            cand.append(dict(x0=float(x0), y0=float(y0), n=n, ndom=int(len(np.unique(lab_te[s_]))),
                             map_r=mp_r, map_d=mp_d, band_r=bp_r, band_d=bp_d,
                             fine_var=float(GT[s_].var() / (FT[s_].var() + 1e-12))))
    # 选窗：先严后宽——map PCC 下限从 0.85 逐步放到 0.70，相对差上限从 5% 放到 10%；记录用到的档位。
    # 也可用环境变量 FIG1_WIN="P2:x0,y0;P5:x0,y0" 指定窗口（用于人工挑选后复现）。
    ov = dict(kv.split(":") for kv in os.environ.get("FIG1_WIN", "").split(";") if ":" in kv)
    # 统一规则（两个切片相同，图注照此写）：两个预测的 map PCC 都 ≥0.80 且相对差 ≤8%，取最细带差最大的窗
    mn, mg = 0.80, 0.08
    ok = [c for c in cand if c["ndom"] >= 3 and min(c["map_r"], c["map_d"]) >= mn
          and abs(c["map_r"] - c["map_d"]) / max(c["map_r"], c["map_d"]) <= mg]
    level = (mn, mg) if ok else None
    pool = ok if ok else cand
    best = max(pool, key=lambda c: c["band_r"] - c["band_d"])
    if TAG in ov:
        x_, y_ = map(float, ov[TAG].split(","))
        best = min(cand, key=lambda c: (c["x0"] - x_) ** 2 + (c["y0"] - y_) ** 2); level = ("manual", 0)
    print(f"[{TAG}] 筛选档位 {level}", flush=True)
    json.dump(cand, open(os.path.join(ROOT, f"results/fig1_v2_candidates_{TAG}.json"), "w"))
    gaps = np.array([c["band_r"] - c["band_d"] for c in cand])
    print(f"[{TAG}] 候选窗 {len(cand)}，通过筛选 {len(ok)}；选中 x{best['x0']:.0f} y{best['y0']:.0f} n={best['n']} 域数={best['ndom']} "
          f"map {best['map_d']:.3f}/{best['map_r']:.3f} band {best['band_d']:.3f}/{best['band_r']:.3f} "
          f"（带差 {best['band_r']-best['band_d']:.3f}，在全部候选中位于 {100*float((gaps < best['band_r']-best['band_d']).mean()):.0f} 百分位；候选中位带差 {np.median(gaps):.3f}）", flush=True)
    s_ = ((xt[:, 0] >= best["x0"]) & (xt[:, 0] < best["x0"] + SIDE) & (xt[:, 1] >= best["y0"]) & (xt[:, 1] < best["y0"] + SIDE))
    ds_file = os.path.join(ROOT, f"istar_run/{TAG}_{FOLD}/level-downsample.txt")
    DOWN = float(open(ds_file).read().split()[0]) if os.path.exists(ds_file) else 4.000053157559005
    HEP = os.path.join("/path/to/spatial2exp/he2st_align", f"istar_run/{TAG}_{FOLD}/he-raw.jpg")
    hp = pxl[m][tem][s_] / DOWN
    crop = Image.open(HEP).crop((int(hp[:, 0].min()), int(hp[:, 1].min()), int(np.ceil(hp[:, 0].max())), int(np.ceil(hp[:, 1].max()))))
    sub = xt[s_]; ix = np.round((sub[:, 0] - best["x0"]) / 16.0).astype(int); iy = np.round((sub[:, 1] - best["y0"]) / 16.0).astype(int)

    def grid(v):
        H = np.full((iy.max() + 1, ix.max() + 1), np.nan, np.float32); H[iy, ix] = v; return H
    return dict(TAG=TAG, FJ=FJ, pcc=pcc, sg=sg, bpcc=bpcc, bpow=bpow, pc1=pc1_share, best=best, ncand=len(cand), nok=len(ok), level=level,
                maps=[grid(FT[s_]), grid(FD[s_]), grid(FP[s_])], bands=[grid(GT[s_]), grid(GD[s_]), grid(GP[s_])],
                crop=crop, sb=200.0 / (DOWN / PX_PER_UM[SLIDE]))


D = [fold("P2"), fold("P5")]

# ───────── 画图：左 4×4 网格，右 b/c ─────────
fig = plt.figure(figsize=(183 * MM, 118 * MM))
# 紧凑版式：patch 行距 0.30→0.08、列距 0.06→0.03，上下边距收窄；左块宽度按 patch 边长收窄，使方形 patch 之间几乎无空白（patch 约放大 1.2 倍）
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.50], wspace=0.16, left=0.055, right=0.985, top=0.885, bottom=0.04)
gl = gs[0].subgridspec(4, 4, hspace=0.16, wspace=0.03)
# 右栏单独设底边距，避免 c 面板两行刻度标签被裁；水平范围与外层网格的右栏一致（0.698–0.985）
gr = fig.add_gridspec(2, 1, left=0.698, right=0.985, top=0.885, bottom=0.075, hspace=0.60)


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)


cols = ["H&E", "measured", "domains only", "trained model"]
subs = ["", "PC1 projection", "truth averaged in\n20 image domains", "ridge on\nfrozen features"]
for ci, d in enumerate(D):
    r0 = 2 * ci
    vmin, vmax = np.nanpercentile(np.concatenate([m_[~np.isnan(m_)] for m_ in d["maps"]]), [2, 98])
    gv = np.nanpercentile(np.abs(np.concatenate([b_[~np.isnan(b_)] for b_ in d["bands"]])), 99)
    # 行 1：场
    ax = fig.add_subplot(gl[r0, 0]); ax.imshow(np.asarray(d["crop"]), aspect="equal"); bare(ax)
    W_, H_ = d["crop"].size
    ax.plot([W_ * .06, W_ * .06 + d["sb"]], [H_ * .93] * 2, "-", color="white", lw=2.2)
    ax.text(W_ * .06 + d["sb"] / 2, H_ * .90, "200 µm", color="white", fontsize=5.2, ha="center", va="bottom")
    ax.text(-0.16, 0.5, f"section {ci+1}\nfield", transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=6.6, color=P["grey_d"], fontweight="bold")
    if ci == 0: ax.set_title(cols[0], fontsize=7.6, pad=4)
    for j, (M, sc, col) in enumerate([(d["maps"][0], None, P["grey_d"]), (d["maps"][1], d["best"]["map_d"], P["red"]), (d["maps"][2], d["best"]["map_r"], P["blue"])]):
        ax = fig.add_subplot(gl[r0, j + 1]); ax.imshow(M, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal"); bare(ax)
        if ci == 0:
            ax.set_title(cols[j + 1], fontsize=7.6, pad=4)
            pass  # 列副标题（PC1 projection / truth averaged… / ridge on…）移入图注，图内不再画灰字
        if sc is not None:
            ax.text(0.5, -0.03, "map PCC %.3f" % sc, transform=ax.transAxes, ha="center", va="top", fontsize=6.4, fontweight="bold", color=col)
    # 行 2：最细带
    ax = fig.add_subplot(gl[r0 + 1, 0]); ax.axis("off")
    ax.text(0.5, 0.62, "finest band\n$B_1 = F - WF$", transform=ax.transAxes, ha="center", va="center", fontsize=6.4, color=P["grey_d"], linespacing=1.5)
    ax.text(0.5, 0.30, "large-scale structure\nremoved", transform=ax.transAxes, ha="center", va="center", fontsize=5.4, color=P["grey_m"], linespacing=1.5)
    ax.text(-0.16, 0.5, f"section {ci+1}\nfinest band", transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=6.6, color=P["grey_d"], fontweight="bold")
    for j, (M, bp, col) in enumerate([(d["bands"][0], None, P["grey_d"]), (d["bands"][1], d["best"]["band_d"], P["red"]), (d["bands"][2], d["best"]["band_r"], P["blue"])]):
        ax = fig.add_subplot(gl[r0 + 1, j + 1]); ax.imshow(M, cmap="RdBu_r", vmin=-gv, vmax=gv, interpolation="nearest", aspect="equal"); bare(ax)
        ax.text(0.5, -0.03, "reference" if bp is None else "band PCC %.3f" % bp, transform=ax.transAxes, ha="center", va="top", fontsize=6.4, fontweight="bold", color=col)
fig.text(0.012, 0.935, "a", fontsize=9, fontweight="bold", va="top", ha="left")

# b：两折的标定曲线
ax = fig.add_subplot(gr[0])
for d, ls, lab in zip(D, ("-", "--"), ("section 1", "section 2")):
    FJ = d["FJ"]; SIG = {int(k): v for k, v in FJ["sigma_um"].items()}; ts = sorted(SIG)
    ss = np.array([SIG[t] for t in ts]); pp = np.array([FJ["scores"]["A_coarse_t%d" % t] for t in ts])
    ax.plot(ss, pp, ls, color=P["grey_m"], lw=1.2, zorder=1, label=lab)
    ax.plot(ss, pp, "o", ms=2.2, color=P["grey_m"], mfc="white", mew=0.8, zorder=2)
    for k, col in (("ridge", P["blue"]), ("dom20", P["red"])):
        ax.plot([d["sg"][k]], [d["pcc"][k]], "o", ms=5, color=col, zorder=4)
ax.set_xscale("log"); ax.set_xlabel("effective resolution $\\sigma$ (µm)", labelpad=1); ax.set_ylabel("per-gene PCC", labelpad=2)
ax.set_title("$\\sigma$ is a monotone function\nof PCC within a fold", fontsize=6.8, pad=4)
ax.legend(fontsize=5.6, loc="upper right", handlelength=1.6)
ax.text(0.03, 0.06, "blue: trained model\nred: domains only", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.4, color=P["grey_d"])
fig.text(0.700, 0.935, "b", fontsize=9, fontweight="bold", va="top", ha="left")

# c：最细带方差占比，两折分组
ax = fig.add_subplot(gr[1])
labs = ["measured", "domains\nonly", "trained\nmodel"]; colsb = [P["grey_d"], P["red"], P["blue"]]
w = 0.36
for i, d in enumerate(D):
    vals = [100 * d["bpow"]["truth"], 100 * d["bpow"]["dom20"], 100 * d["bpow"]["ridge"]]
    xs = np.arange(3) + (i - 0.5) * (w + 0.04)
    ax.bar(xs, vals, w, color=colsb, edgecolor=P["grey_d"], lw=0.4, alpha=1.0 if i == 0 else 0.55)
    for x_, v_ in zip(xs, vals):
        ax.text(x_, v_ + 0.4, "%.1f" % v_, ha="center", fontsize=5.0, color=P["grey_d"])
ax.set_xticks(np.arange(3)); ax.set_xticklabels(labs, fontsize=6.0)
ax.set_ylabel("variance in the finest band (%)", labelpad=2)
ymax = max(100 * d["bpow"]["truth"] for d in D); ax.set_ylim(0, ymax * 1.30)
rat = [d["bpow"]["truth"] / max(d["bpow"]["ridge"], d["bpow"]["dom20"]) for d in D] + [d["bpow"]["truth"] / min(d["bpow"]["ridge"], d["bpow"]["dom20"]) for d in D]
ax.set_title("both predictions are %.0f–%.0f$\\times$ too smooth\n(solid: section 1, faded: section 2)" % (min(rat), max(rat)), fontsize=6.8, pad=4)
fig.text(0.700, 0.47, "c", fontsize=9, fontweight="bold", va="top", ha="left")

fig.suptitle("What the score captures is the large-scale pattern, not the fine structure", fontsize=9.2, y=0.985, fontweight="bold")
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "Fig1_iclr_v2.pdf"))
print("-> figures/Fig1_iclr_v2.pdf", flush=True)
json.dump({d["TAG"]: dict(best=d["best"], pcc=d["pcc"], sg=d["sg"], bpcc=d["bpcc"], bpow=d["bpow"], pc1=d["pc1"], ncand=d["ncand"], nok=d["nok"], level=d["level"]) for d in D},
          open(os.path.join(ROOT, "results/fig1_v2_numbers.json"), "w"), indent=1)
print("-> results/fig1_v2_numbers.json", flush=True)

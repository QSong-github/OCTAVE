#!/usr/bin/env python
"""S14 —— 从旧 Fig 1 挪出的五格：标尺的实测标定与三条数据线。

Fig 1 改为动机图（分数买到的是大尺度图样）后，这五格失去主图位置，但内容全部保留：
  a  σ 的二阶矩标定（σ ∝ t^0.49）
  b  带通相关曲线与 τ 交点 —— 估计量 B 的等效分辨率跨度
  c  信号的尺度分布 vs 该尺度的保真度
  d  广度线 HEST 的队列与平台构成
  e  深度线 Xenium 的区域规模与样本归属
最终组稿时 a–c 应并入 S1（估计量定义），d–e 并入 S13（队列图）。
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figs_new import P, MM, RES, specimen, lab

OUT = "figures_supp"
TAU = "0.2"

d = json.load(open(f"{RES}/effres_hibou_l_hvg50_t2048.json"))
sig = {int(k): v for k, v in d["sigma_um"].items()}
cps = sorted(sig); sv = np.array([sig[c] for c in cps])
tp = {int(k): v for k, v in d["truth_power"].items()}
meths = sorted(d["methods"], key=lambda m: -m["pcc"])

fig = plt.figure(figsize=(183 * MM, 118 * MM))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.62, wspace=0.78,
                      top=0.855, bottom=0.085, left=0.075, right=0.945)

# ── a：σ 的实测标定
ax = fig.add_subplot(gs[0, 0])
ax.plot(cps, sv, "o-", color=P["blue"], lw=1.2, ms=3)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("Diffusion steps $t$"); ax.set_ylabel("Measured $\\sigma$ (µm)")
sl_ = np.polyfit(np.log(cps), np.log(sv), 1)[0]
ax.set_title("Calibrated by\nsecond moment", fontsize=6.6)
ax.text(0.05, 0.95, f"$\\sigma \\propto t^{{{sl_:.2f}}}$", transform=ax.transAxes,
        ha="left", va="top", fontsize=6, color=P["grey_d"])
lab(ax, "a", x=-0.52, y=1.24)

# ── b：带通相关曲线 + τ 交点（估计量 B）
ax = fig.add_subplot(gs[0, 1:3])
_er = {m["method"]: m["eff_res_um"][TAU] for m in meths}
hi = {min(_er, key=_er.get): P["blue"], "Ridge_HEST": P["teal"],
      max(_er, key=_er.get): P["red"]}
for m in meths:
    cur = np.array([{int(k): v for k, v in m["curve"].items()}[c] for c in cps])
    c = hi.get(m["method"])
    ax.plot(sv, cur, "-", lw=1.3 if c else 0.7, color=c or P["grey_l"],
            zorder=3 if c else 1, label=m["method"] if c else None)
er = np.array([m["eff_res_um"][TAU] for m in meths])
ax.axhline(float(TAU), color=P["grey_m"], lw=0.8, ls="--", zorder=2)
ax.scatter(er, [float(TAU)] * len(er), s=14, color=P["black"], zorder=5,
           clip_on=False, marker="v")
ax.set_xscale("log")
ax.set_xlabel("Band centre $\\sigma$ (µm)"); ax.set_ylabel("Band-wise PCC")
ax.set_title(f"Equivalent resolution (estimator B) at $\\tau$ = {TAU} spans "
             f"{er.min():.1f}–{er.max():.1f} µm ({er.max()/er.min():.2f}×, n = {len(meths)})",
             fontsize=6.6)
ax.legend(fontsize=5.4, loc="center right", bbox_to_anchor=(1.0, 0.34),
          handlelength=1.2)
ins = ax.inset_axes([0.07, 0.56, 0.40, 0.40])
for m in meths:
    cur = np.array([{int(k): v for k, v in m["curve"].items()}[c] for c in cps])
    c = hi.get(m["method"])
    ins.plot(sv, cur, "-", lw=1.1 if c else 0.6, color=c or P["grey_l"],
             zorder=3 if c else 1)
ins.axhline(float(TAU), color=P["grey_m"], lw=0.7, ls="--")
ins.scatter(er, [float(TAU)] * len(er), s=8, color=P["black"], zorder=5, marker="v")
ins.set_xscale("log")
ins.set_xlim(sv[0] * 0.93, er.max() * 1.5); ins.set_ylim(0.10, 0.32)
ins.tick_params(labelsize=5.0, length=1.8, pad=1)
ins.xaxis.set_major_locator(FixedLocator([10, 15, 20, 28]))
ins.xaxis.set_major_formatter(FixedFormatter(["10", "15", "20", "28"]))
ins.xaxis.set_minor_locator(FixedLocator([]))
ins.xaxis.set_minor_formatter(NullFormatter())
for s_ in ("top", "right"):
    ins.spines[s_].set_visible(True)
for s_ in ins.spines.values():
    s_.set_linewidth(0.5); s_.set_color(P["grey_m"])
lab(ax, "b", x=-0.075, y=1.24)

# ── c：功率的尺度分布 vs 保真度
ax = fig.add_subplot(gs[0, 3])
tot = sum(tp.values())
sh = np.array([100 * tp[c] / tot for c in cps])
x = np.arange(len(cps))
ax.bar(x, sh, color=P["grey_l"], edgecolor=P["grey_d"], lw=0.5, width=0.75, zorder=2)
ax.set_xticks(x[::2])
ax.set_xticklabels([f"{s:.0f}" for s in sv[::2]], rotation=45, ha="right", fontsize=5.2)
ax.set_xlabel("Band $\\sigma$ (µm)", fontsize=6)
ax.set_ylabel("Share of band-passed\ntruth power (%)", fontsize=6)
best = np.array([{int(k): v for k, v in meths[0]["curve"].items()}[c] for c in cps])
ax2 = ax.twinx()
ax2.plot(x, best, "o-", color=P["blue"], lw=1.3, ms=2.5, zorder=3)
ax2.set_ylabel("Band-wise PCC", color=P["blue"], fontsize=6)
ax2.tick_params(axis="y", colors=P["blue"], labelsize=5.5)
ax2.spines["top"].set_visible(False); ax2.set_ylim(0, 1)
ax.set_title("Signal sits where\nfidelity is lowest", fontsize=6.6)
ax.text(0.97, 0.30, f"{sh[0]:.0f}% of signal\nat PCC {best[0]:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=5.4, color=P["grey_d"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6))
lab(ax, "c", x=-0.56, y=1.24)

# ── d：广度线构成
ax = fig.add_subplot(gs[1, :2])
L = json.load(open(f"{RES}/hest_ladder.json"))
coh, plat = {}, {}
for v in L.values():
    coh.setdefault(v["cohort"], {"Visium": 0, "Xenium": 0})[v["platform"]] += 1
    plat[v["platform"]] = plat.get(v["platform"], 0) + 1
order = sorted(coh, key=lambda c: -sum(coh[c].values()))
y = np.arange(len(order))[::-1]
vv = np.array([coh[c]["Visium"] for c in order], float)
xx = np.array([coh[c]["Xenium"] for c in order], float)
ax.barh(y, vv, color=P["blue"], edgecolor=P["grey_d"], lw=0.4, height=0.68,
        label=f"Visium (n={plat.get('Visium', 0)})")
ax.barh(y, xx, left=vv, color=P["teal"], edgecolor=P["grey_d"], lw=0.4, height=0.68,
        hatch="///", label=f"Xenium (n={plat.get('Xenium', 0)})")
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=5.6)
ax.set_xlabel("samples")
ax.set_title(f"Breadth line: HEST benchmark — {len(L)} samples, "
             f"{len(order)} cohorts, 2 platforms", fontsize=6.6)
ax.legend(fontsize=5.4, loc="lower right", handlelength=1.0)
lab(ax, "d", x=-0.155, y=1.16)

# ── e：深度线规模
ax = fig.add_subplot(gs[1, 2:])
X = [json.load(open(f)) for f in sorted(glob.glob(f"{RES}/xenium/*.json"))]
nb = np.array([d_["n"] for d_ in X], float) / 1e3
o = np.argsort(nb)
cols = [P["violet"] if specimen(X[i]["name"]).startswith("Human_Breast") else P["red"]
        for i in o]
ax.barh(np.arange(len(o)), nb[o], color=cols, edgecolor=P["grey_d"], lw=0.4, height=0.75)
ax.set_yticks([]); ax.set_xlabel("bins per region (×10³)")
nsp = len({specimen(d_["name"]) for d_ in X})
ax.set_ylabel(f"{len(X)} regions")
ax.set_title(f"Depth line: Xenium — {len(X)} regions from {nsp} specimens", fontsize=6.6)
ax.text(0.97, 0.06, "purple: 4 breast blocks (3 regions each)\nred: 4 single-region "
        "specimens", transform=ax.transAxes, ha="right", va="bottom",
        fontsize=5.2, color=P["grey_d"])
lab(ax, "e", x=-0.075, y=1.16)

fig.suptitle("S14 | The equivalent-resolution ladder and the three data lines",
             fontsize=9, y=0.995, fontweight="bold")
os.makedirs(OUT, exist_ok=True)
fig.savefig(f"{OUT}/S14_ruler_and_data_lines.pdf")
print(f"→ {OUT}/S14_ruler_and_data_lines.pdf", flush=True)
print(f"  ER(τ={TAU}) 跨度 {er.min():.1f}–{er.max():.1f} µm ({er.max()/er.min():.2f}×)")
print(f"  σ ∝ t^{sl_:.2f}；最细带占信号 {sh[0]:.1f}%，该带最佳 PCC {best[0]:.3f}")
print(f"  HEST {len(L)} 样本 / {len(order)} 队列；Xenium {len(X)} 区域 / {nsp} 样本")

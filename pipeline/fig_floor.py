# -*- coding: utf-8 -*-
"""Fig — 基准上的「优势」值多少：与同源检索下界的配对比较，及其超参依赖。

四格一条链：
  a 每个编码器 vs **它自己特征**的 kNN 检索下界(k=50)：7/15 低于下界
  b 把下界的一个超参 k 从 10 调到 800：所有编码器整体平移，中位 15.2 pp
  c 但名次几乎不动(ρ 中位 0.945，最多动 3 位) —— 移的是水平不是次序
  d 于是「有几个打得赢检索」完全由 k 决定(2/15 → 10/14)，不是编码器的性质

纪律：k=200/800 有队列退化(邻域占留一训练集 >5%)，一律用 rel_ok(仅非退化队列)，
      退化档在图上以灰带标出。相对量不做逐样本除法（近零分母）。
"""
import json, os
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("S4ST", "/path/to/systema4ST")
OUT = os.path.join(BASE, "figures")
K = json.load(open(os.path.join(BASE, "results/k_sensitivity.json")))
KS = sorted(int(k) for k in K)
DEGEN = {10: 0, 50: 0, 200: 3, 800: 7}          # 退化队列数，见 ksize.py

P = {"blue": "#0F4D92", "red": "#B64342", "teal": "#42949E", "violet": "#9A4D8E",
     "grey_l": "#CFCECE", "grey_m": "#767676", "grey_d": "#4D4D4D", "black": "#272727"}
MM = 1 / 25.4
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.8, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5})


def v(k, e):
    r = K[str(k)][e]
    return r["rel_ok"] if r["rel_ok"] is not None else r["rel"]


ALL = sorted(K["50"], key=lambda e: -v(50, e))
COMMON = sorted(set.intersection(*[set(K[str(k)]) for k in KS]), key=lambda e: -v(10, e))
SHORT = {"dinov3_vitl16": "dinov3-L", "dinov2_large": "dinov2-L", "kaiko_vitb16": "kaiko-B16",
         "kaiko_vitl14": "kaiko-L14", "kaiko_vits16": "kaiko-S16", "lunit_vits8": "lunit-S8",
         "conch_v15": "CONCH v1.5", "hoptimus0": "H-optimus-0", "midnight12k": "Midnight-12k",
         "phikon_v2": "Phikon-v2", "phikon": "Phikon", "uni_v2": "UNI v2",
         "virchow2": "Virchow2", "gigapath": "GigaPath", "ciga": "CIGA"}
sh = lambda e: SHORT.get(e, e)
xs = np.arange(len(KS))

fig = plt.figure(figsize=(183 * MM, 152 * MM))
gs = fig.add_gridspec(2, 2, hspace=0.60, wspace=0.40,
                      left=0.135, right=0.975, top=0.855, bottom=0.075)

# ── a
ax = fig.add_subplot(gs[0, 0])
val = [v(50, e) for e in ALL]
y = np.arange(len(ALL))[::-1]
ax.barh(y, val, color=[P["red"] if x < 0 else P["blue"] for x in val],
        height=0.72, edgecolor=P["grey_d"], lw=0.4)
ax.axvline(0, color=P["black"], lw=1.0)
ax.set_yticks(y); ax.set_yticklabels([sh(e) for e in ALL], fontsize=5.8)
ax.set_xlim(min(val) * 1.12, max(val) * 2.9)
ax.set_xlabel("PCC relative to that encoder's own\nretrieval floor (%)", labelpad=1)
nb = sum(1 for x in val if x < 0)
ax.set_title(f"Each encoder vs $k$-NN retrieval on its own features\n"
             f"{nb}/{len(ALL)} score below their own floor   ($k$ = 50)", fontsize=7.2, pad=6)
ax.text(0.985, 0.045, "same features, same PCA-256\npipeline, same folds, same\n"
        "50 genes — only ridge → retrieval", transform=ax.transAxes, fontsize=5.6,
        color=P["grey_d"], va="bottom", ha="right", linespacing=1.6)

# ── b
ax = fig.add_subplot(gs[0, 1])
for e in COMMON:
    ax.plot(xs, [v(k, e) for k in KS], "-o", ms=2.6, lw=1.0, color=P["grey_m"],
            alpha=0.85, zorder=2)
med = [float(np.median([v(k, e) for e in COMMON])) for k in KS]
ax.plot(xs, med, "-o", ms=5, lw=2.4, color=P["red"], zorder=4)
ax.axhline(0, color=P["black"], lw=0.9, ls=":")
for i, k in enumerate(KS):
    if DEGEN[k]:
        ax.axvspan(i - 0.5, i + 0.5, color=P["grey_l"], alpha=0.32, zorder=0, lw=0)
ax.set_xticks(xs); ax.set_xticklabels([f"$k$ = {k}" for k in KS], fontsize=6.4)
ax.set_xlim(-0.5, len(KS) - 0.5); ax.set_ylim(-30, 26)
ax.set_ylabel("relative to own floor (%)", labelpad=2)
d10_200 = [v(200, e) - v(10, e) for e in COMMON]
SHIFT = abs(float(np.median(d10_200)))
ax.set_title("One hyper-parameter of that baseline shifts\n"
             f"every encoder by {SHIFT:.1f} pp (median)", fontsize=7.2, pad=6)
ax.annotate("", xy=(2, med[2]), xytext=(0, med[0]),
            arrowprops=dict(arrowstyle="-|>", color=P["red"], lw=1.3, shrinkA=8, shrinkB=8))
ax.text(0.02, 0.03, "grey: $k$ degenerate in 3/10 and 7/10 cohorts\n"
        "(neighbourhood > 5% of the training set)", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=5.6, color=P["grey_d"], linespacing=1.5)
ax.text(0.985, 0.955, "red: median", transform=ax.transAxes, ha="right", va="top",
        fontsize=5.9, color=P["red"], fontweight="bold")

# ── c
ax = fig.add_subplot(gs[1, 0])
R = {k: {e: r + 1 for r, e in enumerate(sorted(COMMON, key=lambda x: -v(k, x)))} for k in KS}
for e in COMMON:
    rr = [R[k][e] for k in KS]
    moved = max(rr) - min(rr)
    ax.plot(xs, rr, "-o", ms=3.4, lw=1.7 if moved >= 2 else 1.0,
            color=P["violet"] if moved >= 2 else P["grey_m"],
            alpha=1.0 if moved >= 2 else 0.5, zorder=3 if moved >= 2 else 2)
    ax.text(-0.14, R[KS[0]][e], sh(e), ha="right", va="center", fontsize=5.6, color=P["grey_d"])
rho = [spearmanr([v(a, e) for e in COMMON], [v(b, e) for e in COMMON]).statistic
       for i, a in enumerate(KS) for b in KS[i + 1:]]
mx = max(max(R[k][e] for k in KS) - min(R[k][e] for k in KS) for e in COMMON)
ax.set_xticks(xs); ax.set_xticklabels([f"$k$ = {k}" for k in KS], fontsize=6.4)
ax.set_xlim(-1.75, len(KS) - 0.65); ax.set_ylim(11.4, 0.15)
ax.set_ylabel("rank", labelpad=2)
ax.set_yticks(range(1, len(COMMON) + 1)); ax.tick_params(labelsize=6)
ax.set_title(f"…but it does not reorder them:  $\\rho$ = {np.median(rho):.3f}\n"
             f"(median of {len(rho)} $k$-pairs); largest move {mx} of {len(COMMON)} places",
             fontsize=7.2, pad=6)
ax.text(0.99, 0.03, "purple: moved ≥ 2 places", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=5.9, color=P["violet"], fontweight="bold")

# ── d
ax = fig.add_subplot(gs[1, 1])
nb_k = [sum(1 for e in K[str(k)] if v(k, e) < 0) for k in KS]
nt_k = [len(K[str(k)]) for k in KS]
ns_k = [sum(1 for e in K[str(k)] if (K[str(k)][e]["P_ok"] or K[str(k)][e]["P"]) < 0.05)
        for k in KS]
w = 0.36
ax.bar(xs - w / 2, [100 * a / b for a, b in zip(nb_k, nt_k)], w, color=P["red"],
       edgecolor=P["grey_d"], lw=0.4, label="below their own floor")
ax.bar(xs + w / 2, [100 * a / b for a, b in zip(ns_k, nt_k)], w, color=P["teal"],
       edgecolor=P["grey_d"], lw=0.4,
       label="beat it significantly (cohort sign test, $n$ = 10)")
for i, (a, b, c_) in enumerate(zip(nb_k, ns_k, nt_k)):
    ax.text(i - w / 2, 100 * a / c_ + 2.5, f"{a}/{c_}", ha="center", fontsize=6,
            color=P["red"], fontweight="bold")
    ax.text(i + w / 2, 100 * b / c_ + 2.5, f"{b}/{c_}", ha="center", fontsize=6,
            color=P["teal"], fontweight="bold")
for i, k in enumerate(KS):
    if DEGEN[k]:
        ax.axvspan(i - 0.5, i + 0.5, color=P["grey_l"], alpha=0.32, zorder=0, lw=0)
ax.set_xticks(xs); ax.set_xticklabels([f"$k$ = {k}" for k in KS], fontsize=6.4)
ax.set_xlim(-0.5, len(KS) - 0.5)
ax.set_ylabel("share of encoders (%)", labelpad=2); ax.set_ylim(0, 108)
ax.legend(fontsize=5.7, loc="upper left", handlelength=1.1, borderpad=0.2, labelspacing=0.35)
ax.set_title("So “how many beat retrieval” is a property of the\n"
             "baseline's knob, not of the encoders", fontsize=7.2, pad=6)

for _ch, _x, _y in (("a", 0.018, 0.888), ("b", 0.508, 0.888),
                    ("c", 0.018, 0.408), ("d", 0.508, 0.408)):
    fig.text(_x, _y, _ch, fontsize=9, fontweight="bold", va="top", ha="left")

spans = {k: max(v(k, e) for e in COMMON) - min(v(k, e) for e in COMMON) for k in KS}
ok_spans = [spans[k] for k in KS if DEGEN[k] <= 3]
lo, hi = 100 * SHIFT / max(ok_spans), 100 * SHIFT / min(ok_spans)
fig.suptitle("A margin over a trivial baseline is worth less than one "
             "hyper-parameter of that baseline", fontsize=9.2, y=0.985, fontweight="bold")
fig.text(0.5, 0.938, f"the {SHIFT:.1f} pp shift in b is {lo:.0f}–{hi:.0f}% of the entire "
         f"{min(ok_spans):.1f}–{max(ok_spans):.1f} pp spread across all {len(ALL)} encoders",
         ha="center", va="top", fontsize=7.4, color=P["grey_d"])
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "Fig_matched_floor.pdf"))
print("→ figures/Fig_matched_floor.pdf")
print(f"  k=50 低于下界 {nb}/{len(ALL)}；各档 below={nb_k} sig={ns_k} (n={nt_k})")
print(f"  位移中位 {SHIFT:.2f} pp；非退化档跨度 {min(ok_spans):.1f}–{max(ok_spans):.1f} pp"
      f" ⇒ {lo:.0f}–{hi:.0f}%")
print(f"  ρ 中位 {np.median(rho):.3f}；名次最大变动 {mx} 位")

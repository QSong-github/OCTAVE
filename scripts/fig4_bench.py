# -*- coding: utf-8 -*-
"""Figure 4, the 57 encoders scored by OCTAVE on the 3 datasets (rule B, from results/bench57.json,
written by scripts/mk_tab_bench57.py). a-c: band score of each encoder against band width on Xenium, Visium HD
and HEST, with the highest and lowest in the finest band labelled. d-f: finest-band scores on 2 datasets."""
import json, sys
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts"); from _enc_names import NAMES
PC = sys.argv[1] if len(sys.argv) > 1 else "paper"
J = json.load(open("results/bench57.json")); E = list(J["encoders"]); W = J["summary"]["widths"]
DS = ["Xenium", "Visium HD", "HEST"]
B = {d: {e: np.array(J["encoders"][e][d]) for e in E} for d in DS}
RED, BLUE, GREY, DARK = "#B64342", "#0F4D92", "#B8B8B8", "#4D4D4D"
MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42,
                     "font.size": 7, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
fig = plt.figure(figsize=(183 * MM, 99 * MM))
gs = fig.add_gridspec(2, 3, wspace=0.34, hspace=0.62, left=0.065, right=0.985, top=0.90, bottom=0.10)
for j, d in enumerate(DS):
    ax = fig.add_subplot(gs[0, j]); s = np.array(W[d]); n = len(s)
    for e in E: ax.plot(s, B[d][e][:n], "-", lw=0.6, color=GREY, zorder=1)
    hi = max(E, key=lambda e: round(B[d][e][0], 3)); lo = min(E, key=lambda e: round(B[d][e][0], 3))
    for e, col in ((hi, RED), (lo, BLUE)):
        ax.plot(s, B[d][e][:n], "-o", lw=1.5, ms=2.4, color=col, zorder=3, label=NAMES[e])
    ax.legend(fontsize=6.2, loc="lower right" if d != "HEST" else "upper left", handlelength=1.4, borderaxespad=0.3, frameon=False)
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    ax.set_xlabel("band width $\\sigma$ (µm)", labelpad=1)
    if j == 0: ax.set_ylabel("band score $\\beta_i$", labelpad=2)
    ax.set_title(f"{d}, {len(E)} encoders", fontsize=7.4, pad=5)
    ax.text(-0.17 if j == 0 else -0.12, 1.17, "abc"[j], transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
pairs = [("Xenium", "Visium HD"), ("Xenium", "HEST"), ("Visium HD", "HEST")]
for j, (a, b) in enumerate(pairs):
    ax = fig.add_subplot(gs[1, j])
    x = np.array([B[a][e][0] for e in E]); y = np.array([B[b][e][0] for e in E])
    ax.scatter(x, y, s=9, color=DARK, linewidths=0)
    if b != "HEST":
        lim = (0, 0.28); ax.plot(lim, lim, ls="--", lw=0.8, color="#767676"); ax.set_xlim(lim); ax.set_ylim(lim)
    else:
        ax.set_xlim(0, 0.28); ax.set_ylim(0.03, 0.11)
    ax.set_xlabel(f"finest-band score on {a}", labelpad=1); ax.set_ylabel(f"finest-band score on {b}", labelpad=2)
    rho = spearmanr(x, y).statistic
    ax.text(0.04, 0.96, f"Spearman {rho:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color=DARK)
    ax.set_title(f"{a} against {b}", fontsize=7.4, pad=5)
    ax.text(-0.17 if j == 0 else -0.12, 1.17, "def"[j], transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    print(f"{a} vs {b}: Spearman {rho:.3f}")
fig.suptitle("The 57 encoders scored by OCTAVE on 3 datasets", fontsize=9.2, y=0.985, fontweight="bold")
fig.savefig(f"{PC}/figs/Fig4_bench.pdf"); fig.savefig(f"{PC}/figs/Fig4_bench.png", dpi=200)

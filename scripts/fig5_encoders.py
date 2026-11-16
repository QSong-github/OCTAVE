# -*- coding: utf-8 -*-
"""Figure 5, encoder type, size and version against the finest-band score (rule B, results/bench57.json).
a: beta_1 on Xenium against parameter count, by encoder type. b: beta_1 by encoder type on the 3 datasets, with medians.
c: change in beta_1 from the earlier or smaller to the later or larger version of 8 encoders, on the 3 datasets.
Prints the numbers quoted in Section 4.4."""
import json, sys
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts"); from _enc_names import NAMES
PC = sys.argv[1] if len(sys.argv) > 1 else "paper"
J = json.load(open("results/bench57.json")); E = list(J["encoders"]); PAR = json.load(open("results/encoder_params.json"))
DS = ["Xenium", "Visium HD", "HEST"]
b1 = {d: {e: round(J["encoders"][e][d][0] + 1e-12, 3) for e in E} for d in DS}      # the printed values of Table 2
GEN = {"dinov2_base", "dinov2_large", "dinov2_giant", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16", "clip_vitl14", "siglip2"}
DIS = {"litevirchow2", "h0_mini", "litefm", "litefm_l", "litefm_s", "distillpath_is16", "distillpath_ks16", "pathryoshka_b"}
def kind(e):
    return "general" if e in GEN else ("vlm" if e == "biomedclip" else ("distilled" if e in DIS else "pathology"))
KC = {"pathology": "#0F4D92", "distilled": "#E9A23B", "general": "#8E8E8E", "vlm": "#9A7BB8"}
KL = {"pathology": "pathology", "distilled": "pathology, distilled", "general": "general-purpose", "vlm": "biomedical vision-language"}
PAIRS = [("virchow", "virchow2"), ("hoptimus0", "hoptimus1"), ("uni_v1", "uni_v2"), ("conch_v1", "conch_v15"),
         ("phikon", "phikon_v2"), ("hibou_b", "hibou_l"), ("dinov2_base", "dinov2_giant"), ("dinov3_vitb16", "dinov3_vith16")]
DC = {"Xenium": "#B64342", "Visium HD": "#E9A23B", "HEST": "#5F7FA8"}
MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42,
                     "font.size": 7, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
W_, H_ = 183.0, 65.0
fig = plt.figure(figsize=(W_ * MM, H_ * MM))
box = lambda x, w: fig.add_axes([x / W_, 11.2 / H_, w / W_, 41.5 / H_])
# a: parameters
ax = box(12.0, 44.0)
for k in ("pathology", "distilled", "general", "vlm"):
    es = [e for e in E if kind(e) == k]
    ax.scatter([PAR[e]["params"] / 1e6 for e in es], [b1["Xenium"][e] for e in es], s=11, color=KC[k], label=KL[k], linewidths=0, zorder=3)
ax.set_xscale("log"); ax.set_ylim(0, 0.26)
ax.set_xlabel("parameters (millions)", labelpad=1); ax.set_ylabel("finest-band score $\\beta_1$ on Xenium", labelpad=2)
rho = spearmanr([PAR[e]["params"] for e in E], [b1["Xenium"][e] for e in E]).statistic
ax.text(0.97, 0.04, f"Spearman {rho:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color="#4D4D4D")
ax.legend(fontsize=5.6, loc="upper left", handletextpad=0.1, borderaxespad=0.1, labelspacing=0.25, markerscale=1.0, frameon=False)
ax.set_title("size and type", fontsize=7.4, pad=5)
ax.text(-0.24, 1.2, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
# b: type on the 3 datasets
ax = box(70.0, 46.0); rng = np.random.default_rng(0)
K3 = ["pathology", "distilled", "general"]
for j, d in enumerate(DS):
    for q, k in enumerate(K3):
        v = np.array([b1[d][e] for e in E if kind(e) == k]); x0 = j * 3.6 + q * 1.05
        ax.scatter(x0 + rng.uniform(-0.28, 0.28, len(v)), v, s=6, color=KC[k], linewidths=0, alpha=0.9, zorder=3)
        ax.plot([x0 - 0.4, x0 + 0.4], [np.median(v)] * 2, color="#1B1B1B", lw=1.3, zorder=4)
ax.set_xticks([1.05 + j * 3.6 for j in range(3)]); ax.set_xticklabels(DS)
ax.set_ylim(0, 0.28); ax.set_ylabel("finest-band score $\\beta_1$", labelpad=2)
ax.set_title("type on the 3 datasets", fontsize=7.4, pad=5)
ax.text(-0.24, 1.2, "b", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
# c: versions
ax = box(148.0, 33.5); h = 0.24
for i, (a, b) in enumerate(PAIRS):
    for j, d in enumerate(DS):
        dv = b1[d][b] - b1[d][a]
        ax.barh(i + (j - 1) * h, dv, height=h * 0.92, color=DC[d], label=d if i == 0 else None, zorder=3)
ax.axvline(0, color="#4D4D4D", lw=0.8, zorder=4)
ax.set_yticks(range(len(PAIRS))); ax.set_yticklabels([f"{NAMES[a]} → {NAMES[b]}" for a, b in PAIRS], fontsize=6.0)
ax.invert_yaxis(); ax.set_xlabel("change in $\\beta_1$", labelpad=1)
ax.legend(fontsize=5.8, loc="upper right", handlelength=1.0, borderaxespad=0.2, frameon=False)
ax.set_title("version and size", fontsize=7.4, pad=5)
ax.text(-0.82, 1.2, "c", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
fig.suptitle("Encoder type, size and version", fontsize=9.2, y=0.985, fontweight="bold")
fig.savefig(f"{PC}/figs/Fig5_encoders.pdf"); fig.savefig(f"{PC}/figs/Fig5_encoders.png", dpi=200)
print("Spearman params vs beta1 (Xenium)", round(rho, 3))
for d in DS:
    print(d, {k: round(float(np.median([b1[d][e] for e in E if kind(e) == k])), 4) for k in K3},
          "pathology incl. distilled", round(float(np.median([b1[d][e] for e in E if kind(e) in ("pathology", "distilled")])), 4))
for a, b in PAIRS:
    print(f"{NAMES[a]} -> {NAMES[b]}: " + ", ".join(f"{d} {b1[d][a]:.3f}->{b1[d][b]:.3f}" for d in DS))

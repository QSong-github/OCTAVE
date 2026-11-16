# -*- coding: utf-8 -*-
"""Figure 2 under rule B: scores per region, medians within and then across the 13 specimens; gaps from those medians.
a  relative gap between the domain oracle and the trained model in each band (Hibou-L), dashed line under PCC
b  share of each map's variance in each band
c  rank of the 57 encoders under PCC against the rank in the finest band"""
import json, glob, os, re, itertools
import numpy as np
from math import comb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ["OCTAVE_ROOT"]; OUT = os.environ["OUT"]; R = os.path.join(BASE, "results")
P = {"blue": "#0F4D92", "red": "#B64342", "grey_l": "#CFCECE", "grey_m": "#767676", "grey_d": "#4D4D4D"}
MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "font.size": 7, "axes.linewidth": 0.8, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.8,
                     "ytick.major.width": 0.8, "xtick.major.size": 2.5, "ytick.major.size": 2.5})


def specimen(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    if m: return m.group(1)
    if "Human_Lung_Cancer_FFPE" in n: return "Lung"
    return n


def signp(k, n):
    k = max(k, n - k); return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def load(d):
    return {x["name"]: x for x in (json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, d, "*.json"))))}


def spec_medians(D, f):
    sp = {}
    for n, x in D.items(): sp.setdefault(specimen(n), []).append(f(x))
    return {s: np.median(np.vstack(v), axis=0) for s, v in sp.items()}


D = load("blocks_xen_bands"); TS = sorted(int(t) for t in next(iter(D.values()))["pred"]["ridge"]["bands"])
arr = lambda x, p, key: np.array([x["pred"][p][key][str(t)] for t in TS])
SPm = spec_medians(D, lambda x: np.r_[x["pred"]["ridge"]["pcc"], arr(x, "ridge", "bands")])
SPo = spec_medians(D, lambda x: np.r_[x["pred"]["dom20"]["pcc"], arr(x, "dom20", "bands")])
SIG = spec_medians(D, lambda x: np.array([x["sigma_um"][str(t)] for t in TS]))
names = sorted(SPm); assert len(names) == 13
med = lambda S: np.median(np.vstack([S[s] for s in names]), axis=0)
M, O, sig = med(SPm), med(SPo), med(SIG)
gap = 100 * (1 - O / M); gP, gB = gap[0], gap[1:]
k = sum(1 for s in names if (1 - SPo[s][1] / SPm[s][1]) > (1 - SPo[s][0] / SPm[s][0])); Pv = signp(k, 13)
V = {key: med(spec_medians(D, lambda x, p=p: arr(x, p, "band_var"))) for key, p in (("vt", "truth"), ("vb", "dom20"), ("vr", "ridge"))}

# encoders: per-encoder scores under the same rule
enc = json.load(open(os.path.join(R, "xen_multi_rank.json")))["encoders"]
sc = {}
for e in enc:
    De = load("blocks_xen_bands_base" if e == "hibou_l" else f"blocks_xen_bands_{e}"); assert len(De) == 22, e
    sc[e] = med(spec_medians(De, lambda x: np.array([x["pred"]["ridge"]["pcc"], x["pred"]["ridge"]["bands"]["1"]])))
E = sorted(sc); pcc = np.array([sc[e][0] for e in E]); b1 = np.array([sc[e][1] for e in E])
rk = lambda v: (-v).argsort().argsort() + 1
rp, rb = rk(pcc), rk(b1)
rev = sum((pcc[i] - pcc[j]) * (b1[i] - b1[j]) < 0 for i, j in itertools.combinations(range(len(E)), 2))
rs = np.corrcoef(rp, rb)[0, 1]

fig = plt.figure(figsize=(183 * MM, 66 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.1, 0.95], wspace=0.55, left=0.065, right=0.985, top=0.80, bottom=0.17)
def letter(ax, ch, x):
    ax.text(x, 1.22, ch, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")

ax = fig.add_subplot(gs[0])
ax.plot(sig, gB, "-o", lw=2.0, ms=3.2, color=P["red"], zorder=4)
ax.axhline(gP, color=P["blue"], lw=1.3, ls="--", zorder=3)
ax.annotate(f"gap under PCC: {gP:.0f}%", (sig[-1], gP), textcoords="offset points", xytext=(-2, 4),
            ha="right", fontsize=6.3, color=P["blue"], fontweight="bold")
ax.annotate(f"finest band: {gB[0]:.0f}%", (sig[0], gB[0]), textcoords="offset points", xytext=(8, -3),
            fontsize=6.3, color=P["red"], fontweight="bold")
ax.set_xscale("log"); ax.set_ylim(0, 100)
ax.set_xlabel("band width $\\sigma$ (µm)", labelpad=1)
ax.set_ylabel("relative gap between the domain\noracle and the trained model (%)", labelpad=2)
ax.set_title("relative gap\nin each band", fontsize=7.4, pad=6)
ax.text(0.98, 0.97, f"finest-band gap > gap under PCC\nin {k}/13 specimens (exact $P$ = {Pv:.4f})",
        transform=ax.transAxes, ha="right", va="top", fontsize=5.9, color=P["grey_d"], linespacing=1.5)
letter(ax, "a", -0.17)

ax = fig.add_subplot(gs[1])
for key, col, lab in (("vt", P["grey_d"], "measured"), ("vb", P["red"], "domain oracle"), ("vr", P["blue"], "trained model")):
    ax.plot(sig, 100 * V[key], "-o", lw=1.4, ms=2.8, color=col, label=lab)
ax.set_xscale("log")
ax.set_xlabel("band width $\\sigma$ (µm)", labelpad=1)
ax.set_ylabel("share of the map's variance\nin the band (%)", labelpad=2)
ax.set_title("share of variance\nin each band", fontsize=7.4, pad=6)
ax.legend(fontsize=6.0, loc="upper right", handlelength=1.3)
letter(ax, "b", -0.22)

ax = fig.add_subplot(gs[2])
ax.plot([1, 57], [1, 57], ls="--", lw=0.8, color=P["grey_m"], zorder=1)
ax.scatter(rp, rb, s=9, color=P["grey_d"], zorder=2, linewidths=0)
for e, lab, col, off in (("conch_v1", "CONCH v1", P["red"], (5, -2)), ("keep", "KEEP", P["blue"], (5, -2))):
    i = E.index(e); ax.scatter([rp[i]], [rb[i]], s=18, color=col, zorder=3)
    ax.annotate(lab, (rp[i], rb[i]), textcoords="offset points", xytext=off, fontsize=6.0, color=col, fontweight="bold")
ax.set_xlim(0, 58); ax.set_ylim(58, 0)
ax.set_xlabel("rank under PCC", labelpad=1); ax.set_ylabel("rank in the finest band", labelpad=2)
ax.set_title(f"rank of the {len(E)} encoders", fontsize=7.4, pad=6)
ax.text(0.97, 0.97, f"{rev} of {len(E) * (len(E) - 1) // 2} pairs\nin reversed order", transform=ax.transAxes,
        ha="right", va="top", fontsize=5.9, color=P["grey_d"], linespacing=1.4)
letter(ax, "c", -0.25)

fig.suptitle("Xenium, scored in each band", fontsize=9.2, y=0.985, fontweight="bold")
fig.savefig(os.path.join(OUT, "Fig2_scale.pdf")); fig.savefig(os.path.join(OUT, "Fig2_scale.png"), dpi=200)
print(f"gap PCC {gP:.2f}  bands {np.round(gB[:5], 1)}  widths {np.round(sig[1:6], 1)}  k {k}/13 P {Pv:.5f}")
print(f"variance finest: measured {100*V['vt'][0]:.1f}  oracle {100*V['vb'][0]:.1f}  model {100*V['vr'][0]:.1f}")
print(f"encoders {len(E)}  reversed {rev}  rank r {rs:.3f}")
print("CONCH v1", np.round(sc['conch_v1'], 4), "KEEP", np.round(sc['keep'], 4), "DINOv3-H+", np.round(sc.get('dinov3_vith16', [np.nan, np.nan]), 4))

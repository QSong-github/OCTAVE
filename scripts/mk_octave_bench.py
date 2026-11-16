# -*- coding: utf-8 -*-
"""OCTAVE benchmark on Xenium (rule B): band scores of the 57 encoders per region -> median within specimen -> median over 13 specimens.
Writes <paper>/tab_xen57.tex (Table 2: 4 finest bands, ordered by the finest band), <paper>/figs/Fig4_octave.pdf
(a: score in each band for the 57 encoders; b: finest band on Xenium against Visium HD) and results/octave_bench.json."""
import json, glob, re, os, sys, itertools
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "scripts"); from _enc_names import NAMES
PC = sys.argv[1] if len(sys.argv) > 1 else "paper"; R = "results"
TS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
def spec(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    return m.group(1) if m else ("Lung" if "Human_Lung_Cancer_FFPE" in n else n)
def rule(rows):                       # rows: list of (region name, vector) -> median within specimen, then over specimens
    sp = {}
    for n, v in rows: sp.setdefault(spec(n), []).append(v)
    assert len(sp) == 13
    return np.median(np.vstack([np.median(np.vstack(v), axis=0) for v in sp.values()]), axis=0)
V = json.load(open(f"{R}/vhd57_bench.json")); PAR = json.load(open(f"{R}/encoder_params.json"))
B, O, SIG = {}, {}, None
for e in V:
    D = [json.load(open(f)) for f in glob.glob(f"{R}/" + ("blocks_xen_bands_base" if e == "hibou_l" else f"blocks_xen_bands_{e}") + "/*.json")]
    assert len(D) == 22, e
    B[e] = rule([(d["name"], np.array([d["pred"]["ridge"]["bands"][str(t)] for t in TS])) for d in D])
    O[e] = rule([(d["name"], np.array([d["pred"]["dom20"]["bands"][str(t)] for t in TS])) for d in D])
    if SIG is None: SIG = rule([(d["name"], np.array([d["sigma_um"][str(t)] for t in TS])) for d in D])
E = sorted(B, key=lambda e: -B[e][0])
# display names with citations, from the LABEL table of mk_tab_hest_rule.py
import ast
_t = ast.parse(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mk_tab_hest_rule.py")).read())
LABEL = next(ast.literal_eval(n.value) for n in _t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LABEL")
disp = {e: LABEL[e] for e in E}
fmt = lambda e: f"{disp[e]} & {PAR[e]['params'] / 1e6:.0f} & " + " & ".join(f"{B[e][k]:.3f}" for k in range(4))
med = np.median(np.vstack([np.round(B[e][:4], 3) for e in E]), axis=0)
MED = "\\textit{Median over 57} &  & " + " & ".join(f"{v:.3f}" for v in med)
w = [f"{SIG[k]:.1f}" for k in range(4)]
hdr = "Encoder & M & " + " & ".join(f"$\\beta_{k+1}$" for k in range(4))
sub = " & & " + " & ".join(f"{x}\\,$\\mu$m" for x in w)
out = ["\\begin{tabular}{@{}lrrrrr@{\\hspace{14pt}}|@{\\hspace{14pt}}lrrrrr@{}}", "\\toprule",
       hdr + " & " + hdr + " \\\\", sub + " &" + sub + " \\\\", "\\midrule"]
L, Rr = E[:29], E[29:]
for i in range(29):
    out.append(fmt(L[i]) + " & " + (fmt(Rr[i]) if i < len(Rr) else MED) + " \\\\")
out += ["\\bottomrule", "\\end{tabular}"]
open(f"{PC}/tab_xen57.tex", "w").write("\n".join(out) + "\n")
# numbers for the text
r3 = lambda v: round(float(v), 3)
fac = [r3(max(B[e][k] for e in E)) / r3(min(B[e][k] for e in E)) for k in range(4)]
xb1 = np.array([B[e][0] for e in E]); vb1 = np.array([V[e]["b1"] for e in E])
om = np.median(np.vstack([O[e] for e in E]), axis=0); bm = np.median(np.vstack([B[e] for e in E]), axis=0)
S = dict(widths=SIG.tolist(), factors_b1to4=fac, median_b1to4=med.tolist(), best=E[0], worst=E[-1],
         oracle_median=om.tolist(), model_median=bm.tolist(), spearman_xen_vhd_b1=float(spearmanr(xb1, vb1).statistic))
json.dump(dict(summary=S, encoders={e: dict(model=B[e].tolist(), oracle=O[e].tolist()) for e in E}), open(f"{R}/octave_bench.json", "w"), indent=1)
print(json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in S.items()}))
# Figure 4
MM = 1 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "pdf.fonttype": 42,
                     "font.size": 7, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
fig = plt.figure(figsize=(183 * MM, 60 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.42, left=0.065, right=0.985, top=0.80, bottom=0.19)
ax = fig.add_subplot(gs[0])
for e in E: ax.plot(SIG, B[e], "-", lw=0.6, color="#B8B8B8", zorder=1)
for e, col in ((E[0], "#B64342"), (E[-1], "#0F4D92")):
    ax.plot(SIG, B[e], "-o", lw=1.6, ms=2.6, color=col, zorder=3)
    ax.annotate(NAMES[e], (SIG[0], B[e][0]), textcoords="offset points", xytext=(-4, 0), ha="right", va="center", fontsize=6.3, color=col, fontweight="bold")
ax.set_xscale("log"); ax.set_xlim(SIG[0] * 0.55, SIG[-1] * 1.15)
ax.set_xlabel("band width $\\sigma$ (µm)", labelpad=1); ax.set_ylabel("{\\sc Octave} band score $\\beta_i$".replace("{\\sc Octave}", "OCTAVE"), labelpad=2)
ax.set_title(f"band scores of the {len(E)} encoders on Xenium", fontsize=7.4, pad=6)
ax.text(-0.10, 1.20, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
ax = fig.add_subplot(gs[1])
ax.scatter(xb1, vb1, s=10, color="#4D4D4D", linewidths=0)
lim = (0, 0.28); ax.plot(lim, lim, ls="--", lw=0.8, color="#767676"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("finest-band score on Xenium", labelpad=1); ax.set_ylabel("finest-band score on Visium HD", labelpad=2)
ax.set_title("Xenium against Visium HD", fontsize=7.4, pad=6)
ax.text(0.04, 0.96, f"Spearman {S['spearman_xen_vhd_b1']:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color="#4D4D4D")
ax.text(-0.22, 1.20, "b", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
HB = {o["enc"]: o["b1_model"] for o in json.load(open(f"{R}/hest_rule_numbers.json"))["encoders"]}
hb1 = np.array([HB[e] for e in E]); S["spearman_xen_hest_b1"] = float(spearmanr(xb1, hb1).statistic)
ax = fig.add_subplot(gs[2])
ax.scatter(xb1, hb1, s=10, color="#4D4D4D", linewidths=0)
ax.set_xlim(0, 0.28); ax.set_ylim(0.03, 0.11)
ax.set_xlabel("finest-band score on Xenium", labelpad=1); ax.set_ylabel("finest-band score on HEST", labelpad=2)
ax.set_title("Xenium against HEST", fontsize=7.4, pad=6)
ax.text(0.04, 0.96, f"Spearman {S['spearman_xen_hest_b1']:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color="#4D4D4D")
ax.text(-0.22, 1.20, "c", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
json.dump(dict(summary=S, encoders={e: dict(model=B[e].tolist(), oracle=O[e].tolist()) for e in E}), open(f"{R}/octave_bench.json", "w"), indent=1)
print("Spearman Xenium vs HEST finest band", round(S["spearman_xen_hest_b1"], 3))
fig.suptitle("The 57 encoders scored by OCTAVE", fontsize=9.2, y=0.985, fontweight="bold")
fig.savefig(f"{PC}/figs/Fig4_octave.pdf"); fig.savefig(f"{PC}/figs/Fig4_octave.png", dpi=200)

# -*- coding: utf-8 -*-
"""one table for the 57 frozen encoders on the 3 datasets (rule B), replacing tab_xen57, tab_vhd57 and tab_hest57b1.
Band scores beta_1..beta_4 of ridge on each encoder's features:
  Xenium     per region -> median within specimen -> median over the 13 specimens (results/octave_bench.json)
  Visium HD  per section -> median over the 2 sections (results/blocks_vhd_bands_*/)
  HEST       per sample -> median over the 72 samples (results/hest_effres_ps_selfull_*.json)
Band widths are summarised the same way. Encoders are ordered by beta_1 on Xenium; the highest value of each column is bold.
Writes <paper>/tab_bench57.tex and results/bench57.json, and prints the numbers quoted in the text."""
import json, glob, re, os, sys, itertools
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "scripts"); from _enc_names import NAMES
PC = sys.argv[1] if len(sys.argv) > 1 else "paper"; R = "results"
TS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
OB = json.load(open(f"{R}/octave_bench.json")); PAR = json.load(open(f"{R}/encoder_params.json"))
E = list(OB["encoders"]); assert len(E) == 57
X = {e: np.array(OB["encoders"][e]["model"]) for e in E}; XS = np.array(OB["summary"]["widths"])
V, VS = {}, None
for e in E:
    D = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/blocks_vhd_bands_{e}/*.json"))]; assert len(D) == 2, e
    V[e] = np.median(np.vstack([[d["pred"]["ridge"]["bands"][str(t)] for t in TS] for d in D]), axis=0)
    s = np.median(np.vstack([[d["sigma_um"][str(t)] for t in TS] for d in D]), axis=0)
    VS = s if VS is None else VS; assert np.allclose(VS, s)
H, HS = {}, None
for e in E:
    F = json.load(open(f"{R}/hest_effres_ps_selfull_{e}.json")); sids = sorted(F["per_sample_band_pcc"]); assert len(sids) == 72, e
    nb = len(F["per_sample_band_pcc"][sids[0]])
    H[e] = np.median(np.array([[F["per_sample_band_pcc"][s][str(t)] for t in TS[:nb]] for s in sids]), axis=0)
    s = np.median(np.array([[F["per_sample_sigma_um"][s][str(t)] for t in TS[:nb]] for s in sids]), axis=0)
    HS = s if HS is None else HS; assert np.allclose(HS, s)
HB = {o["enc"]: o["b1_model"] for o in json.load(open(f"{R}/hest_rule_numbers.json"))["encoders"]}
assert max(abs(H[e][0] - HB[e]) for e in E) < 1e-9, "HEST beta_1 differs from hest_rule_numbers.json"
VB = json.load(open(f"{R}/vhd57_bench.json"))
assert max(abs(V[e][0] - VB[e]["b1"]) for e in E) < 1e-9, "Visium HD beta_1 differs from vhd57_bench.json"
E = sorted(E, key=lambda e: -X[e][0])
DS = [("Xenium", X, XS), ("Visium HD", V, VS), ("HEST", H, HS)]

# display names with citations, from the LABEL table of mk_tab_hest_rule.py
import ast
_t = ast.parse(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mk_tab_hest_rule.py")).read())
LABEL = next(ast.literal_eval(n.value) for n in _t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LABEL")
disp = {e: LABEL[e] for e in E}
rnd = {name: {e: np.round(S[e][:4], 3) for e in E} for name, S, _ in DS}
best = {name: [max(rnd[name][e][k] for e in E) for k in range(4)] for name, _, _ in DS}
def cell(name, e, k):
    v = rnd[name][e][k]; s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if v == best[name][k] else s
out = ["\\begin{tabular}{@{}lr" + "rrrr" * 3 + "@{}}", "\\toprule",
       " & & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{n}}}" for n, _, _ in DS) + " \\\\",
       "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(l){11-14}",
       "Encoder & M & " + " & ".join(f"$\\beta_{k+1}$" for _ in DS for k in range(4)) + " \\\\",
       " & & " + " & ".join(f"{w[k]:.1f}\\,$\\mu$m" for _, _, w in DS for k in range(4)) + " \\\\", "\\midrule"]
for e in E:
    out.append(f"{disp[e]} & {PAR[e]['params'] / 1e6:.0f} & " + " & ".join(cell(n, e, k) for n, _, _ in DS for k in range(4)) + " \\\\")
med = {n: np.median(np.vstack([rnd[n][e] for e in E]), axis=0) for n, _, _ in DS}
out += ["\\midrule", "\\textit{Median over 57} & & " + " & ".join(f"{med[n][k]:.3f}" for n, _, _ in DS for k in range(4)) + " \\\\",
        "\\bottomrule", "\\end{tabular}"]
open(f"{PC}/tab_bench57.tex", "w").write("\n".join(out) + "\n")

# numbers for the text, from the printed (rounded) values
S = {"widths": {n: w.tolist() for n, _, w in DS}, "median_b1to4": {n: med[n].tolist() for n in med}}
for n, _, _ in DS:
    b1 = {e: rnd[n][e][0] for e in E}
    hi, lo = max(E, key=lambda e: b1[e]), min(E, key=lambda e: b1[e])
    S[n] = dict(best=hi, worst=lo, b1_max=float(b1[hi]), b1_min=float(b1[lo]),
                factors_b1to4=[round(float(best[n][k] / min(rnd[n][e][k] for e in E)), 2) for k in range(4)])
sp = {}
for (a, A, _), (b, B, _) in itertools.combinations(DS, 2):
    sp[f"{a} vs {b}"] = round(float(spearmanr([A[e][0] for e in E], [B[e][0] for e in E]).statistic), 3)
S["spearman_b1"] = sp
grp = {"pathology": [], "general": [], "distilled": []}
GEN = {"dinov2_base", "dinov2_large", "dinov2_giant", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vith16", "clip_vitl14", "siglip2"}
DIS = {"litevirchow2", "h0_mini", "litefm", "litefm_l", "litefm_s", "distillpath_is16", "distillpath_ks16", "pathryoshka_b"}
for n, _, _ in DS:
    path = [rnd[n][e][0] for e in E if e not in GEN and e != "biomedclip"]
    gen = [rnd[n][e][0] for e in E if e in GEN]
    S[n]["pathology_vs_general_b1"] = [float(np.median(path)), float(np.median(gen)), len(path), len(gen)]
json.dump(dict(summary=S, encoders={e: {n: M[e].tolist() for n, M, _ in DS} for e in E}), open(f"{R}/bench57.json", "w"), indent=1)
print(json.dumps(S, indent=1, default=lambda v: np.round(v, 3).tolist()))

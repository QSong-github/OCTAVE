# -*- coding: utf-8 -*-
"""Visium HD column of Table 2 over the 57 encoders , the counterpart of agg_xen_ctrl57.py.
Inputs: results/vhd_ctrl57 and results/vhd_ind57 (vhd_ctrl57.py, cheap and ind modes, Xenium protocol) and
results/blocks_vhd_bands_<enc> (trained model and domain oracle).
Each value is the median over the two sections (two patients), ratios formed per section, then the median over the 57 encoders.
Output: results/vhd_ctrl57_summary.json"""
import json, os
import numpy as np

R = os.environ.get("OCTAVE_RESULTS", "results")
SECS = ["Visium_HD_Human_Colon_Cancer_P2", "Visium_HD_Human_Colon_Cancer_P5"]
encs = sorted(os.listdir(f"{R}/vhd_ctrl57"))
assert len(encs) == 57
out, missing = {}, []
for e in encs:
    M, D, T, I, C, Z = {}, {}, {}, {}, {}, {}
    for s in SECS:
        b = json.load(open(f"{R}/blocks_vhd_bands_{e}/{s}.json"))["pred"]; c = json.load(open(f"{R}/vhd_ctrl57/{e}/{s}.json"))
        M[s], D[s] = b["ridge"]["pcc"], b["dom20"]["pcc"]
        T[s], C[s], Z[s] = c["trainonly_image"], c["oracle_coord"], c["oracle_random_matched"]
        f = f"{R}/vhd_ind57/{e}/{s}.json"
        if os.path.exists(f): I[s] = json.load(open(f))["inductive"]
    v = {"ridge": float(np.median([M[s] for s in SECS]))}
    for k, X in (("img", D), ("train", T), ("ind", I), ("coord", C), ("rand", Z)):
        if len(X) < len(SECS): missing.append((e, k)); continue
        v[k] = float(np.median([X[s] for s in SECS])); v[k + "_frac"] = float(np.median([X[s] / M[s] for s in SECS]))
    out[e] = v
keys = sorted({k for v in out.values() for k in v})
med = {k: float(np.median([out[e][k] for e in encs if k in out[e]])) for k in keys}
n = {k: sum(k in out[e] for e in encs) for k in keys}
below = sum(out[e]["img_frac"] < 1 for e in encs)
json.dump({"median_over_57": med, "n_encoders": n, "oracle_below_model": below, "missing": missing, "encoders": out},
          open(f"{R}/vhd_ctrl57_summary.json", "w"), indent=1)
print("median over encoders:", {k: round(x, 4) for k, x in med.items()})
print("encoders per value:", n, "| oracle below model:", below, "| missing:", missing)

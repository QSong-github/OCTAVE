# -*- coding: utf-8 -*-
"""Xenium column of Table 2 over the 57 encoders .
Inputs: results/xen_ctrl57 and results/xen_ind57 (xen_ctrl57.py, cheap and ind modes), results/xen_controls (coordinate partition,
which does not use the image) and results/blocks_xen_bands_<enc> (trained model and domain oracle of Table 1; Hibou-L is _base).
Each value is aggregated as in Table 2: per region (ratios to the encoder's own ridge per region), median over the regions of each
specimen, median over the 13 specimens, then the median over the 57 encoders. The oracle row uses Table 1's oracle; the oracle
recomputed in xen_ctrl57 is reported only as a check (K-means can reach another local optimum on single regions).
Output: results/xen_ctrl57_summary.json"""
import glob, json, os, re
import numpy as np

R = os.environ.get("OCTAVE_RESULTS", "results")


def specimen(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    if "Human_Lung_Cancer_FFPE" in n: return "Lung"   # v1 and Prime 5K come from the same block of one donor
    return m.group(1) if m else n


def spec_meds(d):
    g = {}
    for r, v in d.items(): g.setdefault(specimen(r), []).append(v)
    return {k: float(np.median(v)) for k, v in g.items()}


def agg(d):
    return float(np.median(list(spec_meds(d).values())))


regions = sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{R}/xen_controls/*.json"))
coord = {r: json.load(open(f"{R}/xen_controls/{r}.json"))["K20"]["oracle_coord"]["pcc"] for r in regions}
encs = sorted(os.listdir(f"{R}/xen_ctrl57"))
assert len(encs) == 57 and len(regions) == 22
out, check = {}, {"oracle_region_absdiff": [], "learnable_vs_oracle_specimen_reldiff": 0.0}
for e in encs:
    bd = f"{R}/blocks_xen_bands_base" if e == "hibou_l" else f"{R}/blocks_xen_bands_{e}"
    M, D, O, T, I, Z = {}, {}, {}, {}, {}, {}
    for r in regions:
        b = json.load(open(f"{bd}/{r}.json"))["pred"]
        c = json.load(open(f"{R}/xen_ctrl57/{e}/{r}.json")); i = json.load(open(f"{R}/xen_ind57/{e}/{r}.json"))
        M[r], D[r] = b["ridge"]["pcc"], b["dom20"]["pcc"]
        O[r], T[r], Z[r], I[r] = c["oracle_image"], c["trainonly_image"], c["oracle_random_matched"], i["inductive"]
    v = {"ridge": agg(M)}
    for k, X in (("img", D), ("train", T), ("ind", I), ("coord", coord), ("rand", Z)):
        v[k] = agg(X); v[k + "_frac"] = agg({r: X[r] / M[r] for r in regions})
    v["img_recomputed"] = agg(O)
    check["oracle_region_absdiff"] += [abs(O[r] - D[r]) for r in regions]
    so, st = spec_meds(O), spec_meds(T)
    check["learnable_vs_oracle_specimen_reldiff"] = max(check["learnable_vs_oracle_specimen_reldiff"], max(abs(st[s] - so[s]) / so[s] for s in so))
    out[e] = v
med = {k: float(np.median([out[e][k] for e in encs])) for k in out["hibou_l"]}
rng = {k: [float(min(out[e][k] for e in encs)), float(max(out[e][k] for e in encs))] for k in out["hibou_l"] if k.endswith("_frac")}
a = np.array(check["oracle_region_absdiff"])
chk = {"oracle_region_absdiff_median": float(np.median(a)), "oracle_region_absdiff_max": float(a.max()),
       "oracle_regions_over_1e-3": int((a > 1e-3).sum()),
       "oracle_encoder_absdiff_max": float(max(abs(out[e]["img_recomputed"] - out[e]["img"]) for e in encs)),
       "learnable_vs_oracle_specimen_reldiff_max": check["learnable_vs_oracle_specimen_reldiff"]}
json.dump({"median_over_57": med, "range_over_57": rng, "check": chk, "encoders": out}, open(f"{R}/xen_ctrl57_summary.json", "w"), indent=1)
print("median over 57:", {k: round(x, 4) for k, x in med.items()})
print("check:", chk)

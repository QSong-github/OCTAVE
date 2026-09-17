# -*- coding: utf-8 -*-
"""iStar（端到端超分方法）与 ridge / 上下文 ridge / 域 oracle / 可学习域预测器在同一 half 折、同一批测试 bin 上的比较（results/istar_xen/*.json）。
聚合：区域→标本中位数→跨标本中位数；报告标量 PCC、最细带 β1、最细带方差份额，以及 iStar 相对 ridge 的差与标本计数。输出 results/istar_numbers.json 与 paper/tab_istar.tex。"""
import json, glob, os, numpy as np
R = os.environ.get("S4ST_RESULTS", "results")
SPEC = ["Human_Breast_Biomarkers_S1", "Human_Breast_Biomarkers_S2", "Human_Breast_Biomarkers_S3", "Human_Breast_Biomarkers_S4",
        "Xenium_Prime_Cervical", "Xenium_Prime_Ovarian", "Xenium_V1_Human_Kidney", "Xenium_V1_Human_Ovary",
        "Lung", "Xenium_Prime_Breast_Cancer", "Xenium_Prime_Human_Prostate", "Xenium_Prime_Human_Skin", "Xenium_Prime_Human_Lymph_Node"]   # 2026-09-14 新增 5 个标本
def sp(n):
    if "Human_Lung_Cancer_FFPE" in n: return "Lung"   # v1 与 Prime 5K 同一供体同一组织块，按一个标本计
    for s in SPEC:
        if n.startswith(s): return s
    raise SystemExit(f"未知区域 {n}：请把它的标本加进 SPEC")
D = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(f"{R}/istar_xen/*.json")}
H = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(f"{R}/istar_xen_hipt/*.json")}
for n in list(D):
    if n in H: D[n]["ridge_hipt"] = dict(pcc=H[n]["ridge_hipt_official"]["pcc"], beta1=H[n]["ridge_hipt_official"]["beta1"], fine_var_share=float("nan")); D[n]["ridge_hipt_1e4"] = dict(pcc=H[n]["ridge_hipt_1e4"]["pcc"], beta1=H[n]["ridge_hipt_1e4"]["beta1"], fine_var_share=float("nan"))
HAVE_H = all(n in H for n in D)
def specmed(fn):
    per = {}
    for n, d in D.items(): per.setdefault(sp(n), []).append(fn(d))
    m = [float(np.median(v)) for v in per.values()]; return dict(med=float(np.median(m)), lo=min(m), hi=max(m), n=len(m), above0=int(sum(x > 0 for x in m)))
KEYS = [("istar", "iStar (super-resolution, end-to-end)")] + ([("ridge_hipt", "Ridge on iStar's own HIPT features (official $\\lambda$)"), ("ridge_hipt_1e4", "Ridge on iStar's own HIPT features ($\\lambda=10^4$)")] if HAVE_H else []) + [("ridge", "Ridge on frozen Hibou-L features"), ("context_ridge", "Ridge with spatial context"), ("trainonly", "Image partition, means from training blocks"), ("oracle", "Domain oracle")]
out = {"n_regions": len(D), "n_missing_genes": max(d["n_missing"] for d in D.values()) if D else None}
for k, _ in KEYS:
    out[k] = dict(pcc=specmed(lambda d: d[k]["pcc"]), beta1=specmed(lambda d: d[k]["beta1"]), fine_share=specmed(lambda d: d[k]["fine_var_share"]))
out["truth_fine_share"] = specmed(lambda d: d["truth_fine_var_share"])
out["istar_minus_ridge"] = dict(pcc=specmed(lambda d: d["istar"]["pcc"] - d["ridge"]["pcc"]), beta1=specmed(lambda d: d["istar"]["beta1"] - d["ridge"]["beta1"]))
out["istar_ratio_vs_oracle"] = specmed(lambda d: ((d["istar"]["beta1"] - d["oracle"]["beta1"]) / d["istar"]["beta1"]) / ((d["istar"]["pcc"] - d["oracle"]["pcc"]) / d["istar"]["pcc"]) if abs(d["istar"]["pcc"] - d["oracle"]["pcc"]) > 1e-9 else np.nan)
json.dump(out, open(f"{R}/istar_numbers.json", "w"), indent=1)
L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Predictor (Xenium, half split, $200$ genes) & PCC & $\beta_1$ & finest-band variance share \\", r"\midrule"]
for k, lab in KEYS:
    v = out[k]; fs = '--' if not np.isfinite(v['fine_share']['med']) else f"{100*v['fine_share']['med']:.1f}\\%"; L.append(f"{lab} & {v['pcc']['med']:.3f} [{v['pcc']['lo']:.3f}, {v['pcc']['hi']:.3f}] & {v['beta1']['med']:.3f} [{v['beta1']['lo']:.3f}, {v['beta1']['hi']:.3f}] & {fs} \\\\")
L += [r"\midrule", f"Measured field & -- & -- & {100*out['truth_fine_share']['med']:.1f}\\% \\\\", r"\bottomrule", r"\end{tabular}"]
open("paper/tab_istar.tex", "w").write("\n".join(L) + "\n")
print(f"regions={len(D)} | " + " | ".join(f"{k} {out[k]['pcc']['med']:.3f}/{out[k]['beta1']['med']:.3f}" for k, _ in KEYS) + f" | iStar−ridge PCC {out['istar_minus_ridge']['pcc']['med']:+.3f} (标本>0: {out['istar_minus_ridge']['pcc']['above0']}/{out['istar_minus_ridge']['pcc']['n']}) β1 {out['istar_minus_ridge']['beta1']['med']:+.3f} (>0: {out['istar_minus_ridge']['beta1']['above0']}/{out['istar_minus_ridge']['beta1']['n']}) | 缺基因 {out['n_missing_genes']}")

# -*- coding: utf-8 -*-
"""归纳式分区预测器与上下文岭回归的汇总。Xenium：results/xen_inductive/*.json（标本级中位）；HEST：results/hest_inductive_{enc}.json + hest_rsel_ps（队列级中位再跨编码器）。"""
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
out = {}
X = {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(f"{R}/xen_inductive/*.json")}
if X:
    def specmed(fn):
        per = {}
        for n, d in X.items(): per.setdefault(sp(n), []).append(fn(d))
        m = [float(np.median(v)) for v in per.values()]; return dict(med=float(np.median(m)), lo=min(m), hi=max(m), n=len(m), above1=int(sum(x > 1 for x in m)))
    xen = {}
    for k in ["ridge", "context_ridge", "inductive_partition", "transductive_partition", "oracle"]:
        xen[k] = dict(pcc=specmed(lambda d: d[k]["pcc"]), beta1=specmed(lambda d: d[k]["beta1"]), frac=specmed(lambda d: d[k]["pcc"] / d["ridge"]["pcc"]))
        if k != "ridge": xen[k]["ratio"] = specmed(lambda d: d[k]["d_fine"] / d[k]["d_scalar"] if abs(d[k]["d_scalar"]) > 1e-9 else np.nan)
    xen["n_regions"] = len(X); out["xen"] = xen
    print(f"Xenium ({len(X)} 区域，标本级中位 [范围])")
    for k in ["ridge", "context_ridge", "inductive_partition", "transductive_partition", "oracle"]:
        v = xen[k]; extra = f"  Δ_fine/Δ_scalar {v['ratio']['med']:.2f} [{v['ratio']['lo']:.2f}, {v['ratio']['hi']:.2f}] >1: {v['ratio']['above1']}/{v['ratio']['n']}" if k != "ridge" else ""
        print(f"  {k:24s} PCC {v['pcc']['med']:.3f} [{v['pcc']['lo']:.3f}, {v['pcc']['hi']:.3f}]  占ridge {v['frac']['med']:.2f}  β1 {v['beta1']['med']:.3f} [{v['beta1']['lo']:.3f}, {v['beta1']['hi']:.3f}]{extra}")
encs = sorted(f.split("hest_inductive_")[1][:-5] for f in glob.glob(f"{R}/hest_inductive_*.json"))
if encs:
    rows = []
    for e in encs:
        I = json.load(open(f"{R}/hest_inductive_{e}.json"))["samples"]; P = json.load(open(f"{R}/hest_rsel_ps_{e}.json"))["per_sample_pcc"]; C = json.load(open(f"{R}/hest_controls_{e}.json"))["samples"]
        coh = {}
        for s, v in I.items():
            if v.get("inductive") is None or s not in P: continue
            coh.setdefault(v["cohort"], []).append((v["inductive"], P[s], C[s]["trainonly_image"]))
        cm = np.array([np.median(np.array(v), 0) for v in coh.values()])
        rows.append(dict(enc=e, inductive=float(np.median(cm[:, 0])), ridge=float(np.median(cm[:, 1])), transductive=float(np.median(cm[:, 2])), ind_over_ridge=float(np.median(cm[:, 0] / cm[:, 1])), trd_over_ridge=float(np.median(cm[:, 2] / cm[:, 1])), coh_ind_gt_ridge=int((cm[:, 0] > cm[:, 1]).sum())))
    A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "enc"}
    out["hest"] = {k: dict(med=float(np.median(A[k])), lo=float(A[k].min()), hi=float(A[k].max())) for k in A}; out["hest"]["n_enc"] = len(rows)
    print(f"HEST ({len(rows)} 编码器，队列级中位再跨编码器中位)：inductive {np.median(A['inductive']):.3f} 占 ridge {np.median(A['ind_over_ridge']):.2f} [{A['ind_over_ridge'].min():.2f}, {A['ind_over_ridge'].max():.2f}]；transductive 占 ridge {np.median(A['trd_over_ridge']):.2f}；inductive>ridge 队列数中位 {np.median(A['coh_ind_gt_ridge']):.0f}/10")
json.dump(out, open(f"{R}/inductive_numbers.json", "w"), indent=1)

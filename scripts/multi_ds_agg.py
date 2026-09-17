# -*- coding: utf-8 -*-
"""第四份审稿意见 #5：11 个编码器的下游读数与 标量 PCC / 最细带 β1 的跨编码器关系。
输入 results/multi_ds/{tower}_{region}.json（multi_downstream.py）。聚合：区域→标本中位数→跨标本中位数，得到每个编码器一行；
再在 11 个编码器上算每个读数与 PCC、与 β1 的 Spearman（numpy 实现，置换 P），并做「逐标本」版本：每个标本内 11 个编码器的秩相关，再取中位。
输出 results/multi_ds_summary.json 与 paper/tab_multids.tex。"""
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
NAME = {"hibou_l": "Hibou-L", "uni_v2": "UNI v2", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1", "gigapath": "Prov-GigaPath", "phikon_v2": "Phikon-v2", "conch_v15": "CONCH v1.5", "kaiko_vits16": "Kaiko-S16", "ctranspath": "CTransPath", "h0_mini": "H0-mini", "midnight12k": "Midnight-12k"}
READ = [("svg_top_jaccard", "SVG top-$k$ Jaccard", +1), ("svg_rank_rho", "SVG rank $\\rho$", +1), ("hotspot_jaccard", "Hotspot Jaccard", +1), ("boundary_shift_um", "Boundary shift ($\\mu$m)", -1), ("coloc_preserve", "Co-localisation preserved", +1), ("hotspot_recall_selectivity", "Hotspot recall selectivity", +1)]
def rank(a):
    a = np.asarray(a, float); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(len(a)); _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True); s_ = np.zeros(len(cnt)); np.add.at(s_, inv, r); return s_[inv] / cnt[inv]
def spearman(x, y, nperm=20000, seed=0):
    rx, ry = rank(x), rank(y); rho = np.corrcoef(rx, ry)[0, 1]; rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(nperm)]); return float(rho), float((np.abs(perm) >= abs(rho) - 1e-12).mean())
D = {}
for f in glob.glob(f"{R}/multi_ds/*.json"):
    d = json.load(open(f)); D.setdefault(d["tower"], {})[d["name"]] = d
towers = sorted(D); NREG = max((len(v) for v in D.values()), default=0); complete = [t for t in towers if len(D[t]) == NREG]
print(f"编码器 {len(towers)}，其中 {NREG}/{NREG} 区域齐的 {len(complete)}: {complete}")
def specmed(t, fn):
    per = {}
    for n, d in D[t].items():
        v = fn(d)
        if v is not None: per.setdefault(sp(n), []).append(v)
    return float(np.median([np.median(v) for v in per.values()])) if per else None
rows = {t: dict(pcc=specmed(t, lambda d: d["scores"]["pcc"]), beta1=specmed(t, lambda d: d["scores"]["beta1"]), **{k: specmed(t, lambda d, k=k: d["readouts"].get(k)) for k, _, _ in READ}) for t in complete}
out = dict(encoders=rows, correlations={})
if len(complete) >= 5:
    pcc = np.array([rows[t]["pcc"] for t in complete]); b1 = np.array([rows[t]["beta1"] for t in complete])
    print(f"{'readout':28s} {'rho(PCC)':>9s} {'P':>7s} {'rho(β1)':>9s} {'P':>7s}   per-specimen median rho(PCC) / rho(β1)")
    for k, lab, sign in READ:
        v = np.array([rows[t][k] for t in complete], float) * sign
        if np.isnan(v).any() or np.ptp(v) < 1e-12 or np.ptp(pcc) < 1e-12: print(f'{lab:28s} 跨编码器无变化，跳过'); continue
        r1, p1 = spearman(pcc, v); r2, p2 = spearman(b1, v)
        # 逐标本：每个标本内 11 个编码器的秩相关
        per = {}
        for t in complete:
            for n, d in D[t].items():
                vv = d["readouts"].get(k)
                if vv is None: continue
                per.setdefault(sp(n), {}).setdefault(t, []).append((d["scores"]["pcc"], d["scores"]["beta1"], vv * sign))
        ws_p, ws_b = [], []
        for s_, td in per.items():
            if len(td) < 5: continue
            P_ = np.array([np.median([x[0] for x in td[t]]) for t in td]); B_ = np.array([np.median([x[1] for x in td[t]]) for t in td]); V_ = np.array([np.median([x[2] for x in td[t]]) for t in td])
            ws_p.append(np.corrcoef(rank(P_), rank(V_))[0, 1]); ws_b.append(np.corrcoef(rank(B_), rank(V_))[0, 1])
        out["correlations"][k] = dict(rho_pcc=r1, p_pcc=p1, rho_beta1=r2, p_beta1=p2, per_specimen_rho_pcc=float(np.median(ws_p)), per_specimen_rho_beta1=float(np.median(ws_b)), n_specimens=len(ws_p), beta1_wins=int(sum(b > p for p, b in zip(ws_p, ws_b))))
        print(f"{lab:28s} {r1:9.2f} {p1:7.3f} {r2:9.2f} {p2:7.3f}   {np.median(ws_p):.2f} / {np.median(ws_b):.2f}  (β1 更高的标本 {out['correlations'][k]['beta1_wins']}/{len(ws_p)})")
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Downstream readout & $r_s$ with $\mathrm{PCC}$ & $P$ & $r_s$ with $\beta_1$ & $P$ & per-specimen $r_s$ ($\mathrm{PCC}$ / $\beta_1$) & specimens where $\beta_1$ correlates more \\", r"\midrule"]
    for k, lab, sign in READ:
        if k not in out["correlations"]: continue
        c = out["correlations"][k]; L.append(f"{lab} & {c['rho_pcc']:.2f} & {c['p_pcc']:.3f} & {c['rho_beta1']:.2f} & {c['p_beta1']:.3f} & {c['per_specimen_rho_pcc']:.2f} / {c['per_specimen_rho_beta1']:.2f} & {c['beta1_wins']}/{c['n_specimens']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]; open("paper/tab_multids.tex", "w").write("\n".join(L) + "\n")
json.dump(out, open(f"{R}/multi_ds_summary.json", "w"), indent=1)

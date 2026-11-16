# -*- coding: utf-8 -*-
"""附录中由本地 57 个编码器结果即可算出的部分，统一汇总规则：每个编码器逐区域 → 标本内中位数 → 跨 13 个标本中位数，
再跨 57 个编码器取中位数；按标本的值与计数取每个标本上 57 个编码器的标本内中位数的中位数；差距与比值由中位数算。
写 <论文目录>/tab_specimens.tex（逐标本差距；种子一行在 xen57_robust 结果到齐后由 supp57_robust.py 补上）
与 results/supp57_local.json（H.1 可靠性一节、I.4 节用到的数）。用法：在项目根目录运行 python3 scripts/supp57_local.py <论文目录>。"""
import json, glob, os, sys, numpy as np
from math import comb

R = os.environ.get("OCTAVE_RESULTS", "results"); PAPER = sys.argv[1] if len(sys.argv) > 1 else "paper"
SPEC = ["Human_Breast_Biomarkers_S1", "Human_Breast_Biomarkers_S2", "Human_Breast_Biomarkers_S3", "Human_Breast_Biomarkers_S4",
        "Xenium_Prime_Cervical", "Xenium_Prime_Ovarian", "Xenium_V1_Human_Kidney", "Xenium_V1_Human_Ovary",
        "Lung", "Xenium_Prime_Breast_Cancer", "Xenium_Prime_Human_Prostate", "Xenium_Prime_Human_Skin", "Xenium_Prime_Human_Lymph_Node"]
SHORT = {"Human_Breast_Biomarkers_S1": "Breast S1", "Human_Breast_Biomarkers_S2": "Breast S2", "Human_Breast_Biomarkers_S3": "Breast S3",
         "Human_Breast_Biomarkers_S4": "Breast S4", "Xenium_Prime_Cervical": "Cervical (Prime)", "Xenium_Prime_Ovarian": "Ovarian (Prime)",
         "Xenium_V1_Human_Kidney": "Kidney (V1)", "Xenium_V1_Human_Ovary": "Ovary (V1)", "Lung": "Lung", "Xenium_Prime_Breast_Cancer": "Breast (Prime)",
         "Xenium_Prime_Human_Prostate": "Prostate (Prime)", "Xenium_Prime_Human_Skin": "Skin (Prime)", "Xenium_Prime_Human_Lymph_Node": "Lymph node (Prime)"}
sp = lambda n: "Lung" if "Human_Lung_Cancer_FFPE" in n else next(s for s in SPEC if n.startswith(s))
load = lambda d: {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(f"{R}/{d}/*.json")}
r3 = lambda x: round(x, 3)
pct1 = lambda o, m: round(100 * (1 - o / m), 1)


def signp(k, n):
    k = min(k, n - k); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def per_spec(vals):
    per = {}
    for n, v in vals.items(): per.setdefault(sp(n), []).append(v)
    assert len(per) == 13
    return {s: float(np.median(v)) for s, v in per.items()}


ENC = sorted(json.load(open(f"{R}/xen_multi_rank.json"))["encoders"]); assert len(ENC) == 57
D = {e: load("blocks_xen_bands_base" if e == "hibou_l" else f"blocks_xen_bands_{e}") for e in ENC}
C = load("xen_band_ceiling"); c1 = {n: C[n]["c_full"]["1"] for n in C}; assert len(C) == 22


def summ(fn):
    """返回 (跨编码器中位数, {标本: 跨编码器中位数}, {编码器: 跨标本中位数})。"""
    S = {e: per_spec({n: fn(e, x) for n, x in D[e].items()}) for e in ENC}
    enc = {e: float(np.median(list(S[e].values()))) for e in ENC}
    return float(np.median(list(enc.values()))), {s: float(np.median([S[e][s] for e in ENC])) for s in SPEC}, enc


pm, pms, pme = summ(lambda e, x: x["pred"]["ridge"]["pcc"]); po, pos, poe = summ(lambda e, x: x["pred"]["dom20"]["pcc"])
bm, bms, bme = summ(lambda e, x: x["pred"]["ridge"]["bands"]["1"]); bo, bos, boe = summ(lambda e, x: x["pred"]["dom20"]["bands"]["1"])
nreg = {}
for n in D[ENC[0]]: nreg[sp(n)] = nreg.get(sp(n), 0) + 1
rows = []
for s in SPEC:
    a, b = r3(pms[s]), r3(pos[s]); dp = pct1(b, a); df = pct1(bos[s], bms[s]); rows.append((SHORT[s], nreg[s], a, b, dp, df, df / dp))
rows.sort(key=lambda z: -z[6])
a, b = r3(pm), r3(po); DP = pct1(b, a); DF = pct1(bo, bm); RT = DF / DP
agree = sum(z[5] > z[4] for z in rows)
L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
     r"Specimen & Regions & $\mathrm{PCC}_{\mathrm{mod}}$ & $\mathrm{PCC}_{\mathrm{dom}}$ & $\Delta_{\mathrm{PCC}}$ & $\Delta_{\mathrm{fine}}$ & $\Delta_{\mathrm{fine}}/\Delta_{\mathrm{PCC}}$ \\", r"\midrule"]
for n_, k_, a_, b_, dp_, df_, rt_ in rows: L.append(f"{n_} & {k_} & {a_:.3f} & {b_:.3f} & {dp_:.1f}\\% & {df_:.1f}\\% & {rt_:.2f} \\\\")
L += [r"\midrule", f"All $13$ specimens & $22$ & {a:.3f} & {b:.3f} & {DP:.1f}\\% & {DF:.1f}\\% & \\textbf{{{RT:.2f}}} \\\\", r"\bottomrule", r"\end{tabular}"]
open(f"{PAPER}/tab_specimens.tex", "w").write("\n".join(L) + "\n")
OUT = dict(spec=dict(pcc_mod=a, pcc_dom=b, d_pcc=DP, d_fine=DF, ratio=RT, agree=agree, P=signp(agree, 13),
                     ratio_min=min(z[6] for z in rows), ratio_max=max(z[6] for z in rows),
                     ratio_min_spec=min(rows, key=lambda z: z[6])[0], ratio_max_spec=max(rows, key=lambda z: z[6])[0], b1_mod=bm, b1_dom=bo))
# I.4：分母（训练模型的分数）在每个标本上离 0 多远
OUT["gaps_denominator"] = dict(spec_min_pcc=min(pms.values()), spec_min_b1=min(bms.values()), enc_spec_min_b1=min(min(per_spec({n: x["pred"]["ridge"]["bands"]["1"] for n, x in D[e].items()}).values()) for e in ENC),
                               enc_spec_min_b1_encoder=min(ENC, key=lambda e: min(per_spec({n: x["pred"]["ridge"]["bands"]["1"] for n, x in D[e].items()}).values())))
# H.1 可靠性：β1/√c 与最细带方差占比（预测的占比取 57 个编码器的中位数；实测图的占比与编码器无关）
cb_m = summ(lambda e, x: x["pred"]["ridge"]["bands"]["1"] / np.sqrt(c1[x["name"]]))[0]
cb_o = summ(lambda e, x: x["pred"]["dom20"]["bands"]["1"] / np.sqrt(c1[x["name"]]))[0]
sm, sms, _ = summ(lambda e, x: x["pred"]["ridge"]["fine_var_share"]); so, sos, _ = summ(lambda e, x: x["pred"]["dom20"]["fine_var_share"])
T = D[ENC[0]]
raw = per_spec({n: T[n]["pred"]["truth"]["fine_var_share"] for n in T}); sig = per_spec({n: T[n]["pred"]["truth"]["fine_var_share"] * c1[n] / C[n]["scalar_c_full"] for n in T})
raw_m, sig_m = float(np.median(list(raw.values()))), float(np.median(list(sig.values())))
OUT["reliab_text"] = dict(b1_model_corr=cb_m, b1_oracle_corr=cb_o, share_model=sm, share_oracle=so, share_raw=raw_m, share_sig=sig_m,
                          share_sig_min=min(sig.values()), share_sig_max=max(sig.values()),
                          fac_model_sig=sig_m / sm, fac_oracle_sig=sig_m / so, fac_model_raw=raw_m / sm, fac_oracle_raw=raw_m / so,
                          fac_sig_min=min(min(sig[s] / sms[s], sig[s] / sos[s]) for s in SPEC), fac_sig_max=max(max(sig[s] / sms[s], sig[s] / sos[s]) for s in SPEC),
                          fac_raw_min=min(min(raw[s] / sms[s], raw[s] / sos[s]) for s in SPEC), fac_raw_max=max(max(raw[s] / sms[s], raw[s] / sos[s]) for s in SPEC))
json.dump(OUT, open(f"{R}/supp57_local.json", "w"), indent=1)
print("\n".join(L)); print(json.dumps({k: v for k, v in OUT.items() if k != "spec"}, indent=1, default=lambda x: round(float(x), 4)))
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in OUT["spec"].items()})

# ═══ H.2 Xenium 划分对照（tab_controls_xen），57 个编码器的中位数；坐标划分与编码器无关
XC = load("xen_controls"); CT = {e: load(f"xen_ctrl57/{e}") for e in ENC}; IN = {e: load(f"xen_ind57/{e}") for e in ENC}
ROB = {e: load(f"xen57_robust/{e}") for e in ENC} if os.path.isdir(f"{R}/xen57_robust") else None


def summ_src(src, fn):
    S = {e: per_spec({n: fn(x) for n, x in src[e].items()}) for e in ENC}
    enc = [float(np.median(list(S[e].values()))) for e in ENC]
    ps = {s: float(np.median([S[e][s] for e in ENC])) for s in SPEC}
    return float(np.median(enc)), ps


def summ_one(fn):
    ps = per_spec({n: fn(x) for n, x in XC.items()}); return float(np.median(list(ps.values()))), ps


rowsC = [("Ridge on frozen features", summ(lambda e, x: x["pred"]["ridge"]["pcc"])[:2]),
         ("Image partition, measured means ($K=20$, domain oracle)", summ(lambda e, x: x["pred"]["dom20"]["pcc"])[:2]),
         ("Coordinate partition, measured means ($K=20$)", summ_one(lambda x: x["K20"]["oracle_coord"]["pcc"])),
         ("Random matched-size partition, measured means ($K=20$)", summ_src(CT, lambda x: x["oracle_random_matched"])),
         ("Image partition, means from training blocks ($K=20$)", summ_src(CT, lambda x: x["trainonly_image"])),
         ("Image partition learned on training blocks, means from training blocks ($K=20$)", summ_src(IN, lambda x: x["inductive"])),
         ("Coordinate partition, means from training blocks ($K=20$)", summ_one(lambda x: x["K20"]["trainonly_coord"]["pcc"])),
         None,
         ("Image partition, measured means ($K=200$)", summ(lambda e, x: x["pred"]["dom200"]["pcc"])[:2]),
         ("Coordinate partition, measured means ($K=200$)", summ_one(lambda x: x["K200"]["oracle_coord"]["pcc"]))]
if ROB: rowsC.append(("Image partition, means from training blocks ($K=200$)", summ_src(ROB, lambda x: x["ctrl_K200"]["trainonly_image"])))
mod = r3(rowsC[0][1][0])
L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Predictor on Xenium ($22$ regions, $13$ specimens) & PCC & Range over specimens & Of model \\", r"\midrule"]
OUT["controls_xen"] = {}
for r in rowsC:
    if r is None: L.append(r"\midrule"); continue
    lab, (m, ps) = r; OUT["controls_xen"][lab] = dict(pcc=m, lo=min(ps.values()), hi=max(ps.values()), frac=r3(m) / mod)
    L.append(f"{lab} & {m:.3f} & {min(ps.values()):.3f}--{max(ps.values()):.3f} & {100 * r3(m) / mod:.0f}\\% \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{PAPER}/tab_controls_xen.tex", "w").write("\n".join(L) + "\n")
json.dump(OUT, open(f"{R}/supp57_local.json", "w"), indent=1)
print("\n".join(L))

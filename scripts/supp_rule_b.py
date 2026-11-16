# -*- coding: utf-8 -*-
"""附录 Xenium 各表按统一汇总规则 重算。统一汇总规则：每个分数逐区域得到，标本内取中位数，再跨 13 个标本取中位数；
差距与比值都由这些中位数算，输入印在旁边时用印出的四舍五入值；计数比较每个标本的两个中位数；不用均值和标准差。
重写 <论文目录>/tab_specimens、tab_ablate、tab_operator、tab_reliab、tab_grid、tab_multi、tab_controls_xen 的分数比，
并把附录正文要用的数写到 results/supp_rule_b.json。用法：在项目根目录运行 python3 scripts/supp_rule_b.py <论文目录>。"""
import json, glob, os, re, sys, numpy as np
from math import comb

R = os.environ.get("OCTAVE_RESULTS", "results"); PAPER = sys.argv[1] if len(sys.argv) > 1 else "paper"
SPEC = ["Human_Breast_Biomarkers_S1", "Human_Breast_Biomarkers_S2", "Human_Breast_Biomarkers_S3", "Human_Breast_Biomarkers_S4",
        "Xenium_Prime_Cervical", "Xenium_Prime_Ovarian", "Xenium_V1_Human_Kidney", "Xenium_V1_Human_Ovary",
        "Lung", "Xenium_Prime_Breast_Cancer", "Xenium_Prime_Human_Prostate", "Xenium_Prime_Human_Skin", "Xenium_Prime_Human_Lymph_Node"]
SHORT = {"Human_Breast_Biomarkers_S1": "Breast S1", "Human_Breast_Biomarkers_S2": "Breast S2", "Human_Breast_Biomarkers_S3": "Breast S3",
         "Human_Breast_Biomarkers_S4": "Breast S4", "Xenium_Prime_Cervical": "Cervical (Prime)", "Xenium_Prime_Ovarian": "Ovarian (Prime)",
         "Xenium_V1_Human_Kidney": "Kidney (V1)", "Xenium_V1_Human_Ovary": "Ovary (V1)", "Lung": "Lung", "Xenium_Prime_Breast_Cancer": "Breast (Prime)",
         "Xenium_Prime_Human_Prostate": "Prostate (Prime)", "Xenium_Prime_Human_Skin": "Skin (Prime)", "Xenium_Prime_Human_Lymph_Node": "Lymph node (Prime)"}


def sp(n):
    if "Human_Lung_Cancer_FFPE" in n: return "Lung"   # Xenium v1 与 Prime 5K 为同一供体同一组织块，按一个标本计
    return next(s for s in SPEC if n.startswith(s))


def load(d): return {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(f"{R}/{d}/*.json")}


def spec(vals):
    """区域 → 标本中位数 → 跨标本中位数；返回 (跨标本中位数, {标本: 中位数})。"""
    per = {}
    for n, v in vals.items(): per.setdefault(sp(n), []).append(v)
    m = {s: float(np.median(v)) for s, v in per.items()}
    assert len(m) == 13, len(m)
    return float(np.median(list(m.values()))), m


def signp(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def wr(name, L): open(f"{PAPER}/{name}", "w").write("\n".join(L) + "\n")


r3 = lambda x: round(x, 3)
pct1 = lambda o, m: round(100 * (1 - o / m), 1)      # 差距，印一位小数
OUT = {}

# ═══ 逐标本差距（tab_specimens），Hibou-L 默认设置
B = load("blocks_xen_bands_base")
nreg = {}
for n in B: nreg[sp(n)] = nreg.get(sp(n), 0) + 1
f = lambda p, k: {n: d["pred"][p][k] for n, d in B.items()}
PMm, PM = spec(f("ridge", "pcc")); POm, PO = spec(f("dom20", "pcc")); BMm, BM = spec(f("ridge", "band_pcc")); BOm, BO = spec(f("dom20", "band_pcc"))
rows = []
for s in SPEC:
    a, b = r3(PM[s]), r3(PO[s]); dp = pct1(b, a); df = pct1(BO[s], BM[s])
    rows.append((SHORT[s], nreg[s], a, b, dp, df, df / dp))
rows.sort(key=lambda z: -z[6])
a, b = r3(PMm), r3(POm); DP = pct1(b, a); DF = pct1(BOm, BMm); RT = DF / DP
agree = sum(z[5] > z[4] for z in rows)
band_gap = {t: pct1(spec({n: d["pred"]["dom20"]["bands"][t] for n, d in B.items()})[0], spec({n: d["pred"]["ridge"]["bands"][t] for n, d in B.items()})[0])
            for t in B[next(iter(B))]["pred"]["ridge"]["bands"]}
width = {t: spec({n: d["sigma_um"][t] for n, d in B.items()})[0] for t in B[next(iter(B))]["sigma_um"]}
# 16 个 K-means 种子：每个种子完整走一遍统一汇总规则
S = load("blocks_xen_seeds")
NS = min(len(d["seeds"]) for d in S.values())
seed = []
for k in range(NS):
    pm_ = r3(spec({n: d["ridge_pcc"] for n, d in S.items()})[0]); po_ = r3(spec({n: d["seeds"][k]["dom_pcc"] for n, d in S.items()})[0])
    dp_ = pct1(po_, pm_); df_ = pct1(spec({n: d["seeds"][k]["dom_band_pcc"] for n, d in S.items()})[0], spec({n: d["ridge_band_pcc"] for n, d in S.items()})[0])
    seed.append((dp_, df_, df_ / dp_))
seed = np.array(seed)
assert abs(seed[0, 0] - DP) < 0.11 and abs(seed[0, 1] - DF) < 0.11, (seed[0], DP, DF)   # 种子 0 即冻结配置
mr = lambda c: f"{np.median(seed[:, c]):.{2 if c == 2 else 1}f} [{seed[:, c].min():.{2 if c == 2 else 1}f}, {seed[:, c].max():.{2 if c == 2 else 1}f}]"
L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
     r"Specimen & Regions & $\mathrm{PCC}_{\mathrm{mod}}$ & $\mathrm{PCC}_{\mathrm{dom}}$ & $\Delta_{\mathrm{PCC}}$ & $\Delta_{\mathrm{fine}}$ & $\Delta_{\mathrm{fine}}/\Delta_{\mathrm{PCC}}$ \\", r"\midrule"]
for n_, k_, a_, b_, dp_, df_, rt_ in rows: L.append(f"{n_} & {k_} & {a_:.3f} & {b_:.3f} & {dp_:.1f}\\% & {df_:.1f}\\% & {rt_:.2f} \\\\")
L += [r"\midrule", f"Median over $13$ specimens & $22$ & {a:.3f} & {b:.3f} & {DP:.1f}\\% & {DF:.1f}\\% & \\textbf{{{RT:.2f}}} \\\\",
      f"${NS}$ $K$-means seeds & & & & {mr(0)}\\% & {mr(1)}\\% & {mr(2)} \\\\", r"\bottomrule", r"\end{tabular}"]
wr("tab_specimens.tex", L)
OUT["spec"] = dict(pcc_mod=a, pcc_dom=b, d_pcc=DP, d_fine=DF, ratio=RT, agree=agree, P=signp(agree, 13), ratio_min=min(z[6] for z in rows), ratio_max=max(z[6] for z in rows),
                   ratio_min_spec=min(rows, key=lambda z: z[6])[0], ratio_max_spec=max(rows, key=lambda z: z[6])[0], band_gap=band_gap, width=width,
                   seeds=dict(n=NS, d_pcc=[float(np.median(seed[:, 0])), float(seed[:, 0].min()), float(seed[:, 0].max())],
                              d_fine=[float(np.median(seed[:, 1])), float(seed[:, 1].min()), float(seed[:, 1].max())],
                              ratio=[float(np.median(seed[:, 2])), float(seed[:, 2].min()), float(seed[:, 2].max())]),
                   b1_mod=BMm, b1_dom=BOm, min_model_pcc=min(d["pred"]["ridge"]["pcc"] for d in B.values()), min_model_b1=min(d["pred"]["ridge"]["band_pcc"] for d in B.values()))

# ═══ 一次改一个设置（tab_ablate）：每个区域的 oracle 分数取 3 个种子的中位数
A = load("blocks_xen_ablate")
LAB = {"alpha": ("Ridge $\\lambda$", lambda k: "$10^{%d}$" % round(np.log10(float(k)))), "ngene": ("Genes", lambda k: k), "K": ("Clusters $K$", lambda k: k)}
DEF = {"alpha": "10000.0", "ngene": "200", "K": "20"}
L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
     r"Setting & Value & Oracle/model & $\Delta_{\mathrm{PCC}}$ & $\Delta_{\mathrm{fine}}$ & $\Delta_{\mathrm{fine}}/\Delta_{\mathrm{PCC}}$ & Specimens & $P$ \\", r"\midrule"]
OUT["ablate"] = {}
for ax in ("alpha", "ngene", "K"):
    keys = sorted(next(iter(A.values()))[ax], key=float)
    for i, k in enumerate(keys):
        g = lambda key: {n: float(np.median([r[key] for r in d[ax][k]])) for n, d in A.items() if d[ax][k]}
        pm_m, pm_s = spec(g("ridge_pcc")); po_m, po_s = spec(g("oracle_pcc")); bm_m, bm_s = spec(g("ridge_band")); bo_m, bo_s = spec(g("oracle_band"))
        share = round(100 * po_m / pm_m, 1); dp = round(100 - share, 1); df = pct1(bo_m, bm_m); rt = df / dp
        w = sum((1 - bo_s[s] / bm_s[s]) > (1 - po_s[s] / pm_s[s]) for s in SPEC)
        OUT["ablate"][f"{ax}={k}"] = dict(share=share, d_pcc=dp, d_fine=df, ratio=rt, agree=w, P=signp(w, 13))
        cells = [f"{share:.1f}\\%", f"{dp:.1f}\\%", f"{df:.1f}\\%", f"{rt:.2f}", f"{w}/13", f"{signp(w, 13):.4f}"]
        lv = LAB[ax][1](k)
        if k == DEF[ax]:
            lv = lv.replace("$10^", "$\\mathbf{10^").replace("}$", "}}$") if ax == "alpha" else f"\\textbf{{{lv}}}"
            cells = [f"\\textbf{{{c}}}" for c in cells]
        L.append((LAB[ax][0] if i == 0 else "") + f" & {lv} & " + " & ".join(cells) + r" \\")
    if ax != "K": L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
wr("tab_ablate.tex", L)

# ═══ 图、游走与基因口径（tab_operator）
L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
     r"Setting & Value & $\sigma_1$ ($\mu$m) & $\beta_1$ model & $\beta_1$ oracle & $\Delta_{\mathrm{fine}}$ & $\Delta_{\mathrm{fine}}/\Delta_{\mathrm{PCC}}$ & Specimens \\", r"\midrule"]
OUT["operator"] = {}
def oprow(D, key):
    s1 = spec({n: d["sigma_um"]["1"] for n, d in D.items()})[0]
    pm_m, pm_s = spec({n: d["pred"]["ridge"]["pcc"] for n, d in D.items()}); po_m, po_s = spec({n: d["pred"]["dom20"]["pcc"] for n, d in D.items()})
    bm_m, bm_s = spec({n: d["pred"]["ridge"][key] for n, d in D.items()}); bo_m, bo_s = spec({n: d["pred"]["dom20"][key] for n, d in D.items()})
    dp = 100 * (1 - po_m / pm_m); df = pct1(r3(bo_m), r3(bm_m))
    w = sum((1 - bo_s[s] / bm_s[s]) > (1 - po_s[s] / pm_s[s]) for s in SPEC)
    return dict(sigma1=s1, b1m=bm_m, b1o=bo_m, d_fine=df, ratio=df / dp, agree=w)
ops = [("base", "Graph and walk", r"$k=8$, $29\,\mu$m, lazy $1/2$ (default)", "band_pcc"), ("k4", "", r"$k=4$, $17\,\mu$m", "band_pcc"),
       ("k12", "", r"$k=12$, $40\,\mu$m", "band_pcc"), ("lazy75", "", r"lazy $3/4$", "band_pcc"),
       ("base", "Genes in $\\beta_1$", r"drop lowest quartile of finest-band share", "band_pcc_q25"), ("base", "", r"weight genes by finest-band variance", "band_pcc_wvar"),
       ("base", "", r"keep genes with share $\geq 5\%$", "band_pcc_thr5")]
for tag, ax, lab, key in ops:
    r = oprow(load(f"blocks_xen_bands_{tag}"), key); OUT["operator"][f"{tag}:{key}"] = r
    if ax == "Genes in $\\beta_1$": L.append(r"\midrule")
    cells = [f"{r['sigma1']:.1f}", f"{r['b1m']:.3f}", f"{r['b1o']:.3f}", f"{r['d_fine']:.1f}\\%", f"{r['ratio']:.2f}", f"{r['agree']}/13"]
    if "default" in lab: cells = [f"\\textbf{{{c}}}" for c in cells]
    L.append(f"{ax} & {lab} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
wr("tab_operator.tex", L)

# ═══ 逐带可靠性（tab_reliab）：区域 → 标本中位数 → 跨标本中位数，范围为跨标本
C = load("xen_band_ceiling")
fmt = lambda m, s: f"{m:.2f} [{min(s.values()):.2f}, {max(s.values()):.2f}]"
L = [r"\begin{tabular}{rrrrr}", r"\toprule", r"$t$ & $\sigma$ ($\mu$m) & $c_{1/2}$ & $c$ & $\sqrt{c}$ \\", r"\midrule"]
cps = [str(c) for c in next(iter(C.values()))["cps"]]
OUT["reliab"] = {}
for cp in cps[:6] + [cps[-1]]:
    ch = spec({n: C[n]["c_half"][cp] for n in C}); cf = spec({n: C[n]["c_full"][cp] for n in C}); sq = spec({n: float(np.sqrt(C[n]["c_full"][cp])) for n in C})
    sg = spec({n: B[n]["sigma_um"][cp] for n in C})[0]
    OUT["reliab"][cp] = dict(c_full=cf[0], c_full_min=min(cf[1].values()), c_full_max=max(cf[1].values()), low=sorted(cf[1], key=cf[1].get)[:4])
    L.append(f"{cp} & {sg:.1f} & {fmt(*ch)} & {fmt(*cf)} & {fmt(*sq)} \\\\")
    if cp == cps[5]: L.append(r"$\vdots$ & & & & \\")
ch = spec({n: C[n]["scalar_c_half"] for n in C}); cf = spec({n: C[n]["scalar_c_full"] for n in C}); sq = spec({n: float(np.sqrt(C[n]["scalar_c_full"])) for n in C})
OUT["reliab"]["scalar"] = dict(c_full=cf[0], sqrt=sq[0])
L += [r"\midrule", f"Unfiltered & n/a & {fmt(*ch)} & {fmt(*cf)} & {fmt(*sq)} \\\\", r"\bottomrule", r"\end{tabular}"]
wr("tab_reliab.tex", L)
c1 = {n: C[n]["c_full"]["1"] for n in C}
OUT["reliab_text"] = dict(
    b1_model_corr=spec({n: B[n]["pred"]["ridge"]["band_pcc"] / np.sqrt(c1[n]) for n in C})[0],
    b1_oracle_corr=spec({n: B[n]["pred"]["dom20"]["band_pcc"] / np.sqrt(c1[n]) for n in C})[0])
share = {n: B[n]["pred"]["truth"]["fine_var_share"] for n in C}
sig = {n: share[n] * c1[n] / C[n]["scalar_c_full"] for n in C}
sh_m, sh_s = spec(share); sg_m, sg_s = spec(sig)
rm_m, rm_s = spec({n: B[n]["pred"]["ridge"]["fine_var_share"] for n in C}); ro_m, ro_s = spec({n: B[n]["pred"]["dom20"]["fine_var_share"] for n in C})
OUT["reliab_text"].update(share_raw=sh_m, share_sig=sg_m, share_sig_min=min(sg_s.values()), share_sig_max=max(sg_s.values()),
                          fac_model_sig=sg_m / rm_m, fac_oracle_sig=sg_m / ro_m, fac_model_raw=sh_m / rm_m, fac_oracle_raw=sh_m / ro_m,
                          fac_sig_min=min(min(sg_s[s] / rm_s[s], sg_s[s] / ro_s[s]) for s in SPEC), fac_sig_max=max(max(sg_s[s] / rm_s[s], sg_s[s] / ro_s[s]) for s in SPEC),
                          fac_raw_min=min(min(sh_s[s] / rm_s[s], sh_s[s] / ro_s[s]) for s in SPEC), fac_raw_max=max(max(sh_s[s] / rm_s[s], sh_s[s] / ro_s[s]) for s in SPEC),
                          share_model=rm_m, share_oracle=ro_m)

# ═══ 评分网格（tab_grid）：逐区域 PCC 曲线 → 标本中位数 → 跨标本中位数，变化由中位数算
G = json.load(open(f"{R}/grid_reversal.json"))["grid_reversal"]; bins = G["bins_um"]; i8, i64 = bins.index(8), bins.index(64)
per = {}
for r in G["by_region"]: per.setdefault(r["specimen"], []).append(r)
def gchg(key):
    sm = {s: (float(np.median([r[key][i8] for r in v])), float(np.median([r[key][i64] for r in v]))) for s, v in per.items()}
    m8 = float(np.median([x[0] for x in sm.values()])); m64 = float(np.median([x[1] for x in sm.values()]))
    ch = {s: 100 * (x[1] / x[0] - 1) for s, x in sm.items()}
    reg = [100 * (r[key][i64] / r[key][i8] - 1) for v in per.values() for r in v]
    return dict(p8=m8, p64=m64, change=100 * (r3(m64) / r3(m8) - 1), lo=min(ch.values()), hi=max(ch.values()), reg_pos=sum(x > 0 for x in reg), spec_pos=sum(x > 0 for x in ch.values()))
GF, GX = gchg("follow_curve"), gchg("fixed_curve")
nr, ns = len(G["by_region"]), len(per)
L = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Evaluation grid & PCC, $8\,\mu$m & PCC, $64\,\mu$m & Change & Range over specimens & Regions $>0$ & Specimens $>0$ \\", r"\midrule",
     f"Prediction's own grid & {GF['p8']:.3f} & {GF['p64']:.3f} & \\textbf{{{GF['change']:+.1f}\\%}} & {GF['lo']:+.1f} to {GF['hi']:+.1f}\\% & {GF['reg_pos']}/{nr} & {GF['spec_pos']}/{ns} \\\\",
     f"Fixed at $16\\,\\mu$m & {GX['p8']:.3f} & {GX['p64']:.3f} & \\textbf{{{GX['change']:+.1f}\\%}} & {GX['lo']:+.1f} to {GX['hi']:+.1f}\\% & {GX['reg_pos']}/{nr} & {GX['spec_pos']}/{ns} \\\\",
     r"\bottomrule", r"\end{tabular}"]
wr("tab_grid.tex", L)
OUT["grid"] = dict(follow=GF, fixed=GX, n_regions=nr, n_specimens=ns, P=signp(ns, ns))

# ═══ 57 个编码器（tab_multi）：每个编码器的分数先按统一汇总规则 汇总，比值由各编码器的中位数算
E = {"hibou_l": "blocks_xen_bands_base"}
for d in sorted(glob.glob(f"{R}/blocks_xen_bands_*")):
    e = d.split("blocks_xen_bands_")[1]
    if e not in ("base", "k4", "k12", "lazy75") and len(glob.glob(d + "/*.json")) == 22: E[e] = os.path.basename(d)
assert len(E) == 57, len(E)
PAR = json.load(open(f"{R}/encoder_params.json"))
M = {}
for e, d in E.items():
    D = load(d)
    pm_, po_ = spec({n: x["pred"]["ridge"]["pcc"] for n, x in D.items()})[0], spec({n: x["pred"]["dom20"]["pcc"] for n, x in D.items()})[0]
    bm_, bo_ = spec({n: x["pred"]["ridge"]["bands"]["1"] for n, x in D.items()})[0], spec({n: x["pred"]["dom20"]["bands"]["1"] for n, x in D.items()})[0]
    b6 = spec({n: x["pred"]["ridge"]["bands"]["32"] for n, x in D.items()})[0]
    M[e] = dict(pcc=pm_, oracle=po_, b1=bm_, b1o=bo_, b6=b6, frac=po_ / pm_, ratio=(1 - bo_ / bm_) / (1 - po_ / pm_))
OUT["multi"] = M
src = open(f"{PAPER}/tab_multi.tex").read()
OUT["multi_summary"] = dict(pcc_min=min(v["pcc"] for v in M.values()), pcc_max=max(v["pcc"] for v in M.values()), b1_min=min(v["b1"] for v in M.values()), b1_max=max(v["b1"] for v in M.values()),
                            frac_min=min(v["frac"] for v in M.values()), frac_max=max(v["frac"] for v in M.values()), ratio_min=min(v["ratio"] for v in M.values()), ratio_max=max(v["ratio"] for v in M.values()),
                            med_pcc=float(np.median([v["pcc"] for v in M.values()])), med_oracle=float(np.median([v["oracle"] for v in M.values()])),
                            med_b1=float(np.median([v["b1"] for v in M.values()])), med_b1o=float(np.median([v["b1o"] for v in M.values()])), med_b6=float(np.median([v["b6"] for v in M.values()])))

# ═══ tab_multi：保留行序与名称，按统一汇总规则 重写 Oracle/model 与比值两列，并核对 PCC、β1、β6
import ast
_t = ast.parse(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mk_tab_hest_rule.py")).read())
LABEL = next(ast.literal_eval(n.value) for n in _t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LABEL")
KEY = {re.sub(r" \\citep\{[^}]*\}", "", v): k for k, v in LABEL.items()}
ms = OUT["multi_summary"]
med_frac = ms["med_oracle"] / ms["med_pcc"]; med_ratio = (1 - ms["med_b1o"] / ms["med_b1"]) / (1 - med_frac)
def block(c):
    name = c[0].strip()
    if name.startswith("\\textit{Median"):
        assert abs(float(c[2]) - ms["med_pcc"]) < 6e-4 and abs(float(c[3]) - ms["med_b1"]) < 6e-4 and abs(float(c[4]) - ms["med_b6"]) < 6e-4, c
        return c[:5] + [f" {med_frac:.2f} ", f" {med_ratio:.2f} "]
    v = M[KEY[name]]
    for j, k in ((2, "pcc"), (3, "b1"), (4, "b6")): assert abs(float(c[j]) - v[k]) < 6e-4, (name, k, c[j], v[k])
    return c[:5] + [f" {v['frac']:.2f} ", f" {v['ratio']:.2f} "]
T = []
for line in src.split("\n"):
    c = line.rstrip().rstrip("\\").split("&")
    if len(c) == 14 and not line.startswith("Encoder"):
        line = "&".join(block(c[:7]) + block(c[7:])).rstrip() + " \\\\"
    T.append(line)
open(f"{PAPER}/tab_multi.tex", "w").write("\n".join(T))
OUT["multi_summary"].update(med_frac=med_frac, med_ratio=med_ratio)

# ═══ tab_controls_xen：Hibou-L 列的分数比改为印出的 PCC 相除；57 个编码器列取正文表 1 的 Xenium 列（同为统一汇总规则）
X = open(f"{PAPER}/tab_controls_xen.tex").read().split("\n")
main = open(f"{PAPER}/tab_controls_main.tex").read()
x57 = {}
for lab_main, lab_supp in (("Ridge on frozen features (trained model)", "Ridge on frozen features"), ("Image partition, measured means (domain oracle)", "Image partition, measured means ($K=20$, domain oracle)"),
                           ("Coordinate partition, measured means", "Coordinate partition, measured means ($K=20$)"), ("Random matched-size partition, measured means", "Random matched-size partition, measured means ($K=20$)"),
                           ("Image partition, means learned from training data", "Image partition, means from training blocks ($K=20$)"),
                           ("Image partition and means both learned from training data", "Image partition learned on training blocks, means from training blocks ($K=20$)")):
    row = next(l for l in main.split("\n") if l.startswith(lab_main)).split("&")
    x57[lab_supp] = float(row[3]) / float(next(l for l in main.split("\n") if l.startswith("Ridge on frozen")).split("&")[3])
ridge = None; T = []
for line in X:
    c = line.rstrip().rstrip("\\").split("&")
    if len(c) == 5 and re.match(r"\s*0\.\d{3}\s*$", c[1]):
        lab = c[0].strip(); p = float(c[1]); ridge = ridge or p
        c[3] = f" {p / ridge:.2f} "; c[4] = f" {x57[lab]:.2f} " if lab in x57 else " "
        line = "&".join(c).rstrip() + " \\\\"
    T.append(line)
open(f"{PAPER}/tab_controls_xen.tex", "w").write("\n".join(T))
OUT["controls_xen57"] = x57

# ═══ tab_hestdelta：HEST 上按统一汇总规则（72 个样本的中位数，再跨编码器）
HJ = json.load(open(f"{R}/hest_rule_numbers.json")); HE = HJ["encoders"]; HS = HJ["summary"]
sig1 = float(np.median([o["sig1"] for o in json.load(open(f"{R}/hest_octave_delta.json"))]))
rg = lambda k: f"{min(o[k] for o in HE):.3f}--{max(o[k] for o in HE):.3f}"
cm = lambda k: (float(np.median([o[k] for o in HE])), min(o[k] for o in HE), max(o[k] for o in HE))
oc, fc = cm("oracle_coh"), cm("fine_coh")
L = [r"\begin{tabular}{lrr}", r"\toprule", r"Quantity & Median over $57$ encoders & Range \\", r"\midrule",
     f"PCC, ridge & {np.median([o['model_pcc'] for o in HE]):.3f} & {rg('model_pcc')} \\\\", f"PCC, domain oracle & {np.median([o['oracle_pcc'] for o in HE]):.3f} & {rg('oracle_pcc')} \\\\",
     f"Finest-band $\\beta_1$, ridge & {np.median([o['b1_model'] for o in HE]):.3f} & {rg('b1_model')} \\\\", f"Finest-band $\\beta_1$, domain oracle & {np.median([o['b1_oracle'] for o in HE]):.3f} & {rg('b1_oracle')} \\\\",
     r"\midrule", f"Cohorts with oracle above ridge, PCC & {oc[0]:.0f}/10 & {oc[1]}--{oc[2]} \\\\", f"Cohorts with oracle above ridge, finest band & {fc[0]:.0f}/10 & {fc[1]}--{fc[2]} \\\\",
     f"Finest-band width $\\sigma_1$ ($\\mu$m), set by the grid & {sig1:.1f} & {sig1:.1f}--{sig1:.1f} \\\\", r"\bottomrule", r"\end{tabular}"]
wr("tab_hestdelta.tex", L)
OUT["hestdelta"] = dict(model=float(np.median([o['model_pcc'] for o in HE])), oracle=float(np.median([o['oracle_pcc'] for o in HE])), b1m=float(np.median([o['b1_model'] for o in HE])),
                        b1o=float(np.median([o['b1_oracle'] for o in HE])), oracle_coh=oc, fine_coh=fc, n_all10_pcc=sum(o["oracle_coh"] == 10 for o in HE),
                        n_majority_fine=sum(o["fine_coh"] >= 6 for o in HE), n_all10_fine=sum(o["fine_coh"] == 10 for o in HE), n_fine_ahead=sum(o["b1_oracle"] > o["b1_model"] for o in HE), sig1=sig1)


# ═══ 附录正文里其余按统一汇总规则 的数（I.5、I.6、H.2、D 节与引言的 MSSI 一句）
AT = load("xen_attrib"); g = lambda fn: spec({n: fn(d) for n, d in AT.items()})
pm_, pms_ = g(lambda d: d["model_decomp"]["pcc_model"]); po_, pos_ = g(lambda d: d["model_decomp"]["pcc_oracle"])
pb_, pbs_ = g(lambda d: d["model_decomp"]["pcc_model_between_only"])
dp_all = 1 - po_ / pm_; dps_ = {s: 1 - pos_[s] / pms_[s] for s in pms_}
T = dict(between_over_model=r3(pb_) / r3(pm_), between_over_model_range=[min(pbs_[s] / pms_[s] for s in SPEC), max(pbs_[s] / pms_[s] for s in SPEC)])
AS = load("xen_addsplit"); bt_, bts_ = spec({n: d["between_term"] for n, d in AS.items()}); wt_ = spec({n: d["within_term"] for n, d in AS.items()})[0]
pp_, pps_ = spec({n: d["pcc"] for n, d in AS.items()})
T.update(between_term=bt_, within_term=wt_, addsplit_pcc=pp_, between_share=r3(bt_) / r3(pp_), within_share=r3(wt_) / r3(pp_),
         between_share_range=[min(bts_[s] / pps_[s] for s in SPEC), max(bts_[s] / pps_[s] for s in SPEC)])
for op in ("lazy_rw_default", "symmetric_normalised", "gaussian_16um"):
    bm_, bms_ = g(lambda d: d["operators"][op]["beta1_model"]); bo_, bos_ = g(lambda d: d["operators"][op]["beta1_oracle"])
    per_ = {s: (1 - bos_[s] / bms_[s]) / dps_[s] for s in SPEC}
    T[op] = dict(ratio=(1 - bo_ / bm_) / dp_all, lo=min(per_.values()), hi=max(per_.values()), above1=sum(v > 1 for v in per_.values()))
cf = lambda k: g(lambda d: d["crossfit"][k])
pr_, prs_ = cf("pcc_ridge"); br_, brs_ = cf("beta1_ridge")
for k in ("cf", "same"):
    P_, Ps_ = cf(f"pcc_oracle_{k}"); B_, Bs_ = cf(f"beta1_oracle_{k}")
    per_ = {s: (1 - Bs_[s] / brs_[s]) / (1 - Ps_[s] / prs_[s]) for s in SPEC}
    T[f"crossfit_{k}"] = dict(b1=B_, ratio=(1 - B_ / br_) / (1 - P_ / pr_), lo=min(per_.values()), hi=max(per_.values()), above1=sum(v > 1 for v in per_.values()))
tp_, tps_ = g(lambda d: d["trainonly"]["pcc"]); tb_, tbs_ = g(lambda d: d["trainonly"]["beta1"]); bm0, bm0s = g(lambda d: d["model_decomp"]["beta1_model"])
per_ = {s: (1 - tbs_[s] / bm0s[s]) / (1 - tps_[s] / pms_[s]) for s in SPEC}
T["trainonly"] = dict(b1=tb_, d_fine=1 - tb_ / bm0, ratio=(1 - tb_ / bm0) / (1 - tp_ / pm_), lo=min(per_.values()), hi=max(per_.values()), above1=sum(v > 1 for v in per_.values()))
# 57 个编码器的 Xenium 划分对照：可学习预测器与 oracle 的最大相对差（逐编码器逐标本）、归纳式比例范围、重算偏差
within = 0; dev_e = 0; dev_r = 0; ind_frac = []
for e, d in E.items():
    C = load(f"xen_ctrl57/{e}"); I = load(f"xen_ind57/{e}"); D = load(d)
    o_, os2 = spec({n: x["oracle_image"] for n, x in C.items()}); t_, ts2 = spec({n: x["trainonly_image"] for n, x in C.items()})
    within = max(within, max(abs(1 - ts2[s] / os2[s]) for s in SPEC))
    dev_e = max(dev_e, abs(spec({n: x["pred"]["dom20"]["pcc"] for n, x in D.items()})[0] - o_))
    dev_r = max(dev_r, max(abs(D[n]["pred"]["dom20"]["pcc"] - C[n]["oracle_image"]) for n in C))
    ind_frac.append(spec({n: x["inductive"] for n, x in I.items()})[0] / M[e]["pcc"])
T["ctrl57"] = dict(train_vs_oracle_max=within, oracle_dev_encoder_max=dev_e, oracle_dev_region_max=dev_r, ind_frac_range=[min(ind_frac), max(ind_frac)])
# HEST：重算对照时 oracle 的变化（每个编码器 72 个样本中位数之差；单个样本的最大差）
devs, devmax = [], 0
for o in HE:
    Cc = json.load(open(f"{R}/hest_controls_{o['enc']}.json"))["samples"]; Bk = json.load(open(f"{R}/hest_blocks_{o['enc']}.json"))["samples"]; sids = sorted(Cc)
    a_ = np.array([Bk[s]["blk_k20"] for s in sids]); c_ = np.array([Cc[s]["oracle_image"] for s in sids])
    devs.append(abs(np.median(a_) - np.median(c_))); devmax = max(devmax, float(np.max(np.abs(a_ - c_))))
T["hest_ctrl_dev"] = dict(median=float(np.median(devs)), max=max(devs), sample_max=devmax)
# D 节：前 16 个区域上 oracle PCC 在 16 个种子间的极差
first16 = sorted(json.load(open(f))["name"] for f in glob.glob(f"{R}/istar_xen/*.json"))
T["seed_range_median"] = float(np.median([max(x["dom_pcc"] for x in S[n]["seeds"]) - min(x["dom_pcc"] for x in S[n]["seeds"]) for n in first16]))
# 引言：MSSI 下 oracle 落后模型的差距（统一汇总规则），与 PCC 下的差距比较
MS = load("xen_mssi")
mm_, mms_ = spec({n: x["pred"]["ridge"]["mssi"] for n, x in MS.items()}); mo_, mos_ = spec({n: x["pred"]["dom20"]["mssi"] for n, x in MS.items()})
T["mssi"] = dict(model=mm_, oracle=mo_, gap=1 - mo_ / mm_, pcc_gap=1 - POm / PMm, specimens_larger=sum((1 - mos_[s] / mms_[s]) > (1 - PO[s] / PM[s]) for s in SPEC))
T["sigma_L"] = dict(median=width["2048"], region_min=min(d["sigma_um"]["2048"] for d in B.values()), region_max=max(d["sigma_um"]["2048"] for d in B.values()))
OUT["text"] = T

json.dump(OUT, open(f"{R}/supp_rule_b.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in OUT.items() if k not in ("multi", "ablate", "operator", "reliab")}, indent=1, default=lambda x: round(float(x), 4))[:6000])
print("ablate", {k: (v["share"], v["d_pcc"], v["d_fine"], round(v["ratio"], 2), v["agree"]) for k, v in OUT["ablate"].items()})
print("operator", {k: (round(v["sigma1"], 1), round(v["b1m"], 3), round(v["b1o"], 3), v["d_fine"], round(v["ratio"], 2), v["agree"]) for k, v in OUT["operator"].items()})

# -*- coding: utf-8 -*-
"""附录 F 的算子/基因口径敏感性表（tab_operator.tex）与附录 I.4 的逐带可靠性表（tab_reliab.tex）。
输入：results/blocks_xen_bands_{base,k4,k12,lazy75}/*.json（src/blocks_xen_bands.py，--knn/--cut_um/--lazy）、
      results/xen_band_ceiling/*.json（xen_band_ceiling.py，二项拆半，3 次重复）。聚合与正文一致：区域→标本中位数→跨标本中位数。"""
import json, glob, os, numpy as np
os.environ.setdefault("S4ST_RESULTS", "results"); R = os.environ["S4ST_RESULTS"]
SPEC = ["Human_Breast_Biomarkers_S1", "Human_Breast_Biomarkers_S2", "Human_Breast_Biomarkers_S3", "Human_Breast_Biomarkers_S4",
        "Xenium_Prime_Cervical", "Xenium_Prime_Ovarian", "Xenium_V1_Human_Kidney", "Xenium_V1_Human_Ovary"]
sp = lambda n: next(s for s in SPEC if n.startswith(s))
def load(d): return {json.load(open(f))["name"]: json.load(open(f)) for f in glob.glob(f"{R}/{d}/*.json")}
def specmed(vals):
    per = {}
    for n, v in vals.items(): per.setdefault(sp(n), []).append(v)
    m = {s: float(np.median(v)) for s, v in per.items()}; return float(np.median(list(m.values()))), m
def signP(k, n=8):
    from math import comb
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, n - k + 1)) / 2 ** n) if k > n / 2 else 1.0
def row(D, key="band_pcc"):
    dfine = {n: (x["pred"]["ridge"][key] - x["pred"]["dom20"][key]) / x["pred"]["ridge"][key] for n, x in D.items()}
    ratio = {n: dfine[n] / x["rel_gap"]["dom20"]["overall"] for n, x in D.items()}
    rm, per = specmed(ratio); k = sum(v > 1 for v in per.values())
    return dict(sigma1=float(np.median([x["sigma_um"]["1"] for x in D.values()])), b1m=specmed({n: x["pred"]["ridge"][key] for n, x in D.items()})[0],
                b1o=specmed({n: x["pred"]["dom20"][key] for n, x in D.items()})[0], dfine=100 * specmed(dfine)[0], ratio=rm, k=k, P=signP(k), rmin=min(per.values()))
rows, num = [], {}
for tag, lab in [("base", r"$k=8$, $29\,\mu$m, lazy $1/2$ (default)"), ("k4", r"$k=4$, $17\,\mu$m"), ("k12", r"$k=12$, $40\,\mu$m"), ("lazy75", r"lazy $3/4$")]:
    r = row(load(f"blocks_xen_bands_{tag}")); num[f"op_{tag}"] = r
    rows.append(("Graph and walk" if tag == "base" else "", lab, r))
D = load("blocks_xen_bands_base")
for key, lab in [("band_pcc_q25", r"drop lowest quartile of finest-band share"), ("band_pcc_wvar", r"weight genes by finest-band variance"), ("band_pcc_thr5", r"keep genes with share $\geq 5\%$ (all qualify)")]:
    r = row(D, key); num[f"gene_{key}"] = r; rows.append(("Genes in $\\beta_1$" if key == "band_pcc_q25" else "", lab, r))
L = [r"\begin{tabular}{llrrrrrr}", r"\toprule", r"Axis & Setting & $\sigma_1$ ($\mu$m) & $\beta_1$ model & $\beta_1$ oracle & $\Delta_{\mathrm{fine}}$ (\%) & Ratio & Specimens \\", r"\midrule"]
for i, (ax, lab, r) in enumerate(rows):
    if ax == "Genes in $\\beta_1$": L.append(r"\midrule")
    cells = [f"{r['sigma1']:.1f}", f"{r['b1m']:.3f}", f"{r['b1o']:.3f}", f"{r['dfine']:.1f}", f"{r['ratio']:.2f}", f"{r['k']}/8"]
    if "default" in lab: cells = [r"\textbf{%s}" % c for c in cells]
    L.append(f"{ax} & {lab} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]; open("paper/tab_operator.tex", "w").write("\n".join(L) + "\n")
# ── 可靠性表 ──
C = load("xen_band_ceiling"); cps = [str(c) for c in next(iter(C.values()))["cps"]]
L = [r"\begin{tabular}{rrrrr}", r"\toprule", r"$t$ & $\sigma$ ($\mu$m) & $c_{1/2}$ & $c$ & $\sqrt{c}$ \\", r"\midrule"]
def fmt(v): return f"{np.median(v):.2f} [{min(v):.2f}, {max(v):.2f}]"
rel = {}
for cp in cps[:6] + [cps[-1]]:
    ch = [C[n]["c_half"][cp] for n in C]; cf = [C[n]["c_full"][cp] for n in C]; sg = [D[n]["sigma_um"][cp] for n in C]
    rel[cp] = dict(sigma=float(np.median(sg)), c_half=float(np.median(ch)), c_full=float(np.median(cf)), c_full_min=min(cf), c_full_max=max(cf))
    L.append(f"{cp} & {np.median(sg):.1f} & {fmt(ch)} & {fmt(cf)} & {fmt(np.sqrt(cf))} \\\\")
    if cp == cps[5]: L.append(r"$\vdots$ & & & & \\")
sc = [C[n]["scalar_c_full"] for n in C]; sh = [C[n]["scalar_c_half"] for n in C]
L += [r"\midrule", f"scalar & -- & {fmt(sh)} & {fmt(sc)} & {fmt(np.sqrt(sc))} \\\\", r"\bottomrule", r"\end{tabular}"]
open("paper/tab_reliab.tex", "w").write("\n".join(L) + "\n")
# ── 「太平滑」两种口径 ──
per = {}
for n, x in D.items():
    share = x["pred"]["truth"]["fine_var_share"]; c1 = C[n]["c_full"]["1"]; ct = C[n]["scalar_c_full"]; sig = share * c1 / ct
    per.setdefault(sp(n), []).append((share / x["pred"]["ridge"]["fine_var_share"], share / x["pred"]["dom20"]["fine_var_share"], sig / x["pred"]["ridge"]["fine_var_share"], sig / x["pred"]["dom20"]["fine_var_share"], share, sig, c1))
M = np.array([np.median(np.array(v), 0) for v in per.values()])
both_raw = np.concatenate([M[:, 0], M[:, 1]]); both_sig = np.concatenate([M[:, 2], M[:, 3]])
num["smooth"] = dict(raw_med=float(np.median(both_raw)), raw_min=float(both_raw.min()), raw_max=float(both_raw.max()), sig_med=float(np.median(both_sig)), sig_min=float(both_sig.min()), sig_max=float(both_sig.max()),
                     share_med=float(np.median(M[:, 4])), sigshare_med=float(np.median(M[:, 5])), sigshare_min=float(M[:, 5].min()), sigshare_max=float(M[:, 5].max()), c1_med=float(np.median(M[:, 6])), c1_min=float(M[:, 6].min()), c1_max=float(M[:, 6].max()))
b = np.array([(x["pred"]["ridge"]["band_pcc"] / np.sqrt(C[n]["c_full"]["1"]), x["pred"]["dom20"]["band_pcc"] / np.sqrt(C[n]["c_full"]["1"])) for n, x in D.items()])
num["n_thr5_min"] = int(min(x["pred"]["ridge"]["n_genes_thr5"] for x in D.values()))
num["beta1_corrected"] = dict(model=float(np.median(b[:, 0])), oracle=float(np.median(b[:, 1])))
num["reliab"] = rel; json.dump(num, open(f"{R}/operator_reliab_numbers.json", "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else {a: (round(b_, 3) if isinstance(b_, float) else b_) for a, b_ in v.items()}) for k, v in num.items() if k != "reliab"}, indent=0, ensure_ascii=False)[:2500])

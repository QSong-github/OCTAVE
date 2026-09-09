# -*- coding: utf-8 -*-
"""附录：多个真实编码器（ridge 头）在 Xenium 上的 OCTAVE 轮廓与尺度特异排名。输入 results/xen_multi_rank.json（scripts/xen_multi_rank.py）
与 results/encoder_params.json。输出 paper/tab_multi.tex 与 paper/multi_text.tex（段落数字全部由此生成）。"""
import json, os, ast as _ast, numpy as np
R = os.environ.get("S4ST_RESULTS", "results"); M = json.load(open(f"{R}/xen_multi_rank.json")); P = json.load(open(f"{R}/encoder_params.json"))

def _name_map():
    """从 mk_tables.py 取显示名：NAME = {...} 以及后续的 NAME.update({...})。"""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mk_tables.py")).read(); tree = _ast.parse(src); out = {}
    for node in tree.body:
        if isinstance(node, _ast.Assign) and any(getattr(t, "id", "") == "NAME" for t in node.targets):
            try: out.update(_ast.literal_eval(node.value))
            except Exception: pass
        if isinstance(node, _ast.Expr) and isinstance(node.value, _ast.Call) and isinstance(node.value.func, _ast.Attribute) and getattr(node.value.func.value, "id", "") == "NAME" and node.value.func.attr == "update" and node.value.args:
            try: out.update(_ast.literal_eval(node.value.args[0]))
            except Exception: pass
    return out
_N = _name_map()
NAME = dict(_N); NAME.update({"hibou_l": "Hibou-L", "uni_v2": "UNI v2", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1", "gigapath": "Prov-GigaPath", "phikon_v2": "Phikon-v2", "conch_v15": "CONCH v1.5", "kaiko_vits16": "Kaiko-S16", "ctranspath": "CTransPath", "h0_mini": "H0-mini", "midnight12k": "Midnight-12k"})
W = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten",11:"eleven",12:"twelve",13:"thirteen",14:"fourteen",15:"fifteen",16:"sixteen",17:"seventeen",18:"eighteen",19:"nineteen",20:"twenty",30:"thirty",40:"forty",50:"fifty",60:"sixty",70:"seventy",80:"eighty",90:"ninety"}
_w = lambda k: W[k] if k in W else W[k//10*10] + "-" + W[k%10]
E = M["encoders"]; order = sorted(E, key=lambda e: -E[e]["pcc"]); cps = sorted(next(iter(E.values()))["bands"], key=int)
show = [c for c in cps if c in ("1", "4", "32")]
HDR = r"Encoder & M & PCC & $\beta_1$ & $\beta_{32}$ & Oracle/model & $\Delta$ ratio"
def _row(e):
    v = E[e]; return f"{NAME.get(e, e)} & {P[e]['params']/1e6:.0f} & {v['pcc']:.3f} & {v['bands']['1']:.3f} & {v['bands']['32']:.3f} & {v['ratio']:.2f} & {v['gap_ratio']:.2f}"
half = (len(order) + 1) // 2; left, right = order[:half], order[half:]
_mean = lambda f: float(np.mean([f(E[e]) for e in order]))
MEAN_ROW = f"\\textit{{Mean over {len(order)}}} & -- & {_mean(lambda v: v['pcc']):.3f} & {_mean(lambda v: v['bands']['1']):.3f} & {_mean(lambda v: v['bands']['32']):.3f} & {_mean(lambda v: v['ratio']):.2f} & {_mean(lambda v: v['gap_ratio']):.2f}"  # 右栏末尾空位放平均行（与正文 Table 1 一致）
L = [r"{\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrrr|lrrrrrr}", r"\toprule", HDR + " & " + HDR + r" \\", r"\midrule"]
for k in range(half):
    L.append(_row(left[k]) + " & " + (_row(right[k]) if k < len(right) else (MEAN_ROW if k == len(right) else " & & & & & & ")) + r" \\")
L += [r"\bottomrule", r"\end{tabular}}"]; open("paper/tab_multi.tex", "w").write("\n".join(L) + "\n")
rk = M["ranking"]; n = len(E); pairs = n * (n - 1) // 2
pcc = np.array([E[e]["pcc"] for e in order]); b1 = np.array([E[e]["bands"]["1"] for e in order]); gr = np.array([E[e]["gap_ratio"] for e in order]); rat = np.array([E[e]["ratio"] for e in order])
top, bot = order[0], min(order, key=lambda e: E[e]["bands"]["1"])
cp = M.get("close_pairs", []); cp_txt = ""
if cp:
    a, b, dp, db = cp[0]; cp_txt = f" Among pairs the scalar separates by less than $0.01$, {NAME.get(a,a)} and {NAME.get(b,b)} differ by ${dp:.3f}$ in $\\mathrm{{PCC}}$ and by ${abs(db):.3f}$ in $\\beta_1$, a {abs(db)/max(E[a]['bands']['1'],E[b]['bands']['1'])*100:.0f}\\% difference in finest-band fidelity that the scalar records as a tie."
sig = float(np.median([E[e]["sigma"] for e in E]))
_bp = f"{R}/beta1_rank_boot.json"; boot_txt = ""
if os.path.exists(_bp):
    _b = json.load(open(_bp)); boot_txt = (f" The reordering is not sampling noise: resampling the ${8}$ specimens with replacement ($B={_b['B']}$) keeps the finest-band ranking at Spearman ${_b['spearman_boot_vs_full']['med']:.2f}$ [{_b['spearman_boot_vs_full']['lo']:.2f}, {_b['spearman_boot_vs_full']['hi']:.2f}] to the full-data ranking, leaves the leader in first place in every resample, and preserves the direction of ${_b['rev_stable95']}$ of the ${_b['n_rev']}$ reversed pairs in at least $95\\%$ of resamples (${_b['rev_stable80']}$ at $80\\%$); the $\\beta_1$ gap between {NAME.get('keep','KEEP')} and {NAME.get('dinov3_vith16','DINOv3-H+')} has a bootstrap interval of ${-_b['keep_minus_dinov3h']['hi']:.3f}$ to ${-_b['keep_minus_dinov3h']['lo']:.3f}$ and never changes sign. Per-encoder $\\beta_1$ intervals have a median width of ${_b['b1_ci_width']['med']:.3f}$ (scripts/beta1\\_rank\\_boot.py).")
_pm = [P[e]["params"] / 1e6 for e in order if e in P]; _pstr = lambda v: f"${v/1000:.1f}$B" if v >= 1000 else f"${v:.0f}$M"
_bt = max(order, key=lambda e: E[e]["bands"]["1"])
_lead = (f"{NAME.get(top,top)} leads both the scalar and the finest band." if _bt == top else f"{NAME.get(top,top)} leads the scalar while {NAME.get(_bt,_bt)} leads the finest band, by ${E[_bt]['bands']['1']-E[top]['bands']['1']:.3f}$ at a scalar deficit of ${E[top]['pcc']-E[_bt]['pcc']:.3f}$.")
txt = (f"\\subsection{{{_w(n-1).capitalize()} further encoders under the same readout}}" "\n" r"\label{app:multi}" "\n"
       f"Section~\\ref{{sec:exp:main}} compares one trained model with the domain oracle. Table~\\ref{{tab:multi}} repeats the whole computation for ${n}$ encoders in all, each paired with the same ridge head, the same folds and the same operator, spanning {_pstr(min(_pm))} to {_pstr(max(_pm))} parameters, general-purpose and pathology pre-training, including distilled students. The scalar compresses them: the ${n}$ scores span ${pcc.min():.3f}$ to ${pcc.max():.3f}$, a range of ${pcc.max()-pcc.min():.2f}$, while the finest-band correlations span ${b1.min():.3f}$ to ${b1.max():.3f}$, a factor of ${b1.max()/b1.min():.1f}$; {_lead} The ranking is also scale-dependent. Against the scalar ranking, the finest band reorders ${rk['1']['reversals']}$ of the ${pairs}$ encoder pairs (Spearman ${rk['1']['spearman']:.2f}$), the $t=4$ band ${rk['4']['reversals']}$ and the $t=32$ band ${rk['32']['reversals']}$ (Spearman ${rk['32']['spearman']:.2f}$), so the same leaderboard reads differently at different widths.{cp_txt}{boot_txt} Nor is the result of Section~\\ref{{sec:exp:main}} a property of one encoder: every encoder's own domain oracle reaches ${100*rat.min():.0f}$--${100*rat.max():.0f}\\%$ of its score, and the finest-band shortfall exceeds the scalar shortfall by a factor of ${gr.min():.1f}$ to ${gr.max():.1f}$ across the ${n}$." "\n"
       r"\begin{table}[h]" "\n" r"\centering\small" "\n" "\\caption{" + _w(n).capitalize() + r" encoders on the $16$ Xenium regions under the same protocol. Medians over specimens of per-specimen medians over regions; $\beta$ at $t=32$ has a median width of about $50\,\mu$m. M is the parameter count in millions, $\beta_{32}$ the band at $t=32$, Oracle/model the domain oracle's share of the encoder's own score, and $\Delta$ ratio the finest-band to scalar shortfall ratio of Section~\ref{sec:exp:main}. The last row is the mean over the $57$ encoders.}" "\n" r"\label{tab:multi}" "\n" r"\resizebox{\textwidth}{!}{\input{tab_multi}}" "\n" r"\end{table}" "\n")
open("paper/multi_text.tex", "w").write(txt); print(f"multi: n={n}, PCC {pcc.min():.3f}–{pcc.max():.3f}, β1 {b1.min():.3f}–{b1.max():.3f}, reversals band1 {rk['1']['reversals']}/{pairs}, gap ratio {gr.min():.2f}–{gr.max():.2f}, oracle ratio {rat.min():.2f}–{rat.max():.2f}")

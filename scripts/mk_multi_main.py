# -*- coding: utf-8 -*-
"""把多编码器（Xenium）结果的数字写进正文：摘要句、贡献句、§3.3 段、讨论句。输入 results/xen_multi_rank.json、results/encoder_params.json；
名称映射取自 mk_tables.py。用法：python scripts/mk_multi_main.py [--dry]（--dry 只写到 scratch，不动 paper/main.tex）。"""
import json, os, re, sys, ast as _ast, numpy as np
R = os.environ.get("S4ST_RESULTS", "results"); M = json.load(open(f"{R}/xen_multi_rank.json")); P = json.load(open(f"{R}/encoder_params.json"))
E = M["encoders"]; order = sorted(E, key=lambda e: -E[e]["pcc"]); n = len(order); rk = M["ranking"]; pairs = rk["1"]["pairs"]
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
NAME = _name_map()
W = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten",11:"eleven",12:"twelve",13:"thirteen",14:"fourteen",15:"fifteen",16:"sixteen",17:"seventeen",18:"eighteen",19:"nineteen",20:"twenty",30:"thirty",40:"forty",50:"fifty",60:"sixty",70:"seventy",80:"eighty",90:"ninety"}
def w(k): return W[k] if k in W else W[k//10*10] + "-" + W[k%10]
pcc = np.array([E[e]["pcc"] for e in order]); b1 = np.array([E[e]["bands"]["1"] for e in order]); gr = np.array([E[e]["gap_ratio"] for e in order])
rng = pcc.max() - pcc.min(); fac = b1.max() / b1.min(); rev1 = rk["1"]["reversals"]; rev32 = rk["32"]["reversals"]
pm = [float(P[e]["params"]) / 1e6 for e in order if e in P]; pmin, pmax = min(pm), max(pm)
def pstr(v): return f"${v/1000:.1f}$B" if v >= 1000 else f"${v:.0f}$M"
cp = M.get("close_pairs", []); a, b, dp, db = cp[0]
short = (f"across {w(n)} encoders the scalar spans ${rng:.2f}$ while the finest band spans a factor of ${fac:.1f}$ and reorders ${rev1}$ of ${pairs}$ pairs.")
para = (f"\\paragraph{{Across {w(n)} encoders the finest band spans a factor of ${fac:.1f}$ where the scalar spans ${rng:.2f}$.}} Table~\\ref{{tab:multi}} (Appendix~\\ref{{app:multi}}) repeats the protocol for {w(n-1)} further encoders ({pstr(pmin)} to {pstr(pmax)} parameters). "
        f"Scalar scores run from ${pcc.min():.3f}$ to ${pcc.max():.3f}$ and finest-band correlations from ${b1.min():.3f}$ to ${b1.max():.3f}$. "
        f"Relative to the scalar ranking the finest band reorders ${rev1}$ of the ${pairs}$ encoder pairs; among pairs the scalar separates by less than $0.01$, {NAME.get(a,a)} and {NAME.get(b,b)} at ${dp:.3f}$ apart differ by ${abs(db):.3f}$ in $\\beta_1$. "
        f"For all {w(n)} the finest-band to scalar shortfall ratio lies between ${gr.min():.1f}$ and ${gr.max():.1f}$ (Appendix~\\ref{{app:multi}}).")
disc = f"Among real encoders, {w(n)} within ${rng:.2f}$ on the scalar differ by a factor of ${fac:.1f}$ in the finest band, reordering ${rev1}$ of ${pairs}$ pairs."
s = open("paper/main.tex").read()
pat_short = re.compile(r"[Aa]cross [a-z-]+ encoders the scalar spans \$[^$]*\$ while the finest band spans a factor of \$[^$]*\$ and reorders \$[^$]*\$ of \$[^$]*\$ pairs\.")
# 摘要里这句在 2026-09 的改写中拆成了两句，单独匹配
short_abs = (f"Across {w(n)} encoders the scalar spans ${rng:.2f}$. The finest band spans a factor of ${fac:.1f}$ and reorders ${rev1}$ of ${pairs}$ pairs.")
pat_abs = re.compile(r"Across [a-z-]+ encoders the scalar spans \$[^$]*\$\. The finest band spans a factor of \$[^$]*\$ and reorders \$[^$]*\$ of \$[^$]*\$ pairs\.")
pat_para = re.compile(r"\\paragraph\{Across [a-z-]+ encoders the finest band spans[^}]*\} Table~\\ref\{tab:multi\}.*?\(Appendix~\\ref\{app:multi\}\)\.", re.S)
pat_disc = re.compile(r"Among real encoders, [a-z-]+ within \$[^$]*\$ on the scalar differ by a factor of \$[^$]*\$ in the finest band, reordering \$[^$]*\$ of \$[^$]*\$ pairs\.")
c1, c2, c3, c0 = len(pat_short.findall(s)), len(pat_para.findall(s)), len(pat_disc.findall(s)), len(pat_abs.findall(s))
assert (c1, c2, c0) == (1, 1, 1) and c3 <= 1, (c1, c2, c3, c0)  # 讨论句已从正文删去（与 §3.3 重复），存在时才替换
s = pat_short.sub(lambda m: (short[0].upper() if m.group(0)[0] == "A" else short[0]) + short[1:], s); s = pat_abs.sub(lambda m: short_abs, s); s = pat_para.sub(lambda m: para, s); s = pat_disc.sub(lambda m: disc, s)
out = "paper/main.tex" if "--dry" not in sys.argv else os.path.join(os.environ.get("SCRATCH", "/tmp"), "main_dry.tex")
open(out, "w").write(s)
print(f"n={n} PCC {pcc.min():.3f}–{pcc.max():.3f} (range {rng:.2f}) β1 {b1.min():.3f}–{b1.max():.3f} (×{fac:.1f}) reversals band1 {rev1}/{pairs}, t=32 {rev32}; close pair {a}/{b} Δpcc {dp:.3f} Δβ1 {db:.3f}; gap ratio {gr.min():.1f}–{gr.max():.1f}; params {pmin:.0f}M–{pmax:.0f}M -> {out}")

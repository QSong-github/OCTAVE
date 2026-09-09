# -*- coding: utf-8 -*-
"""审稿意见：多个真实编码器（ridge 头）在 Xenium 上的 OCTAVE 轮廓与尺度特异排名。
输入 results/blocks_xen_bands_{enc}/*.json（hibou_l 取 blocks_xen_bands_base）。聚合：区域→标本中位数→跨标本中位数。
输出：每编码器的标量 PCC、各带 β_i、自身分区 oracle 的比值；标量排名 vs 各带排名的 Spearman 与成对名次反转数；PCC 接近但 β1 相差大的编码器对。"""
import json, glob, os, numpy as np
R = os.environ.get("S4ST_RESULTS", "results")

def _name_map():
    """从 scripts/mk_tables.py 源码里静态提取 NAME 字典（不 import，避免触发其建表副作用）。"""
    import ast as _ast, os as _os
    src = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "mk_tables.py")).read(); tree = _ast.parse(src); out = {}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assign) and any(isinstance(t, _ast.Name) and t.id == "NAME" for t in node.targets):
            try: out.update(_ast.literal_eval(node.value))
            except Exception: pass
        if isinstance(node, _ast.Expr) and isinstance(node.value, _ast.Call) and getattr(getattr(node.value.func, "value", None), "id", None) == "NAME" and node.value.func.attr == "update":
            try: out.update(_ast.literal_eval(node.value.args[0]))
            except Exception: pass
    return out
_NAME = _name_map()
_TAGS = {"base", "k4", "k12", "lazy75"}
ENC = ["hibou_l"] + sorted(d.split("blocks_xen_bands_")[1] for d in glob.glob(f"{R}/blocks_xen_bands_*") if d.split("blocks_xen_bands_")[1] not in _TAGS and len(glob.glob(d + "/*.json")) == 16)
SPEC = ["Human_Breast_Biomarkers_S1", "Human_Breast_Biomarkers_S2", "Human_Breast_Biomarkers_S3", "Human_Breast_Biomarkers_S4", "Xenium_Prime_Cervical", "Xenium_Prime_Ovarian", "Xenium_V1_Human_Kidney", "Xenium_V1_Human_Ovary"]
sp = lambda n: next(s for s in SPEC if n.startswith(s))
def specmed(d):
    per = {}
    for n, v in d.items(): per.setdefault(sp(n), []).append(v)
    return float(np.median([np.median(v) for v in per.values()])), len(per)
def rank(a):
    a = np.asarray(a, float); o = a.argsort()[::-1]; r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1); return r
res = {}
for e in dict.fromkeys(ENC):
    d = "blocks_xen_bands_base" if e == "hibou_l" else f"blocks_xen_bands_{e}"
    F = glob.glob(f"{R}/{d}/*.json")
    if len(F) < 16: print(f"  {e}: 仅 {len(F)}/16 区域，跳过"); continue
    D = {json.load(open(f))["name"]: json.load(open(f)) for f in F}
    cps = sorted(next(iter(D.values()))["pred"]["ridge"]["bands"], key=int)
    res[e] = dict(pcc=specmed({n: x["pred"]["ridge"]["pcc"] for n, x in D.items()})[0], oracle_pcc=specmed({n: x["pred"]["dom20"]["pcc"] for n, x in D.items()})[0],
                  ratio=specmed({n: x["pred"]["dom20"]["pcc"] / x["pred"]["ridge"]["pcc"] for n, x in D.items()})[0],
                  bands={cp: specmed({n: x["pred"]["ridge"]["bands"][cp] for n, x in D.items()})[0] for cp in cps},
                  gap_ratio=specmed({n: x["rel_gap"]["dom20"]["fineband"] / x["rel_gap"]["dom20"]["overall"] for n, x in D.items()})[0],
                  sigma=specmed({n: x["sigma_um"]["1"] for n, x in D.items()})[0])
E = list(res); pcc = np.array([res[e]["pcc"] for e in E]); rs = rank(pcc); out = dict(encoders=res, ranking={})
print(f"{'encoder':14s} PCC    oracle  ratio  " + " ".join(f"β{cp:>4s}" for cp in cps[:6]) + "   fine/scalar")
for e in sorted(E, key=lambda e: -res[e]["pcc"]): print(f"{e:14s} {res[e]['pcc']:.3f}  {res[e]['oracle_pcc']:.3f}  {res[e]['ratio']:.2f}  " + " ".join(f"{res[e]['bands'][cp]:.3f}" for cp in cps[:6]) + f"   {res[e]['gap_ratio']:.2f}")
for cp in cps:
    b = np.array([res[e]["bands"][cp] for e in E]); rb = rank(b)
    rev = sum(1 for i in range(len(E)) for j in range(i + 1, len(E)) if (pcc[i] - pcc[j]) * (b[i] - b[j]) < 0)
    rho = np.corrcoef(rs, rb)[0, 1]; out["ranking"][cp] = dict(spearman=float(rho), reversals=int(rev), pairs=len(E) * (len(E) - 1) // 2)
    print(f"  band t={cp:>4s}: Spearman(标量排名, 带排名) = {rho:.2f}；成对名次反转 {rev}/{len(E)*(len(E)-1)//2}")
close = [(a, b_, abs(res[a]["pcc"] - res[b_]["pcc"]), res[a]["bands"][cps[0]] - res[b_]["bands"][cps[0]]) for i, a in enumerate(E) for b_ in E[i + 1:] if abs(res[a]["pcc"] - res[b_]["pcc"]) < 0.01]
close.sort(key=lambda t: -abs(t[3])); out["close_pairs"] = close[:8]
print("PCC 相差 <0.01 而 β1 相差最大的对:", [(a, b_, round(dp, 4), round(db, 3)) for a, b_, dp, db in close[:5]])
json.dump(out, open(f"{R}/xen_multi_rank.json", "w"), indent=1)

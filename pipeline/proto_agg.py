"""协议效应 vs 方法效应：同一片、同一模型、同一编码器，只改空间块划分的粗细。

g 越小，块越大、训练与测试的空间隔离越强。
FINDINGS §2.4 在 Visium HD 两片上测到协议效应 0.20–0.30 PCC，是方法间差异(≈0.03)的 10 倍。
本表检验该结论在 Xenium 16 片上是否成立。
"""
import json, glob, os
import numpy as np
from collections import defaultdict

GR = [16, 8, 4, 2]
d = defaultdict(dict)
for g in GR:
    for f in glob.glob(f"results/proto_g{g}/*.json"):
        j = json.load(open(f))
        n = j["name"].replace(f"_g{g}", "")
        d[n][g] = j

full = {k: v for k, v in d.items() if len(v) >= 2}
print(f"有 ≥2 档的样本: {len(full)}/{len(d)}\n")
avail = sorted({g for v in full.values() for g in v}, reverse=True)
hdr = "样本".ljust(40) + "".join(f"g={g}".rjust(9) for g in avail)
print(hdr + "\n" + "-" * len(hdr))
for k in sorted(full):
    line = k[:38].ljust(40)
    for g in avail:
        v = full[k].get(g)
        line += (f"{v['pcc']:.4f}" if v and isinstance(v.get("pcc"), float) else "—").rjust(9)
    print(line)

print()
stats = {}
for g in avail:
    v = [full[k][g]["pcc"] for k in full if g in full[k]]
    s = [full[k][g]["eq_sigma"] for k in full if g in full[k]
         and full[k][g].get("eq_flag") == "ok" and full[k][g]["eq_sigma"] == full[k][g]["eq_sigma"]]
    if v:
        stats[g] = (np.median(v), np.median(s) if s else float("nan"), len(v))
        print(f"g={g:<3} n={len(v):2d}  PCC 中位 {np.median(v):.4f}  σ 中位 "
              f"{np.median(s) if s else float('nan'):6.1f}µm")

# 协议效应 = 同一片内跨划分的 PCC 极差
swing = [max(full[k][g]["pcc"] for g in full[k]) - min(full[k][g]["pcc"] for g in full[k])
         for k in full if len(full[k]) >= 2]
print(f"\n协议效应（同一片内跨划分的 PCC 极差）:")
print(f"  中位 {np.median(swing):.4f}   范围 [{min(swing):.4f}, {max(swing):.4f}]   n={len(swing)}")
print(f"\n参照 —— 方法间差异（同协议下）:")
print(f"  广度线 15 塔 PCC 跨度 0.1094（0.1894→0.2988）")
print(f"  方法·跨片 5 法 PCC 极差 0.0447（0.4368→0.4815）")
if swing:
    print(f"\n协议效应 / 方法效应 = {np.median(swing)/0.0447:.1f}×（对方法·跨片）")

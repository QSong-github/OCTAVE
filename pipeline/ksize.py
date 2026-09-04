import json, glob
import numpy as np
rows = []
for f in sorted(glob.glob("results/hest_floor/*.json")):
    d = json.load(open(f))
    ns = [v["n"] for v in d["samples"].values()]
    rows.append((d["cohort"], len(ns), int(np.median(ns)), int(sum(ns))))
hdr = ("队列".ljust(12) + "样本".rjust(6) + "spot中位".rjust(12)
       + "留一训练集".rjust(14) + "k=200占比".rjust(12) + "k=800占比".rjust(12))
print(hdr); print("-" * 70)
warn = []
for c, n, med, tot in sorted(rows, key=lambda r: -r[1]):
    tr = max(tot - med, 1)
    p2, p8 = 100 * 200 / tr, 100 * 800 / tr
    flag = ""
    if p8 > 5:
        warn.append((c, "k=800", p8)); flag = "  <-- k=800 退化风险"
    if p2 > 5:
        warn.append((c, "k=200", p2)); flag += "  <-- k=200 也偏大"
    print(f"{c:<12s}{n:>6d}{med:>12d}{tr:>14d}{p2:>11.2f}%{p8:>11.2f}%{flag}")
print()
if warn:
    print("邻域占训练集 >5% 的组合（检索退化为近似全局均值，下界被人为削弱）:")
    for c, k, p in warn:
        print(f"  {c:<12s} {k:<7s} {p:.1f}%")
else:
    print("所有队列在 k=800 下邻域占比均 <5%，无退化风险。")

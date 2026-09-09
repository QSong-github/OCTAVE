# -*- coding: utf-8 -*-
"""把三条消融轴汇到样本层级。口径与正文一致：逐区域 → 样本内中位 → 跨样本中位。"""
import glob, json, os, re
import numpy as np
from math import comb
R = "/path/to/systema4ST/results"


def spec(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    return m.group(1) if m else n


def signp(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


D = [json.load(open(f)) for f in sorted(glob.glob(R + "/blocks_xen_ablate/*.json"))]
print("区域 %d 个" % len(D))
if len(D) != 16:
    print("⚠ 不足 16 个区域，先不汇总")
    raise SystemExit(1)

OUT = {}
for axis, default in (("alpha", "10000.0"), ("ngene", "200"), ("K", "20")):
    keys = sorted(D[0][axis], key=float)
    OUT[axis] = {"default": default, "levels": {}}
    print("\n=== %s（默认 %s）" % (axis, default))
    print("%10s %9s %9s %9s %9s %9s %10s %8s" %
          ("level", "share%", "scalar%", "fine%", "ratio", "sd(seed)", "specimens", "P"))
    for k in keys:
        per, sdv = {}, []
        for d in D:
            rs = d[axis][k]
            if not rs:
                continue
            per.setdefault(spec(d["name"]), []).append(
                (float(np.median([r["overall"] for r in rs])),
                 float(np.median([r["fineband"] for r in rs])),
                 float(np.median([r["ratio"] for r in rs])),
                 float(np.median([r["share"] for r in rs]))))
            sdv.append(float(np.std([r["ratio"] for r in rs], ddof=1)))
        if not per:
            continue
        ov = [np.median([x[0] for x in v]) for v in per.values()]
        fi = [np.median([x[1] for x in v]) for v in per.values()]
        rt = [np.median([x[2] for x in v]) for v in per.values()]
        sh = [np.median([x[3] for x in v]) for v in per.values()]
        w = sum(1 for a, b in zip(ov, fi) if b > a)
        OUT[axis]["levels"][k] = dict(
            n_specimens=len(per), share=float(np.median(sh)),
            scalar=float(np.median(ov)), fine=float(np.median(fi)),
            ratio=float(np.median(rt)), ratio_min=float(min(rt)), ratio_max=float(max(rt)),
            seed_sd_median=float(np.median(sdv)), specimens_agree=w, P=signp(w, len(per)))
        o = OUT[axis]["levels"][k]
        print("%10s %9.1f %9.1f %9.1f %9.2f %9.3f %10s %8.4f" %
              (k, o["share"], o["scalar"], o["fine"], o["ratio"], o["seed_sd_median"],
               "%d/%d" % (w, len(per)), o["P"]))

# 结论稳健性
print("\n=== 三条轴上的结论方向")
for axis in OUT:
    L = OUT[axis]["levels"]
    allw = all(v["specimens_agree"] == v["n_specimens"] for v in L.values())
    rr = [v["ratio"] for v in L.values()]
    print("  %-6s 比值跨越 %.2f–%.2f；每一档都 %s"
          % (axis, min(rr), max(rr),
             "8/8 同向" if allw else "**有档位不是全同向**: " +
             str({k: "%d/%d" % (v["specimens_agree"], v["n_specimens"])
                  for k, v in L.items() if v["specimens_agree"] != v["n_specimens"]})))
json.dump(OUT, open(R + "/ablate_spread.json", "w"), indent=1)
print("\n-> %s/ablate_spread.json" % R)

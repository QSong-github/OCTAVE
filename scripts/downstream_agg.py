# -*- coding: utf-8 -*-
"""附录图 7 的六个下游读数：逐读数核对 16→64 µm 的退化在多少区域/样本成立。

正文此前笼统写「六个读数全部 15/15 区域、7/7 样本、P=0.0156」，从未逐读数验过。
"""
import glob, json, os, re
import numpy as np
from math import comb
R = "/path/to/systema4ST/results"


def spec(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    if "Human_Lung_Cancer_FFPE" in n: return "Lung"   # v1 与 Prime 5K 同一供体同一组织块，按一个标本计
    return m.group(1) if m else n


def signp(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


# (键, 来源, 面板, 变差的方向: -1 表示越小越差, +1 表示越大越差)
M = [("ari", "downstream", "a  spatial-domain ARI", -1),
     ("edge_ratio", "downstream", "b  interface gradient fidelity", -1),
     ("coloc_preserve", "downstream2", "c  gene co-localisation", -1),
     ("hotspot_jaccard", "downstream2", "d  hotspot Jaccard", -1),
     ("boundary_shift_um", "downstream2", "e  interface displacement", +1),
     ("hotspot_recall_by_size", "downstream2", "f  hotspot recall by size", -1)]

D1, D2 = {}, {}
for f in sorted(glob.glob(R + "/downstream_*.json")):
    if "downstream2" in f: continue
    d = json.load(open(f)); D1[d["name"]] = d["scales"]
for f in sorted(glob.glob(R + "/downstream2_*.json")):
    d = json.load(open(f)); D2[d["name"]] = d["scales"]
names = sorted(set(D1) & set(D2))
print("区域 %d 个（两套文件都齐全）" % len(names))

OUT = {}
print("\n%-34s %10s %10s %8s %8s %9s" % ("读数", "16 µm", "64 µm", "区域", "样本", "P(样本)"))
for key, src, panel, worse in M:
    S = (D1 if src == "downstream" else D2)
    a16, a64, per = [], [], {}
    for n in names:
        sc = S[n]
        v16, v64 = sc.get("16", {}).get(key), sc.get("64", {}).get(key)
        if isinstance(v16, dict) or isinstance(v64, dict):   # 按热点大小分层：{类: [召回, n]}
            ORD = ["<50\u00b5m", "50-100\u00b5m", "100-200\u00b5m", "\u2265200\u00b5m"]
            ks = [k for k in ORD if k in v16 and k in v64]
            if len(ks) < 2: continue
            g = lambda z: z[0] if isinstance(z, (list, tuple)) else z
            # 选择性 = 最大类的召回 − 最小类的召回；越小表示对大热点越无偏好
            v16 = g(v16[ks[-1]]) - g(v16[ks[0]]); v64 = g(v64[ks[-1]]) - g(v64[ks[0]])
        if v16 is None or v64 is None or not np.isfinite([v16, v64]).all(): continue
        a16.append(v16); a64.append(v64)
        per.setdefault(spec(n), []).append(worse * (v64 - v16))
    if not a16: print("%-34s （无数据）" % panel); continue
    reg = sum(1 for x, y in zip(a16, a64) if worse * (y - x) > 0)
    sp = {k: float(np.median(v)) for k, v in per.items()}
    nsp = sum(1 for v in sp.values() if v > 0)
    P = signp(nsp, len(sp))
    OUT[key] = dict(panel=panel, source=src, v16=float(np.median(a16)), v64=float(np.median(a64)),
                    n_regions=len(a16), regions_worse=reg, n_specimens=len(sp),
                    specimens_worse=nsp, P=P, worse_direction=worse)
    print("%-34s %10.3f %10.3f %8s %8s %9.4f"
          % (panel, np.median(a16), np.median(a64), "%d/%d" % (reg, len(a16)),
             "%d/%d" % (nsp, len(sp)), P))
allreg = all(v["regions_worse"] == v["n_regions"] for v in OUT.values())
allsp = all(v["specimens_worse"] == v["n_specimens"] for v in OUT.values())
print("\n六个读数全部区域一致: %s；全部样本一致: %s" % (allreg, allsp))
if not allreg:
    print("  不满 100%% 的读数:", {v["panel"]: "%d/%d" % (v["regions_worse"], v["n_regions"])
                                 for v in OUT.values() if v["regions_worse"] != v["n_regions"]})
json.dump(OUT, open(R + "/downstream_summary.json", "w"), indent=1)
print("-> %s/downstream_summary.json" % R)

# -*- coding: utf-8 -*-
"""把 16 个 KMeans 种子汇成表 1 的误差棒。

口径与正文一致：**逐区域算比值**，再在样本内取中位，再跨 8 个样本取中位。
对每个种子完整走一遍这条链，于是表头的 3.61 得到 16 个取值，而不是一个点。
"""
import glob, json, os, re
import numpy as np
from math import comb
R = "/blue/qsong1/wang.qing/systema4ST/results"


def spec(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    return m.group(1) if m else n


def signp(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


F = sorted(glob.glob(R + "/blocks_xen_seeds/*.json"))
D = [json.load(open(f)) for f in F]
print("区域 %d 个" % len(D))
anchor = [d for d in D if d.get("frozen_check") and not d["frozen_check"]["reproduces"]]
if anchor:
    print("锚点不一致的区域:", [d["name"] for d in anchor])
else:
    print("锚点：全部 %d 个区域的 seed 0 与冻结值逐位相同" % sum(1 for d in D if d.get("frozen_check")))

NS = min(len(d["seeds"]) for d in D)
head, per_spec_all = [], {}
for s in range(NS):
    by = {}
    for d in D:
        r = d["seeds"][s]
        by.setdefault(spec(d["name"]), []).append(r)
    med = {k: float(np.median([x["ratio"] for x in v])) for k, v in by.items()}
    ov = {k: float(np.median([x["overall"] for x in v])) for k, v in by.items()}
    fi = {k: float(np.median([x["fineband"] for x in v])) for k, v in by.items()}
    sh = {k: float(np.median([x["share"] for x in v])) for k, v in by.items()}
    won = sum(1 for k in med if fi[k] > ov[k])
    head.append(dict(seed=s, ratio_median=float(np.median(list(med.values()))),
                     ratio_min=float(min(med.values())), ratio_max=float(max(med.values())),
                     scalar_median=float(np.median(list(ov.values()))),
                     fine_median=float(np.median(list(fi.values()))),
                     share_median=float(np.median(list(sh.values()))),
                     share_min=float(min(sh.values())), share_max=float(max(sh.values())),
                     n_specimens=len(med), specimens_agree=won, P=signp(won, len(med))))
    for k, v in med.items():
        per_spec_all.setdefault(k, []).append(v)
    print("seed %2d: 8 样本比值中位 %.3f  标量落差 %.1f%%  细带落差 %.1f%%  "
          "块占比 %.1f%%  同向 %d/%d  P=%.4f"
          % (s, head[-1]["ratio_median"], head[-1]["scalar_median"], head[-1]["fine_median"],
             head[-1]["share_median"], won, len(med), head[-1]["P"]))

v = np.array([h["ratio_median"] for h in head])
ag = [h["specimens_agree"] for h in head]
ar = [x["ari_vs_seed0"] for d in D for x in d["seeds"][1:]]
S = dict(n_seeds=NS, n_regions=len(D), n_specimens=head[0]["n_specimens"],
         headline_ratio_seed0=head[0]["ratio_median"],
         headline_ratio_median=float(np.median(v)), headline_ratio_min=float(v.min()),
         headline_ratio_max=float(v.max()), headline_ratio_sd=float(v.std(ddof=1)),
         scalar_median=float(np.median([h["scalar_median"] for h in head])),
         scalar_sd=float(np.std([h["scalar_median"] for h in head], ddof=1)),
         fine_median=float(np.median([h["fine_median"] for h in head])),
         fine_sd=float(np.std([h["fine_median"] for h in head], ddof=1)),
         share_median=float(np.median([h["share_median"] for h in head])),
         share_sd=float(np.std([h["share_median"] for h in head], ddof=1)),
         seeds_all_specimens_agree=int(sum(1 for a in ag if a == head[0]["n_specimens"])),
         min_specimens_agree=int(min(ag)), max_P=float(max(h["P"] for h in head)),
         ari_median=float(np.median(ar)), ari_min=float(np.min(ar)), ari_max=float(np.max(ar)))
print("\n=== 表 1 的误差棒（%d 个种子）" % NS)
print("表头比值        中位 %.2f  [%.2f, %.2f]  sd %.3f   （seed 0 = %.2f，即冻结值）"
      % (S["headline_ratio_median"], S["headline_ratio_min"], S["headline_ratio_max"],
         S["headline_ratio_sd"], S["headline_ratio_seed0"]))
print("标量落差        %.1f%% ± %.2f      细带落差 %.1f%% ± %.2f"
      % (S["scalar_median"], S["scalar_sd"], S["fine_median"], S["fine_sd"]))
print("块预言机占比    %.1f%% ± %.2f" % (S["share_median"], S["share_sd"]))
print("8/8 样本同向的种子 %d/%d，最差 %d/8，最大 P = %.4f"
      % (S["seeds_all_specimens_agree"], NS, S["min_specimens_agree"], S["max_P"]))
print("划分间 ARI      中位 %.3f  [%.3f, %.3f]  ← 划分本身在变"
      % (S["ari_median"], S["ari_min"], S["ari_max"]))
json.dump(dict(summary=S, by_seed=head, by_specimen=per_spec_all,
               by_region={d["name"]: d["summary"] for d in D}),
          open(R + "/seed_spread.json", "w"), indent=1)
print("\n-> %s/seed_spread.json" % R)

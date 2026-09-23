# -*- coding: utf-8 -*-
"""扩集后的重算：把全部编码器（原 15 + 新 12）放进本文表 2 的框架。
口径与原表完全一致：报告分数逐样本均值；队列统计量为队列内逐样本配对差的中位数。
出口断言：赢的队列数必须与逐队列明细自洽。"""
import glob, json, os, re
import numpy as np
from math import comb
R = "/path/to/project/results"


def signp(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


enc = sorted({os.path.basename(f)[len("hest_effres_ps_"):-5]
              for f in glob.glob(R + "/hest_effres_ps_*.json")})
OUT, skip = {}, []
for e in enc:
    fs = sorted(glob.glob(R + "/hest_floor_%s/*.json" % e))
    if len(fs) != 10:
        skip.append((e, "k=50 地板 %d/10" % len(fs))); continue
    M = json.load(open(R + "/hest_effres_ps_%s.json" % e))["per_sample_pcc"]
    fl, coh = {}, {}
    for f in fs:
        d = json.load(open(f))
        for s, v in d["samples"].items():
            fl[s] = v["pcc"]; coh[s] = d["cohort"]
    ids = sorted(set(fl) & set(M))
    if len(ids) != 72:
        skip.append((e, "配对样本 %d/72" % len(ids))); continue
    dif = {s: M[s] - fl[s] for s in ids}
    by = {}
    for s in ids:
        by.setdefault(coh[s], []).append(dif[s])
    med = np.array([np.median(v) for v in by.values()])
    won = int((med > 0).sum())
    sem = float(med.std(ddof=1) / np.sqrt(len(med)))
    ks = {}
    for k in (10, 200, 800):
        g = sorted(glob.glob(R + "/hest_floor_k%d_%s/*.json" % (k, e)))
        if len(g) != 10: continue
        f2 = {}
        for f in g:
            d = json.load(open(f))
            for s, v in d["samples"].items(): f2[s] = v["pcc"]
        i2 = sorted(set(f2) & set(M))
        b2 = float(np.mean([f2[s] for s in i2]))
        ks[k] = 100 * (np.mean([M[s] for s in i2]) - b2) / b2
    f0 = float(np.mean([fl[s] for s in ids]))
    m0 = float(np.mean([M[s] for s in ids]))
    OUT[e] = dict(model=m0, floor=f0, rel=100 * (m0 - f0) / f0,
                  cohort_mean=float(med.mean()), cohort_sem=sem,
                  t=float(med.mean() / sem) if sem > 0 else None,
                  cohorts_won=won, P=signp(won, len(med)), rel_by_k=ks)
print("编码器 %d 个（跳过 %d）" % (len(OUT), len(skip)))
for e, why in skip: print("  跳过 %-16s %s" % (e, why))
print()
print("%-16s %8s %8s %8s %10s %8s %8s" % ("编码器", "PCC", "地板", "余量%", "队列均值", "|t|", "赢"))
for e in sorted(OUT, key=lambda x: -OUT[x]["model"]):
    o = OUT[e]
    print("%-16s %8.4f %8.4f %+8.1f  %+.4f±%.4f %6.2f %6s"
          % (e, o["model"], o["floor"], o["rel"], o["cohort_mean"], o["cohort_sem"],
             abs(o["t"]), "%d/10" % o["cohorts_won"]))
below = [e for e in OUT if OUT[e]["rel"] < 0]
t2 = [e for e in OUT if abs(OUT[e]["t"]) >= 2]
print("\n低于自身地板: %d/%d" % (len(below), len(OUT)))
print("余量达到两个标准误: %d/%d %s" % (len(t2), len(OUT), t2))
sig = [e for e in OUT if OUT[e]["P"] < 0.05 and OUT[e]["cohort_mean"] > 0]
print("队列层级显著为正: %d/%d %s" % (len(sig), len(OUT), sig))
json.dump(OUT, open(R + "/bench27.json", "w"), indent=1)
print("-> %s/bench27.json" % R)

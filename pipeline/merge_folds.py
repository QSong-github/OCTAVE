# -*- coding: utf-8 -*-
"""把逐折并行产生的分片合并成与其它方法同结构的一个 JSON。"""
import glob, json, os, sys
import numpy as np
R = "/path/to/systema4ST/results"
tag = sys.argv[1]                      # heclip / hggep
out = os.path.join(R, "%s_hest.json" % tag)
acc = {}
fs = sorted(glob.glob(os.path.join(R, "%s_*_f*.json" % tag)))
for f in fs:
    for s, d in json.load(open(f)).items():
        acc.setdefault(s, {"cohort": d["cohort"], "folds": []})["folds"] += d["folds"]
res = {s: {"cohort": d["cohort"], "folds": d["folds"], "pcc": float(np.mean(d["folds"]))}
       for s, d in acc.items()}
json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
byc = {}
for s, d in res.items(): byc.setdefault(d["cohort"], []).append(d["pcc"])
print("%s：%d 个分片 → %d 个样本 / %d 个队列" % (tag, len(fs), len(res), len(byc)))
for c in sorted(byc):
    v = byc[c]
    print("  %-10s n=%2d  PCC=%.4f  [%.4f, %.4f]" % (c, len(v), np.mean(v), min(v), max(v)))
z = [s for s, d in res.items() if abs(d["pcc"]) < 1e-9]
print("  恰好为零的样本: %d 个 %s" % (len(z), z[:6] if z else ""))
print("  合计逐样本均值 %.4f  -> %s" % (np.mean([d["pcc"] for d in res.values()]), out))

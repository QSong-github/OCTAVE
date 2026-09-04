# -*- coding: utf-8 -*-
"""配对下界的**早期形状检查**（部分数据，不是结论）。

只在两边都有的样本上做同特征配对；每个编码器报它自己已完成的队列。
目的：确认方向与量级合理，而不是出结论。最终结论等 150/150 齐了再由
matched_matrix.py 按队列层级出。
"""
import json, glob, os
import numpy as np
from math import comb


def signp(k, n):
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


PS = {}
for f in sorted(glob.glob("results/hest_effres_ps_*.json")):
    d = json.load(open(f)); PS[d["encoder"]] = d
L = json.load(open("results/hest_ladder.json"))

print(f"{'编码器':<16s}{'队列':>5s}{'样本':>5s}{'整体Δ':>9s}{'相对':>8s}"
      f"{'细带Δ':>9s}{'相对':>8s}{'倍':>6s}{'队列胜':>8s}{'P':>9s}")
ALL = []
for e in sorted(PS):
    dirp = f"results/hest_floor_{e}"
    if not os.path.isdir(dirp):
        continue
    fl_p, fl_b, coh = {}, {}, {}
    for f in sorted(glob.glob(f"{dirp}/*.json")):
        d = json.load(open(f))
        for s, v in d["samples"].items():
            fl_p[s] = v["pcc"]; fl_b[s] = v["band_pcc"]; coh[s] = d["cohort"]
    ids = sorted(set(fl_p) & set(PS[e]["per_sample_band_pcc"]))
    if len(ids) < 10:
        continue
    t0 = str(min(int(t) for t in PS[e]["sigma_um"]))
    do = np.array([PS[e]["per_sample_pcc"][s] - fl_p[s] for s in ids])
    db = np.array([PS[e]["per_sample_band_pcc"][s][t0] - fl_b[s][t0] for s in ids])
    fo = np.mean([fl_p[s] for s in ids]); fb = np.mean([fl_b[s][t0] for s in ids])
    by = {}
    for i, s in enumerate(ids):
        by.setdefault(coh[s], []).append(db[i])
    k = sum(1 for v in by.values() if np.median(v) > 0); n = len(by)
    ro, rb = 100 * do.mean() / fo, 100 * db.mean() / fb
    ALL.append((e, n, len(ids), do.mean(), ro, db.mean(), rb, rb / ro if ro else np.nan,
                k, signp(k, n)))
    print(f"{e:<16s}{n:>5d}{len(ids):>5d}{do.mean():>+9.4f}{ro:>+7.2f}%"
          f"{db.mean():>+9.4f}{rb:>+7.2f}%{(rb/ro if ro else np.nan):>6.2f}"
          f"{k:>5d}/{n}{signp(k, n):>9.5f}")

if ALL:
    a = np.array([[r[4], r[6], r[7]] for r in ALL], float)
    print(f"\n{len(ALL)} 个编码器（部分队列）: 整体相对差中位 {np.median(a[:, 0]):+.2f}%  "
          f"细带 {np.median(a[:, 1]):+.2f}%  倍数中位 {np.median(a[:, 2]):.2f}×")
    print(f"整体差 > 0 的编码器: {int((a[:, 0] > 0).sum())}/{len(ALL)}；"
          f"细带差 > 0 的: {int((a[:, 1] > 0).sum())}/{len(ALL)}")
    print("\n[注意] 各编码器覆盖的队列不同，上面这几行是形状检查，不是结论。")

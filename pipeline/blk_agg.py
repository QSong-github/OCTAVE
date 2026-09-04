import os
# -*- coding: utf-8 -*-
"""块预言机在 30 个编码器上的汇总（PCC 单位，不做逐样本比值）。

纪律：队列层级的逐样本 PCC 可以接近零（HCC 约 0.07），逐样本比值会炸。
因此统计量一律用**配对差** blk − mod，单位是 PCC；占比只在聚合后的均值上算一次，
即 mean(blk)/mean(mod)，并标明它是比值的比而非比的比。
队列统计量 = 队列内逐样本配对差的中位数（与 cohort_spread.py 同口径）；
再给 10 个队列统计量的均值±sem、|t|、赢的队列数与精确双侧符号检验。
分母两套：官方 ridge（α=100/(D·G)，欠正则）与留一队列选 α 的 ridge。以后者为准。"""
import json, glob, os, numpy as np
from math import comb
R = "/blue/qsong1/wang.qing/systema4ST/results"
signp = lambda k, n: min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
ENC = sorted(x for x in (os.path.basename(f)[len("hest_blocks_"):-5] for f in [f for f in glob.glob(f"{R}/hest_blocks_*.json") if not os.path.basename(f).startswith(("hest_blocks_z_", "hest_blocks_summary"))]) if x != "summary")
rows = []
for e in ENC:
    D = json.load(open(f"{R}/hest_blocks_{e}.json"))["samples"]
    off = json.load(open(f"{R}/hest_effres_ps_{e}.json"))["per_sample_pcc"]
    sel = json.load(open(f"{R}/hest_rsel_ps_{e}.json"))["per_sample_pcc"]
    ids = sorted(set(D) & set(off) & set(sel)); coh = {s: D[s]["cohort"] for s in ids}
    r = {"enc": e, "n_samples": len(ids), "n_cohorts": len(set(coh.values()))}
    for k in (20, 50, 200):
        key = f"blk_k{k}"
        for tag, mod in (("off", off), ("sel", sel)):
            by = {}
            for s in ids: by.setdefault(coh[s], []).append(D[s][key] - mod[s])
            g = np.array([np.median(v) for _, v in sorted(by.items())])
            sem = float(g.std(ddof=1) / np.sqrt(len(g))); won = int((g > 0).sum())
            mb = float(np.mean([D[s][key] for s in ids])); mm = float(np.mean([mod[s] for s in ids]))
            r[f"{key}_{tag}"] = dict(cohort_mean=float(g.mean()), cohort_sem=sem,
                                     t=float(g.mean() / sem) if sem > 0 else None, won=won,
                                     P=signp(won, len(g)), blk=mb, mod=mm, share=100.0 * mb / mm)
    rows.append(r)
rows.sort(key=lambda x: -x["blk_k20_sel"]["share"])
print("块预言机 vs 训练模型，PCC 单位的配对差（队列内中位 → 10 个队列的均值±sem）")
print("占比 = mean(blk)/mean(mod)，在聚合后算一次，不是逐样本比值的中位\n")
print("%-14s | %-34s | %-34s" % ("编码器", "K=20，分母=官方 α", "K=20，分母=留一选 α"))
print("%-14s | %7s %18s %5s %6s | %7s %18s %5s %6s" % ("", "占比", "配对差±sem", "|t|", "赢", "占比", "配对差±sem", "|t|", "赢"))
for r in rows:
    a, b = r["blk_k20_off"], r["blk_k20_sel"]
    print("%-14s | %6.1f%% %+9.4f±%.4f %5.2f %4d/10 | %6.1f%% %+9.4f±%.4f %5.2f %4d/10" % (
        r["enc"], a["share"], a["cohort_mean"], a["cohort_sem"], abs(a["t"]), a["won"],
        b["share"], b["cohort_mean"], b["cohort_sem"], abs(b["t"]), b["won"]))
print()
for k in (20, 50, 200):
    for tag, lab in (("off", "官方 α"), ("sel", "选 α ")):
        v = [r[f"blk_k{k}_{tag}"] for r in rows]
        sh = [x["share"] for x in v]
        print("K=%-4d %s 占比中位 %6.1f%% (%.1f–%.1f)；配对差为正 %2d/30，|t|≥2 %2d/30，符号检验 P<0.05 %2d/30" % (
            k, lab, np.median(sh), min(sh), max(sh),
            sum(1 for x in v if x["cohort_mean"] > 0), sum(1 for x in v if abs(x["t"]) >= 2),
            sum(1 for x in v if x["P"] < 0.05 and x["cohort_mean"] > 0)))
json.dump(rows, open(f"{R}/hest_blocks_summary.json", "w"), indent=1)
print("\n已存 results/hest_blocks_summary.json")

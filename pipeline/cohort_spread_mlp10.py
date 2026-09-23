# -*- coding: utf-8 -*-
"""表 2 的误差棒：与 k_sens.py 完全同一口径的队列层级离散度。

口径必须与「赢的队列数」和符号检验一致，否则同一行里两个数说的不是一件事。
k_sens.py 的队列统计量 = 该队列内**逐样本配对差的中位数**；本脚本沿用它，
再给这 10 个队列统计量的均值与标准误。

只用 PCC 单位的绝对量。**不做逐队列比值** —— 队列地板可以接近零，
相对量会炸（本项目已四次踩到近零分母）。

出口带断言：赢的队列数必须与 results/k_sensitivity_mlp10.json 逐个相等。
"""
import glob, json, os
import numpy as np
R = "/path/to/project/results"
# 编码器表自动发现（2026-08-29 从 15 扩到 27）。omiclip_raw 是 omiclip 的
# 未归一化变体，同一模型，排除以免占两行。
import glob as _g, os as _o
ENC = sorted({_o.path.basename(f)[len("hest_mlp10_ps_"):-5]
              for f in _g.glob(R + "/hest_mlp10_ps_*.json")} - {"omiclip_raw"})
K50 = json.load(open(R + "/k_sensitivity_mlp10.json"))["50"]
OUT, bad = {}, []
for e in ENC:
    ps = R + "/hest_mlp10_ps_%s.json" % e
    fs = sorted(glob.glob(R + "/hest_floor_%s/*.json" % e))
    if not os.path.exists(ps) or len(fs) != 10:
        print("跳过", e, os.path.exists(ps), len(fs))
        continue
    M = json.load(open(ps))["per_sample_pcc"]
    fl, coh = {}, {}
    for f in fs:
        d = json.load(open(f))
        for s, v in d["samples"].items():
            fl[s] = v["pcc"]
            coh[s] = d["cohort"]
    ids = sorted(set(fl) & set(M))
    dif = {s: M[s] - fl[s] for s in ids}
    by = {}
    for s in ids:
        by.setdefault(coh[s], []).append(dif[s])
    rows = [dict(cohort=c, n_samples=len(v), median_paired_diff=float(np.median(v)),
                 mean_model=float(np.mean([M[s] for s in ids if coh[s] == c])),
                 mean_floor=float(np.mean([fl[s] for s in ids if coh[s] == c])))
            for c, v in sorted(by.items())]
    g = np.array([r["median_paired_diff"] for r in rows])
    n = len(rows)
    sem = float(g.std(ddof=1) / np.sqrt(n))
    won = int((g > 0).sum())
    f0 = float(np.mean([fl[s] for s in ids]))
    OUT[e] = dict(n_cohorts=n, n_samples=len(ids), statistic="median paired per-sample difference, by cohort",
                  cohort_mean=float(g.mean()), cohort_sem=sem,
                  cohort_median=float(np.median(g)), cohort_min=float(g.min()),
                  cohort_max=float(g.max()), t=float(g.mean() / sem) if sem > 0 else None,
                  cohorts_won=won, floor_mean_sample=f0,
                  model_mean_sample=float(np.mean([M[s] for s in ids])),
                  margin_mean_sample=float(np.mean(list(dif.values()))), by_cohort=rows)
    if won != K50[e]["n_cohorts_win"]:
        bad.append((e, won, K50[e]["n_cohorts_win"]))
    print("%-16s n=%2d 队列 / %2d 样本  队列统计量 %+.4f +- %.4f  |t|=%.2f  赢 %d/%d  (k_sens %d)"
          % (e, n, len(ids), g.mean(), sem, abs(g.mean() / sem), won, n, K50[e]["n_cohorts_win"]))
assert not bad, "赢的队列数与 k_sensitivity 不符: %s" % bad
tm = max(abs(OUT[e]["t"]) for e in OUT)
print("\n口径一致性断言通过：%d 个编码器的赢队列数与 k_sensitivity 逐个相等" % len(OUT))
print("跨队列 |t| 最大 %.2f（%s）" % (tm, max(OUT, key=lambda e: abs(OUT[e]["t"]))))
json.dump(OUT, open(R + "/cohort_spread_mlp10.json", "w"), indent=1)
print("%d 个编码器 -> %s/cohort_spread_mlp10.json" % (len(OUT), R))

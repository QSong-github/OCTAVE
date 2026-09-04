# -*- coding: utf-8 -*-
"""§42 补测：同特征配对(phikon_v2 ridge vs phikon_v2 kNN k=50)的**逐带队列层级**符号检验。

§41 原文把这项列为「尚缺」。逐样本带 PCC 现已齐备(hest_effres_ps_*.json，
与主文件逐位一致)，可以补上。检验单元是队列(n=10，最小可得 P=0.00195)，
不是样本(72 个里 47 个来自 CCRCC+PRAD，按样本计数是伪重复)。
"""
import json, glob
import numpy as np
from math import comb


def signp(k, n):
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


fl_p, fl_b, coh = {}, {}, {}
for f in sorted(glob.glob("results/hest_floor/*.json")):
    d = json.load(open(f))
    assert d["encoder"] == "phikon_v2"
    for s, v in d["samples"].items():
        fl_p[s] = v["pcc"]; fl_b[s] = v["band_pcc"]; coh[s] = d["cohort"]
E = json.load(open("results/hest_effres_ps_phikon_v2.json"))
ps_p, ps_b = E["per_sample_pcc"], E["per_sample_band_pcc"]
ids = sorted(set(fl_p) & set(ps_b))
ts = sorted(int(t) for t in E["sigma_um"])
sig = {t: E["sigma_um"][str(t)] for t in ts}
print(f"配对样本 {len(ids)}/72，队列 {len(set(coh[s] for s in ids))}")
print(f"下界 encoder={d['encoder']} k={d['k']}（与 ridge 同 PCA-256 管线，仅回归→检索）\n")


def test(vals):
    by = {}
    for s in ids:
        by.setdefault(coh[s], []).append(vals[s])
    med = {c: float(np.median(v)) for c, v in by.items()}
    k = sum(1 for v in med.values() if v > 0)
    return k, len(med), signp(k, len(med)), med, \
        int(sum(1 for v in vals.values() if v > 0))


print(f"{'σ(µm)':>8s}{'池化差':>10s}{'相对':>9s}{'样本胜':>9s}{'队列胜':>9s}{'P':>10s}")
rows = {}
for t in ts:
    dif = {s: ps_b[s][str(t)] - fl_b[s][str(t)] for s in ids}
    k, n, P, med, nw = test(dif)
    fl = float(np.mean([fl_b[s][str(t)] for s in ids]))
    g = float(np.mean(list(dif.values())))
    rows[t] = {"sigma_um": sig[t], "gap": g, "rel": g / fl,
               "n_cohorts_win": k, "n_cohorts": n, "P": P, "n_samples_win": nw}
    print(f"{sig[t]:>8.0f}{g:>+10.4f}{100*g/fl:>+8.2f}%{nw:>6d}/72{k:>6d}/{n}{P:>10.5f}")
d0 = {s: ps_p[s] - fl_p[s] for s in ids}
k, n, P, med0, nw = test(d0)
f0 = float(np.mean([fl_p[s] for s in ids])); g0 = float(np.mean(list(d0.values())))
print(f"{'全谱':>8s}{g0:>+10.4f}{100*g0/f0:>+8.2f}%{nw:>6d}/72{k:>6d}/{n}{P:>10.5f}")

best = min(rows, key=lambda t: rows[t]["P"])
print(f"\n所有带里 P 最小的一档: σ={rows[best]['sigma_um']:.0f}µm  "
      f"{rows[best]['n_cohorts_win']}/{rows[best]['n_cohorts']}  P={rows[best]['P']:.5f}")
print(f"显著(P<0.05)的带: {sum(1 for r in rows.values() if r['P'] < 0.05)}/{len(rows)}")
print(f"\n队列中位差(全谱):")
for c in sorted(med0, key=lambda c: -med0[c]):
    print(f"  {c:<11s}{med0[c]:+.4f}")
json.dump({"encoder": "phikon_v2", "floor_k": d["k"], "n": len(ids),
           "bands": {str(t): rows[t] for t in ts},
           "overall": {"gap": g0, "rel": g0 / f0, "n_cohorts_win": k,
                       "n_cohorts": n, "P": P, "cohort_median": med0}},
          open("results/matched_band_test_phikon_v2.json", "w"), indent=1)
print("\n已存 results/matched_band_test_phikon_v2.json")

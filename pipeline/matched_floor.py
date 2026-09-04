# -*- coding: utf-8 -*-
"""下界用的是 phikon_v2 特征的 kNN。唯一同特征、可归因于「回归 vs 检索」的比较
是 phikon_v2-ridge vs phikon_v2-kNN。其余 14 个编码器与下界的差里混着特征差异。"""
import json, glob
import numpy as np

fs_pcc, fs_band, coh = {}, {}, {}
FENC = FK = None
for f in sorted(glob.glob("results/hest_floor/*.json")):
    d = json.load(open(f))
    FENC, FK = d["encoder"], d["k"]
    for sid, v in d["samples"].items():
        fs_pcc[sid] = v["pcc"]
        fs_band[sid] = {int(k): x for k, x in v["band_pcc"].items()}
        coh[sid] = d["cohort"]
print(f"下界 encoder={FENC}  k={FK}  样本={len(fs_pcc)}")

E = {}
for f in sorted(glob.glob("results/hest_effres_*.json")):
    if "_ps_" in f:
        continue
    d = json.load(open(f))
    if d["n_samples"] == 72:
        E[d["encoder"]] = d
ids = sorted(fs_pcc)
ts = sorted(int(k) for k in E["phikon_v2"]["band_pcc"])
sig = {t: E["phikon_v2"]["sigma_um"][str(t)] for t in ts}
FB = {t: float(np.nanmean([fs_band[s][t] for s in ids])) for t in ts}
F0 = float(np.nanmean([fs_pcc[s] for s in ids]))

M = "phikon_v2"
m0 = E[M]["pcc_check"]["mine"]
print(f"\n=== 同特征配对 ({M} ridge vs {M} kNN k={FK}) ===")
print(f"整体: ridge {m0:.4f}  vs  kNN {F0:.4f}   差 {m0-F0:+.4f} ({100*(m0-F0)/F0:+.2f}%)")
print(f"\n{'band':>5s} {'σ(µm)':>7s} {'kNN':>8s} {'ridge':>8s} {'差':>9s} {'相对':>8s}")
for t in ts:
    b = E[M]["band_pcc"][str(t)]
    print(f"{t:>5d} {sig[t]:>7.0f} {FB[t]:>8.4f} {b:>8.4f} {b-FB[t]:>+9.4f} "
          f"{100*(b-FB[t])/FB[t]:>+7.2f}%")

# 逐样本 / 逐队列配对（整体 PCC 有逐样本值，可做检验）
ref = json.load(open(f"results/hest_reported_pcc_{M}.json"))
per = {s: ref[s]["pcc"] for s in ids if s in ref}
print(f"\n逐样本整体 PCC 可配对 {len(per)}/{len(ids)}")
d_s = np.array([per[s] - fs_pcc[s] for s in per])
print(f"  ridge 高于 kNN 的样本: {int((d_s > 0).sum())}/{len(d_s)}  "
      f"中位差 {np.median(d_s):+.4f}")
cl = {}
for s in per:
    cl.setdefault(coh[s], []).append(per[s] - fs_pcc[s])
med = {c: float(np.median(v)) for c, v in cl.items()}
k = sum(1 for v in med.values() if v > 0); n = len(med)
from math import comb
Pv = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)
print(f"  队列层级: {k}/{n} 个队列 ridge 胜, 精确双侧符号检验 P = {Pv:.5f} "
      f"(n={n} 最小可得 P = {2/2**n:.5f})")
for c in sorted(med, key=lambda c: -med[c]):
    print(f"    {c:<10s} {med[c]:+.4f}  (n={len(cl[c])})")

print(f"\n=== 对照：其余 14 个编码器与同一个 {M} 下界比（混入特征差异，不可归因） ===")
for e in sorted(E, key=lambda e: -E[e]["pcc_check"]["mine"]):
    v = E[e]["pcc_check"]["mine"]
    tag = "  <-- 同特征" if e == M else ""
    print(f"  {e:<16s} 整体 {v:.4f}  vs 下界 {F0:.4f}  {v-F0:+.4f}{tag}")

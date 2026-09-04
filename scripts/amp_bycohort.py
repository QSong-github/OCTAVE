# -*- coding: utf-8 -*-
"""§40 的分队列稳健性：细带放大 1.29× 是不是被 CCRCC+PRAD（72 中占 47）带出来的。

在每个队列内部独立重算「跨编码器离散度的细带放大」，得到 10 个估计。
只用 t ≤ 256（§43：t=512/1024 是非同质带）。
"""
import json, glob
import numpy as np

E = {}
for f in sorted(glob.glob("results/hest_effres_ps_*.json")):
    d = json.load(open(f))
    E[d["encoder"]] = d
encs = sorted(E)
d0 = E[encs[0]]
ids = sorted(d0["per_sample_band_pcc"])
L = json.load(open("results/hest_ladder.json"))
coh = {s: L[s]["cohort"] for s in ids}
TS = [t for t in sorted(int(t) for t in d0["sigma_um"]) if t <= 256]
sig = {t: d0["sigma_um"][str(t)] for t in TS}
print(f"编码器 {len(encs)}；样本 {len(ids)}；用带 t≤256（σ≤{sig[TS[-1]]:.0f}µm，同质）")


def cv(v):
    v = np.asarray(v, float)
    return float(v.std(ddof=1) / v.mean())


def iqrm(v):
    q1, q3 = np.percentile(v, [25, 75])
    return float((q3 - q1) / np.median(v))


def amp(sub):
    """给定样本子集，返回 (细带 CV / 全谱 CV, 同 IQR/中位, n)"""
    o = np.array([np.mean([E[e]["per_sample_pcc"][s] for s in sub]) for e in encs])
    b = np.array([np.mean([E[e]["per_sample_band_pcc"][s][str(TS[0])] for s in sub])
                  for e in encs])
    return cv(b) / cv(o), iqrm(b) / iqrm(o), len(sub)


aC, aI, _ = amp(ids)
print(f"\n全体 72 样本: CV 放大 {aC:.2f}×   IQR/中位放大 {aI:.2f}×")

by = {}
for s in ids:
    by.setdefault(coh[s], []).append(s)
print(f"\n{'队列':<12s}{'n':>4s}{'CV放大':>9s}{'IQR放大':>9s}")
rows = []
for c in sorted(by, key=lambda c: -len(by[c])):
    if len(by[c]) < 3:
        print(f"{c:<12s}{len(by[c]):>4d}{'n<3 跳过':>18s}"); continue
    a, b, n = amp(by[c])
    rows.append((c, n, a, b))
    print(f"{c:<12s}{n:>4d}{a:>9.2f}×{b:>8.2f}×")
av = np.array([r[2] for r in rows])
print(f"\n{len(rows)} 个队列独立估计: 中位 {np.median(av):.2f}×  "
      f"范围 {av.min():.2f}–{av.max():.2f}×  >1 的 {int((av > 1).sum())}/{len(av)}")

# 留出主导队列
for drop in (["CCRCC"], ["PRAD"], ["CCRCC", "PRAD"]):
    sub = [s for s in ids if coh[s] not in drop]
    a, b, n = amp(sub)
    print(f"去掉 {'+'.join(drop):<12s} (剩 {n:>2d} 样本): CV 放大 {a:.2f}×  IQR {b:.2f}×")

# 逐带（全体样本），只到 t=256
print(f"\n{'σ(µm)':>8s}{'CV放大':>9s}{'IQR放大':>9s}")
o = np.array([np.mean([E[e]["per_sample_pcc"][s] for s in ids]) for e in encs])
for t in TS:
    b = np.array([np.mean([E[e]["per_sample_band_pcc"][s][str(t)] for s in ids])
                  for e in encs])
    print(f"{sig[t]:>8.0f}{cv(b)/cv(o):>9.2f}×{iqrm(b)/iqrm(o):>8.2f}×")

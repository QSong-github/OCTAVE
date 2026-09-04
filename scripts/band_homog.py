# -*- coding: utf-8 -*-
"""检验 §40 的前提：池化的「σ=50 µm 带」在 72 个样本上是不是同一个物理尺度。

hest_effres 把 σ 按样本取均值后报一个数。但 HEST 混了 Visium 与 Xenium，间距不同，
同一个 t 对应的 σ 可能差很多。若差很多，跨编码器比同一个 t 仍然合法（同样本集），
但把该带称作「σ=50 µm」就是错的，且跨平台的带不可解释为同一尺度。
"""
import json, glob, os
import numpy as np

d = json.load(open("results/hest_effres_ps_phikon.json"))
PSS = d["per_sample_sigma_um"]
meta = d["meta"]
ts = sorted(int(t) for t in list(PSS.values())[0])
ids = sorted(PSS)
plat = {}
L = json.load(open("results/hest_ladder.json"))
for s in ids:
    plat[s] = L[s]["platform"] if s in L else "?"
pitch = {s: meta[s].get("pitch_um") for s in ids}

print(f"样本 {len(ids)}；平台 " +
      str({p: sum(1 for s in ids if plat[s] == p) for p in set(plat.values())}))
pv = np.array([pitch[s] for s in ids], float)
print(f"间距 µm: min {pv.min():.1f}  中位 {np.median(pv):.1f}  max {pv.max():.1f}")
for p in sorted(set(plat.values())):
    q = np.array([pitch[s] for s in ids if plat[s] == p], float)
    print(f"  {p:<8s} n={len(q):>3d}  间距 {q.min():.1f}–{q.max():.1f} (中位 {np.median(q):.1f})")

print(f"\n{'t':>5s}{'池化σ':>9s}{'样本σ min':>11s}{'中位':>9s}{'max':>9s}{'max/min':>9s}"
      f"{'Visium中位':>11s}{'Xenium中位':>11s}")
BAD = []
for t in ts:
    v = np.array([PSS[s][str(t)] for s in ids])
    vi = np.array([PSS[s][str(t)] for s in ids if plat[s] == "Visium"])
    xe = np.array([PSS[s][str(t)] for s in ids if plat[s] == "Xenium"])
    r = v.max() / v.min()
    if r > 1.5:
        BAD.append((t, r))
    print(f"{t:>5d}{v.mean():>9.1f}{v.min():>11.1f}{np.median(v):>9.1f}{v.max():>9.1f}"
          f"{r:>9.2f}{np.median(vi):>11.1f}{np.median(xe):>11.1f}")

print(f"\n带中心跨样本离散超过 1.5× 的档: {len(BAD)}/{len(ts)}")
if BAD:
    print("  " + ", ".join(f"t={t}({r:.2f}×)" for t, r in BAD))
t0 = ts[0]
v0 = np.array([PSS[s][str(t0)] for s in ids])
vi = np.array([PSS[s][str(t0)] for s in ids if plat[s] == "Visium"])
xe = np.array([PSS[s][str(t0)] for s in ids if plat[s] == "Xenium"])
print(f"\n最细带 t=1: 报作 σ={v0.mean():.1f}µm，实际逐样本 {v0.min():.1f}–{v0.max():.1f}µm")
print(f"  Visium (n={len(vi)}): {vi.min():.1f}–{vi.max():.1f}  中位 {np.median(vi):.1f}")
print(f"  Xenium (n={len(xe)}): {xe.min():.1f}–{xe.max():.1f}  中位 {np.median(xe):.1f}")
print(f"  两平台中位之比 {np.median(vi)/np.median(xe):.2f}×")

# 跨编码器比较是否仍合法：所有编码器在同一样本上的 σ 必须完全相同
E = {}
for f in sorted(glob.glob("results/hest_effres_ps_*.json")):
    dd = json.load(open(f)); E[dd["encoder"]] = dd["per_sample_sigma_um"]
mx = 0.0
for s in ids:
    for t in ts:
        vals = [E[e][s][str(t)] for e in E if s in E[e]]
        mx = max(mx, max(vals) - min(vals))
print(f"\n同一样本同一 t 上，{len(E)} 个编码器的 σ 最大差异 = {mx:.3e}"
      f"  ⇒ 跨编码器比同一 t {'合法' if mx < 1e-9 else '不合法'}（σ 只依赖坐标）")

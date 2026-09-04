# -*- coding: utf-8 -*-
"""k 是移动水平还是重排名次？——与本文「归一化旋钮移动水平不移动名次」同一口径。"""
import json, itertools
import numpy as np
from scipy.stats import spearmanr

D = json.load(open("results/k_sensitivity.json"))
ks = sorted(int(k) for k in D)
common = sorted(set.intersection(*[set(D[str(k)]) for k in ks]))
print(f"{len(common)} 个编码器在 k = {ks} 上都完整\n")


def val(k, e):
    r = D[str(k)][e]
    return r["rel_ok"] if r["rel_ok"] is not None else r["rel"]


V = {k: np.array([val(k, e) for e in common]) for k in ks}
print("=== 名次相关（Spearman ρ）")
print("       " + "".join(f"{('k=%d' % k):>9s}" for k in ks))
for a in ks:
    row = f"k={a:<4d} "
    for b in ks:
        row += f"{spearmanr(V[a], V[b]).statistic:>9.3f}"
    print(row)
rho = [spearmanr(V[a], V[b]).statistic for a, b in itertools.combinations(ks, 2)]
print(f"\n所有 {len(rho)} 对 k 的 ρ: 中位 {np.median(rho):.3f}  最小 {min(rho):.3f}")

print("\n=== 水平位移 vs 名次变动")
base = ks[0]
for k in ks[1:]:
    d = V[k] - V[base]
    print(f"k={base}→{k}: 位移中位 {np.median(d):+6.2f} pp  "
          f"（范围 {d.min():+.2f}…{d.max():+.2f}，即几乎整体平移）  "
          f"ρ={spearmanr(V[base], V[k]).statistic:.3f}")

print("\n=== 名次变动幅度")
R = {k: np.argsort(np.argsort(-V[k])) + 1 for k in ks}
shift = np.array([max(R[k][i] for k in ks) - min(R[k][i] for k in ks)
                  for i in range(len(common))])
print(f"{'编码器':<16s}" + "".join(f"{('k=%d' % k):>8s}" for k in ks) + f"{'名次跨度':>9s}")
for i, e in sorted(enumerate(common), key=lambda z: R[ks[0]][z[0]]):
    print(f"{e:<16s}" + "".join(f"{R[k][i]:>8d}" for k in ks) + f"{shift[i]:>9d}")
print(f"\n名次最大变动 {shift.max()} 位，中位 {np.median(shift):.1f} 位")
print(f"⇒ k 主要{'移动水平' if np.median(rho) > 0.8 else '既移水平也重排'}"
      f"（ρ 中位 {np.median(rho):.3f}，名次中位变动 {np.median(shift):.1f} 位）")

print("\n=== k=10 下达到队列层级显著的编码器")
for e in common:
    r = D["10"][e]
    if r["P"] < 0.05:
        print(f"  {e:<16s} 相对差 {val(10,e):+6.2f}%  队列胜 {r['n_cohorts_win']}/10  "
              f"P={r['P']:.5f}")

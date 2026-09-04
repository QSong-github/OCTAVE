# -*- coding: utf-8 -*-
"""§46 的 k 敏感性:「7/15 编码器低于自身检索下界、0/15 显著」是否依赖 k=50。

只改 kNN 的 k，其余(特征、PCA-256 管线、划分、50 基因、log1p 目标)完全不变。
检验单元一律是队列(n=10，最小可得 P=0.00195)。
"""
import json, glob, os
import numpy as np
from math import comb


def signp(k, n):
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


# 编码器表由 results/ 自动发现（2026-08-29 从 15 扩到 27）。
# omiclip_raw 是 omiclip 的未归一化变体，同一模型，排除以免在榜单里占两行。
import glob as _g, os as _o
ENC = sorted({_o.path.basename(f)[len("results/hest_rsel_ps_") - len("results/"):-5]
              for f in _g.glob("results/hest_rsel_ps_*.json")} - {"omiclip_raw"})
PS = {}
for e in ENC:
    f = f"results/hest_rsel_ps_{e}.json"
    if os.path.exists(f):
        PS[e] = json.load(open(f))
L = json.load(open("results/hest_ladder.json"))

# 每个队列的留一训练集规模 → 邻域占比。占比过大时 kNN 退化为近似全局均值，
# 下界被人为削弱，编码器会“轻松打赢”，这是假象不是发现。
NTR = {}
for f in sorted(glob.glob("results/hest_floor/*.json")):
    dd = json.load(open(f))
    ns = [v["n"] for v in dd["samples"].values()]
    NTR[dd["cohort"]] = max(sum(ns) - int(np.median(ns)), 1)
FRAC_MAX = 0.05          # 邻域超过训练集 5% 即视为退化


def degenerate(c, K):
    return min(K, NTR[c]) / NTR[c] > FRAC_MAX

KDIRS = {50: "results/hest_floor_{e}"}
for K in (10, 200, 800):
    KDIRS[K] = "results/hest_floor_k%d_{e}" % K

OUT = {}
print("退化判据: 邻域 k 超过该队列留一训练集的 %.0f%% 即视为退化"
      "（检索塌成近似全局均值，下界被人为削弱）" % (100 * FRAC_MAX))
for K in sorted(KDIRS):
    bad = [c for c in NTR if degenerate(c, K)]
    print(f"  k={K:<4d} 退化队列 {len(bad):>2d}/10" + (f": {bad}" if bad else ""))
print()
print(f"{'k':>5s} {'编码器':>7s} {'低于下界':>10s} {'显著':>8s} {'队列胜中位':>11s} "
      f"{'相对差中位':>11s} | {'非退化队列':>10s} {'低于下界':>10s} {'显著':>8s}")
for K in sorted(KDIRS):
    rows = {}
    for e in ENC:
        d = KDIRS[K].format(e=e)
        fs = sorted(glob.glob(f"{d}/*.json"))
        if len(fs) != 10 or e not in PS:
            continue
        fl, coh = {}, {}
        for f in fs:
            dd = json.load(open(f))
            for s, v in dd["samples"].items():
                fl[s] = v["pcc"]; coh[s] = dd["cohort"]
        ids = sorted(set(fl) & set(PS[e]["per_sample_pcc"]))
        if len(ids) != 72:
            continue
        dif = {s: PS[e]["per_sample_pcc"][s] - fl[s] for s in ids}
        by = {}
        for s in ids:
            by.setdefault(coh[s], []).append(dif[s])
        med = {c: float(np.median(v)) for c, v in by.items()}
        kw = sum(1 for v in med.values() if v > 0)
        f0 = float(np.mean([fl[s] for s in ids]))
        g = float(np.mean(list(dif.values())))
        # 仅用非退化队列重算（下界未被人为削弱的那些）
        okc = [c for c in med if not degenerate(c, K)]
        ids_ok = [s for s in ids if not degenerate(coh[s], K)]
        kw_ok = sum(1 for c in okc if med[c] > 0)
        g_ok = float(np.mean([dif[s] for s in ids_ok])) if ids_ok else float("nan")
        f_ok = float(np.mean([fl[s] for s in ids_ok])) if ids_ok else float("nan")
        rows[e] = {"gap": g, "rel": 100 * g / f0, "floor": f0,
                   "n_cohorts_win": kw, "P": signp(kw, len(med)),
                   "n_cohorts_ok": len(okc), "n_cohorts_win_ok": kw_ok,
                   "P_ok": signp(kw_ok, len(okc)) if okc else None,
                   "rel_ok": 100 * g_ok / f_ok if okc else None}
    if not rows:
        print(f"{K:>5d} {'(尚无完整结果)':>10s}")
        continue
    OUT[K] = rows
    below = sum(1 for r in rows.values() if r["gap"] < 0)
    sig = sum(1 for r in rows.values() if r["P"] < 0.05)
    best = max(rows, key=lambda e: rows[e]["rel"])
    blab = "%s (%+.1f%%)" % (best, rows[best]["rel"])
    cwm = float(np.median([r["n_cohorts_win"] for r in rows.values()]))
    rlm = float(np.median([r["rel"] for r in rows.values()]))
    nok = list(rows.values())[0]["n_cohorts_ok"]
    below_ok = sum(1 for r in rows.values()
                   if r["rel_ok"] is not None and r["rel_ok"] < 0)
    sig_ok = sum(1 for r in rows.values() if r["P_ok"] is not None and r["P_ok"] < 0.05)
    print(f"{K:>5d} {len(rows):>7d} {below:>7d}/{len(rows)} {sig:>5d}/{len(rows)} "
          f"{cwm:>11.1f} {rlm:>10.2f}% | {nok:>8d}/10 {below_ok:>7d}/{len(rows)} "
          f"{sig_ok:>5d}/{len(rows)}   最佳 {blab}")

if len(OUT) > 1:
    ks = sorted(OUT)
    common = set.intersection(*[set(OUT[k]) for k in ks])
    print(f"\n=== 逐编码器：相对差(%)【仅非退化队列】随 k 的变化（{len(common)} 个编码器在所有 k 上都完整）")
    print(f"{'编码器':<16s}" + "".join(f"{('k=%d' % k):>10s}" for k in ks) + f"{'符号一致':>10s}")
    flip = []
    for e in sorted(common, key=lambda e: -OUT[ks[0]][e]["rel"]):
        v = [(OUT[k][e]["rel_ok"] if OUT[k][e]["rel_ok"] is not None
              else OUT[k][e]["rel"]) for k in ks]
        same = all(x > 0 for x in v) or all(x < 0 for x in v)
        if not same:
            flip.append(e)
        print(f"{e:<16s}" + "".join(f"{x:>9.2f}%" for x in v) +
              f"{('是' if same else '**否**'):>10s}")
    print(f"\n符号随 k 翻转的编码器: {len(flip)}/{len(common)} {flip}")
    if len(common) < 10:
        print(f"⇒ [不下结论] 只有 {len(common)}/15 个编码器在所有 k 上完整，"
              "样本太少，等跑齐再判。")
    elif flip:
        print(f"⇒ 结论对 k **不完全稳健**，{len(flip)}/{len(common)} 个编码器的符号会翻")
    else:
        print("⇒ 符号对 k 稳健（但幅度仍随 k 变，见上表）")
    # 幅度敏感性：k 的影响 vs 编码器之间的全部差距
    # 只有编码器够多时才算 —— 否则跨度近零，比值会炸（本项目已四次踩到近零分母）。
    if len(common) < 10:
        raise SystemExit(0)
    for kk in ks:
        v = [(OUT[kk][e]["rel_ok"] if OUT[kk][e]["rel_ok"] is not None
              else OUT[kk][e]["rel"]) for e in common]
        print(f"   k={kk:<4d} 编码器间跨度 {max(v)-min(v):>6.1f} pp "
              f"({min(v):+.1f}% … {max(v):+.1f}%)")
    if len(ks) > 1:
        shifts = []
        for e in common:
            v = [(OUT[k][e]["rel_ok"] if OUT[k][e]["rel_ok"] is not None
                  else OUT[k][e]["rel"]) for k in ks]
            shifts.append(max(v) - min(v))
        v0 = [(OUT[ks[0]][e]["rel_ok"] if OUT[ks[0]][e]["rel_ok"] is not None
               else OUT[ks[0]][e]["rel"]) for e in common]
        span = max(v0) - min(v0)
        print(f"   同一编码器因 k 而移动: 中位 {np.median(shifts):.1f} pp "
              f"(范围 {min(shifts):.1f}–{max(shifts):.1f})")
        if span > 5.0:
            print(f"   ⇒ 单个超参 k 的影响 ≈ 编码器间全部差距的 "
                  f"{100*np.median(shifts)/span:.0f}%")
        else:
            print(f"   ⇒ 编码器间跨度仅 {span:.1f} pp，太小，不报比值")
    json.dump({str(k): OUT[k] for k in OUT}, open("results/k_sensitivity_rsel.json", "w"), indent=1)
    print("\n已存 results/k_sensitivity_rsel.json")

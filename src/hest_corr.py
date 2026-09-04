# -*- coding: utf-8 -*-
"""
Systema Fig 3c 的对应分析 —— 报告 PCC 到底被什么绑架。

Systema: 数据集层面的"系统性变异"与 PearsonΔ 相关 0.91–0.95, 换参考点后掉到 0.21。
本文: 72 个样本(10 癌种, 2 平台)上, 把官方协议报告的 PCC 分别与两个【与模型无关】的
      数据属性求相关:
        (a) A_coarse(σ) —— 该样本的信号有多少是粗尺度解剖结构
        (b) c_raw       —— 该样本的噪声天花板(拆半可重复性, SB 校正)
并给出每样本的【等价 σ】: 报告 PCC 在该样本自己的 A_coarse 阶梯上插值, 读成
"相当于知道真实表达场被模糊到多少微米"。

输入: results/hest_ladder.json(模型无关) + results/hest_reported_pcc.json(官方协议)
"""
import json, numpy as np
from scipy.stats import pearsonr, spearmanr

import sys
ENC = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
R = "/blue/qsong1/wang.qing/systema4ST/results"
L = json.load(open(f"{R}/hest_ladder.json"))
P = json.load(open(f"{R}/hest_reported_pcc_{ENC}.json" if ENC != "resnet50"
                   else f"{R}/hest_reported_pcc.json"))
print(f"编码器: {ENC}")
sids = sorted(set(L) & set(P))
print(f"配对样本 {len(sids)}")

cps = sorted(int(k) for k in L[sids[0]]["ladder"])
sig_all = {c: np.mean([L[s]["sigma_um"][str(c)] for s in sids]) for c in cps}


def eq_sigma(sid, val):
    """在该样本自己的阶梯上把 PCC 插值成等价 σ(µm)。阶梯随 σ 单调下降。"""
    d = L[sid]
    pts = sorted(((d["sigma_um"][str(c)], d["ladder"][str(c)]) for c in cps), key=lambda t: t[0])
    if val >= pts[0][1]:
        return pts[0][0]                      # 优于最细档 → 截断
    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
        if v0 >= val >= v1:
            return s0 + (v0 - val) / max(v0 - v1, 1e-12) * (s1 - s0)
    return np.nan                             # 比最粗档还差


rows = []
for s in sids:
    d, p = L[s], P[s]["pcc"]
    rows.append(dict(sid=s, cohort=d["cohort"], plat=d["platform"], n=d["n"],
                     zero=d["zero_frac"], c_raw=d["c_raw"], pcc=p,
                     norm=p / d["c_raw"] if d["c_raw"] > 0.05 else np.nan,
                     eqs=eq_sigma(s, p),
                     **{f"A{int(sig_all[c])}": d["ladder"][str(c)] for c in cps}))

get = lambda k: np.array([r[k] for r in rows], float)
pcc = get("pcc")

print(f"\n=== 报告 PCC 与【模型无关的数据属性】的相关 (n={len(rows)}) ===")
print(f"{'数据属性':28s}{'Pearson r':>11s}{'p':>10s}{'Spearman':>10s}")
cands = [("噪声天花板 c_raw", "c_raw"), ("零元比例", "zero"), ("spot 数", "n")]
cands += [(f"A_coarse(σ≈{int(sig_all[c])}µm)", f"A{int(sig_all[c])}") for c in cps]
for nm, k in cands:
    v = get(k)
    m = np.isfinite(v) & np.isfinite(pcc)
    r, pv = pearsonr(v[m], pcc[m]); sr, _ = spearmanr(v[m], pcc[m])
    print(f"{nm:28s}{r:>11.3f}{pv:>10.2e}{sr:>10.3f}")

print(f"\n=== 平台分层 ===")
print(f"{'平台':10s}{'n':>4s}{'报告PCC':>10s}{'c_raw':>9s}{'PCC/c_raw':>11s}{'等价σ(µm)':>11s}")
for pl in sorted(set(r["plat"] for r in rows)):
    sub = [r for r in rows if r["plat"] == pl]
    print(f"{pl:10s}{len(sub):>4d}{np.mean([r['pcc'] for r in sub]):>10.3f}"
          f"{np.mean([r['c_raw'] for r in sub]):>9.3f}"
          f"{np.nanmean([r['norm'] for r in sub]):>11.3f}"
          f"{np.nanmedian([r['eqs'] for r in sub]):>11.0f}")

print(f"\n=== 队列层面 (按报告 PCC 排序) ===")
print(f"{'队列':11s}{'平台':9s}{'n':>4s}{'报告PCC':>10s}{'c_raw':>9s}{'PCC/c_raw':>11s}{'等价σ':>9s}")
coh = {}
for r in rows:
    coh.setdefault(r["cohort"], []).append(r)
for c, sub in sorted(coh.items(), key=lambda kv: -np.mean([r["pcc"] for r in kv[1]])):
    print(f"{c:11s}{sub[0]['plat']:9s}{len(sub):>4d}"
          f"{np.mean([r['pcc'] for r in sub]):>10.3f}{np.mean([r['c_raw'] for r in sub]):>9.3f}"
          f"{np.nanmean([r['norm'] for r in sub]):>11.3f}"
          f"{np.nanmedian([r['eqs'] for r in sub]):>9.0f}")

# 排名是否翻转: 原始 PCC 排序 vs 归一化后排序
o1 = [c for c, _ in sorted(coh.items(), key=lambda kv: -np.mean([r["pcc"] for r in kv[1]]))]
o2 = [c for c, _ in sorted(coh.items(), key=lambda kv: -np.nanmean([r["norm"] for r in kv[1]]))]
print(f"\n原始 PCC 队列排名 : {' > '.join(o1)}")
print(f"扣噪声天花板后   : {' > '.join(o2)}")
sr, _ = spearmanr([o1.index(c) for c in o1], [o2.index(c) for c in o1])
print(f"两种排序的 Spearman = {sr:.3f}  {'⇒ 排名显著重排' if sr < 0.8 else ''}")

# ---- 平台内相关: 排除"整条相关性只是平台差异"的可能
print(f"\n=== 平台内相关(控制平台后是否仍成立) ===")
print(f"{'平台':10s}{'n':>4s}{'PCC~c_raw':>12s}{'PCC~A_86µm':>12s}")
for pl in sorted(set(r["plat"] for r in rows)):
    sub = [r for r in rows if r["plat"] == pl]
    if len(sub) < 5: continue
    a = np.array([r["pcc"] for r in sub]); b = np.array([r["c_raw"] for r in sub])
    k86 = [k for k in sub[0] if k.startswith("A") and k[1:].isdigit()][1]
    d = np.array([r[k86] for r in sub])
    print(f"{pl:10s}{len(sub):>4d}{pearsonr(b,a)[0]:>12.3f}{pearsonr(d,a)[0]:>12.3f}")

# ---- 偏相关: c_raw 与 A_coarse 互相控制后各自还剩多少
def partial(x, y, z):
    """r(x,y|z): 各自对 z 回归取残差再求相关。"""
    Z = np.stack([z, np.ones_like(z)], 1)
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return pearsonr(rx, ry)

kA = [k for k in rows[0] if k.startswith("A") and k[1:].isdigit()][1]
cr, Ac = get("c_raw"), get(kA)
print(f"\n=== 偏相关 (c_raw 与 {kA} 彼此相关 r={pearsonr(cr,Ac)[0]:.3f}) ===")
r1, p1 = partial(cr, pcc, Ac); r2, p2 = partial(Ac, pcc, cr)
print(f"  PCC ~ c_raw | 控制 {kA}   : r={r1:.3f} p={p1:.2e}")
print(f"  PCC ~ {kA} | 控制 c_raw   : r={r2:.3f} p={p2:.2e}")
X = np.stack([cr, Ac, np.ones_like(cr)], 1)
beta, res, *_ = np.linalg.lstsq(X, pcc, rcond=None)
r2tot = 1 - ((pcc - X @ beta) ** 2).sum() / ((pcc - pcc.mean()) ** 2).sum()
print(f"  二元回归 R² = {r2tot:.3f}  (β_c_raw={beta[0]:.3f}, β_A={beta[1]:.3f})")

eqs = get("eqs")
print(f"\n=== 等价 σ 分布 (n={np.isfinite(eqs).sum()}) ===")
print(f"  中位={np.nanmedian(eqs):.0f}µm  四分位=[{np.nanpercentile(eqs,25):.0f},"
      f"{np.nanpercentile(eqs,75):.0f}]µm  范围=[{np.nanmin(eqs):.0f},{np.nanmax(eqs):.0f}]µm")
print(f"  低于最粗档(σ>{max(sig_all.values()):.0f}µm)的样本: {int((~np.isfinite(eqs)).sum())}")
json.dump(rows, open("/blue/qsong1/wang.qing/systema4ST/results/hest_corr.json", "w"),
          indent=2, ensure_ascii=False, default=float)
print("\n已存 results/hest_corr.json")

# -*- coding: utf-8 -*-
"""对称协议：模型与地板用同一条留一队列规则各自选超参，再比较。
四个臂（全部用同一批 72 样本、10 队列、50 基因、log1p 目标、同一 PCA-256 输入）：
  A 官方 ridge（α=100/(D·G)）    vs 地板 k=50            —— 论文现状
  B 留一选 α 的 ridge            vs 地板 k=50            —— 只给模型调，不对称
  C 官方 ridge                   vs 留一选 k 的地板       —— 只给地板调，不对称
  D 留一选 α 的 ridge            vs 留一选 k 的地板       —— 对称，两边同规则
留一规则：对被留出的队列 c，用其余 9 个队列的样本均值挑该侧的超参，再把它用在 c 上。
两侧都不看被留出的队列，也不互相看。
队列统计量 = 队列内逐样本配对差的中位数（与 cohort_spread.py 同口径）；
报队列层级的均值±sem、|t|、赢的队列数与精确符号检验 P。"""
import json, glob, os, sys, numpy as np
from math import comb
R = "/path/to/systema4ST/results"
KS = [10, 50, 200, 800]
ALPHAS = ["0.1", "1", "10", "100", "1000", "10000", "100000", "1000000", "10000000", "100000000", "1000000000"]
signp = lambda k, n: min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)

def floor_at(e, K):
    d = f"{R}/hest_floor_{e}" if K == 50 else f"{R}/hest_floor_k{K}_{e}"
    out, coh = {}, {}
    for f in glob.glob(f"{d}/*.json"):
        dd = json.load(open(f))
        for s, v in dd["samples"].items():
            out[s] = v["pcc"]; coh[s] = dd["cohort"]
    return out, coh

def loco(cands, ids, coh, pick_key):
    """对每个队列，用其余队列的样本均值选出 cands 中的键；返回逐样本取值与逐队列选中的键。"""
    sel, out = {}, {}
    for c in sorted(set(coh[s] for s in ids)):
        others = [s for s in ids if coh[s] != c]
        best = max(cands, key=lambda kk: float(np.mean([cands[kk][s] for s in others])))
        sel[c] = pick_key(best)
        for s in ids:
            if coh[s] == c: out[s] = cands[best][s]
    return out, sel

ENC = sys.argv[1].split(",")
ARMS = {"A": "官方ridge vs k=50", "B": "选α ridge vs k=50", "C": "官方ridge vs 选k地板", "D": "选α ridge vs 选k地板", "E": "MLP λ=100 vs 选k地板"}
RES = {a: {} for a in ARMS}
for e in ENC:
    off = json.load(open(f"{R}/hest_effres_ps_{e}.json"))["per_sample_pcc"]
    fl50, coh = floor_at(e, 50)
    ids = sorted(set(off) & set(fl50))
    assert len(ids) == 72, (e, len(ids))
    # 模型侧候选：官方 + α 网格
    M = {"official": off}
    for a in ALPHAS:
        f = f"{R}/ridge_alpha/hest_ra_{e}_a{a}.json"
        if os.path.exists(f): M[a] = json.load(open(f))["per_sample_pcc"]
    # 地板侧候选：四档 k
    F = {}
    for K in KS:
        d, _ = floor_at(e, K)
        if len(set(d) & set(ids)) == len(ids): F[K] = d
    m_sel, a_by_c = loco(M, ids, coh, lambda x: x)
    f_sel, k_by_c = loco(F, ids, coh, lambda x: int(x))
    mp = f"{R}/hest_mlp100_ps_{e}.json"
    arms = {"A": (off, fl50), "B": (m_sel, fl50), "C": (off, f_sel), "D": (m_sel, f_sel)}
    if os.path.exists(mp): arms["E"] = (json.load(open(mp))["per_sample_pcc"], f_sel)
    for arm, (mm, ff) in arms.items():
        by = {}
        for s in ids: by.setdefault(coh[s], []).append(mm[s] - ff[s])
        g = np.array([np.median(v) for _, v in sorted(by.items())])
        sem = float(g.std(ddof=1) / np.sqrt(len(g))); won = int((g > 0).sum())
        f0 = float(np.mean([ff[s] for s in ids])); m0 = float(np.mean([mm[s] for s in ids]))
        RES[arm][e] = dict(model=m0, floor=f0, rel=100 * (m0 - f0) / f0, cohort_mean=float(g.mean()),
                           cohort_sem=sem, t=float(g.mean() / sem), won=won, P=signp(won, len(g)),
                           alpha_by_cohort=a_by_c, k_by_cohort=k_by_c)
print("%-14s %8s %8s %8s | %s" % ("编码器", "模型", "地板", "余量%", "  ".join("%-22s" % ARMS[a] for a in "ABCD")))
for e in sorted(ENC, key=lambda e: -RES["D"][e]["rel"]):
    r = RES["D"][e]
    if any(e not in RES[a] for a in "ABCD"): continue
    print("%-14s %8.4f %8.4f %+8.1f | %s" % (e, r["model"], r["floor"], r["rel"],
          "  ".join("%+7.4f±%.4f t=%4.2f %2d/10" % (RES[a][e]["cohort_mean"], RES[a][e]["cohort_sem"], abs(RES[a][e]["t"]), RES[a][e]["won"]) for a in "ABCD")))
print("\n%-26s %10s %10s %10s %10s %s" % ("臂", "低于地板", "|t|≥2(正)", "符号检验", "余量中位%", "地板均值"))
for a in [x for x in "ABCDE" if RES.get(x)]:
    d = RES[a]; n = len(d)
    print("%-26s %6d/%-3d %6d/%-3d %6d/%-3d %9.1f%% %9.4f" % (
        ARMS[a], sum(1 for e in d if d[e]["rel"] < 0), n,
        sum(1 for e in d if abs(d[e]["t"]) >= 2 and d[e]["cohort_mean"] > 0), n,
        sum(1 for e in d if d[e]["P"] < 0.05 and d[e]["cohort_mean"] > 0), n,
        float(np.median([d[e]["rel"] for e in d])), float(np.mean([d[e]["floor"] for e in d]))))
kc = {}
for e in ENC:
    for c, k in RES["D"][e]["k_by_cohort"].items(): kc[k] = kc.get(k, 0) + 1
ac = {}
for e in ENC:
    for c, a in RES["D"][e]["alpha_by_cohort"].items(): ac[a] = ac.get(a, 0) + 1
print("\n地板选中的 k（30 编码器 × 10 队列）:", dict(sorted(kc.items())))
print("模型选中的 α:", dict(sorted(ac.items(), key=lambda x: float(x[0]) if x[0] != "official" else -1)))
print("官方 α = 100/(256*50) =", 100/(256*50))
json.dump(RES, open(f"{R}/symmetric_protocol.json", "w"), indent=1)
print("已存 results/symmetric_protocol.json")

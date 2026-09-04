# -*- coding: utf-8 -*-
"""Xenium 块识别上界的样本层级聚合与检验。

16 个区域来自 8 个独立样本（乳腺 S1–S4 各含 Top/Mid/Bot）。按区域计数是伪重复。
主检验一律在样本层级：n=8 → 最小可得 P=0.0078；与 Fig 3 口径一致的 15 区域 /
7 样本子集（去掉 Prime Cervical）→ 0.0156。
"""
import json, glob, re, os
import numpy as np
from math import comb

D = "results/blocks_xen"


def specimen(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    return m.group(1) if m else n


def signp(k, n):
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


F = sorted(glob.glob(f"{D}/*.json"))
R = [json.load(open(f)) for f in F]
print(f"区域 {len(R)}；独立样本 {len({specimen(r['name']) for r in R})}")
if len(R) < 4:
    print("[停] 产出太少，等作业完成。"); raise SystemExit(0)

for K in ("dom20", "dom200"):
    print(f"\n{'='*78}\n=== 纯块预测器 {K} vs 训练 ridge（片内 16×16 块 CV，同协议）\n{'='*78}")
    print(f"{'区域':<44s}{'ridge':>7s}{'blocks':>8s}{'比':>7s}"
          f"{'σr':>7s}{'σb':>7s}{'整体差':>8s}{'细带差':>8s}{'比':>7s}")
    rows = []
    for r in R:
        p = r["pred"]
        if K not in p:
            continue
        g = r["rel_gap"][K]
        sr, sb = p["ridge"]["sigma_um"], p[K]["sigma_um"]
        rows.append({"name": r["name"], "spec": specimen(r["name"]),
                     "pcc_r": p["ridge"]["pcc"], "pcc_b": p[K]["pcc"],
                     "band_r": p["ridge"]["band_pcc"], "band_b": p[K]["band_pcc"],
                     "sig_r": sr, "sig_b": sb,
                     "go": g["overall"], "gf": g["fineband"],
                     "ratio": (g["fineband"] / g["overall"]) if g["overall"] else np.nan,
                     "score_ratio": g["score_ratio"], "sigma_ratio": g["sigma_ratio"]})
        w = rows[-1]
        print(f"{w['name'][:43]:<44s}{w['pcc_r']:>7.3f}{w['pcc_b']:>8.3f}"
              f"{100*w['score_ratio']:>6.0f}%"
              f"{(sr or float('nan')):>7.0f}{(sb or float('nan')):>7.0f}"
              f"{100*w['go']:>7.1f}%{100*w['gf']:>7.1f}%{w['ratio']:>7.2f}")

    def spec_med(key):
        by = {}
        for w in rows:
            by.setdefault(w["spec"], []).append(w[key])
        return {s: float(np.median(v)) for s, v in by.items()}

    print(f"\n  区域层级（伪重复，仅描述）: 分数比中位 "
          f"{100*np.median([w['score_ratio'] for w in rows]):.1f}%  "
          f"细带/整体差距比中位 {np.median([w['ratio'] for w in rows]):.2f}×")

    for tag, keep in (("全部 8 样本", None),
                      ("Fig 3 口径 7 样本", "Xenium_Prime_Cervical")):
        rr = [w for w in rows if not (keep and keep in w["name"])]
        if not rr:
            continue
        by = {}
        for w in rr:
            by.setdefault(w["spec"], []).append(w)
        n = len(by)
        # 主检验：细带相对差距 > 整体相对差距
        k = sum(1 for s, v in by.items()
                if np.median([x["gf"] for x in v]) > np.median([x["go"] for x in v]))
        P = signp(k, n)
        # 副检验：分数比 > 80%（纯块拿到大部分分数）
        k2 = sum(1 for s, v in by.items()
                 if np.median([x["score_ratio"] for x in v]) > 0.80)
        P2 = signp(k2, n)
        rt = [float(np.median([x["ratio"] for x in v])) for v in by.values()]
        sr = [float(np.median([x["score_ratio"] for x in v])) for v in by.values()]
        print(f"\n  【{tag}】n={n}，最小可得 P={2/2**n:.5f}")
        print(f"    细带相对差距 > 整体相对差距: {k}/{n}  P = {P:.5f}")
        print(f"      样本层级比值中位 {np.median(rt):.2f}×  范围 {min(rt):.2f}–{max(rt):.2f}×")
        print(f"    纯块拿到 >80% 的分数:        {k2}/{n}  P = {P2:.5f}")
        print(f"      样本层级分数比中位 {100*np.median(sr):.1f}%  "
              f"范围 {100*min(sr):.1f}–{100*max(sr):.1f}%")

    # 细带方差占比：两者是否都远低于真值
    tv = [r["pred"]["truth"]["fine_var_share"] for r in R]
    rv = [r["pred"]["ridge"]["fine_var_share"] for r in R]
    bv = [r["pred"][K]["fine_var_share"] for r in R if K in r["pred"]]
    print(f"\n  细带方差占比中位: 真值 {100*np.median(tv):.1f}%  "
          f"ridge {100*np.median(rv):.1f}%  {K} {100*np.median(bv):.1f}%  "
          f"⇒ 过度平滑 {np.median(tv)/np.median(rv):.1f}× / "
          f"{np.median(tv)/np.median(bv):.1f}×")

json.dump({"n_regions": len(R)}, open("results/blocks_xen_summary.json", "w"), indent=1)

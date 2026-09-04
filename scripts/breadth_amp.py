# -*- coding: utf-8 -*-
"""广度线（HEST 72 样本 × 15 编码器）：细带里编码器之间的差距是否被放大。

带 PCC 在同一 t 上跨编码器可比（同样本、同图、同带），σ 不可比 —— 故全程只用带 PCC。
"""
import json, glob, os, itertools
import numpy as np

F = sorted(glob.glob("results/hest_effres_*.json"))
E, BAD = {}, []
for f in F:
    d = json.load(open(f))
    enc = d["encoder"]
    if d["n_samples"] != 72:
        BAD.append((enc, d["n_samples"])); continue
    E[enc] = d
print(f"编码器 {len(E)} 个，均 72 样本；剔除 {BAD}")

ts = sorted((int(k) for k in list(E.values())[0]["band_pcc"]))
sig = {t: float(np.mean([E[e]["sigma_um"][str(t)] for e in E])) for t in ts}
sd = {t: float(np.std([E[e]["sigma_um"][str(t)] for e in E])) for t in ts}
print("σ 阶梯（跨编码器完全一致，因为只依赖坐标）: "
      + " ".join(f"t{t}={sig[t]:.0f}±{sd[t]:.2g}" for t in ts[:4]))

overall = np.array([E[e]["pcc_check"]["mine"] for e in E])
encs = list(E)
print(f"\n整体 PCC: 均值 {overall.mean():.4f}  min {overall.min():.4f} "
      f"({encs[int(overall.argmin())]})  max {overall.max():.4f} "
      f"({encs[int(overall.argmax())]})")


def stats(v):
    v = np.asarray(v, float)
    q1, q3 = np.percentile(v, [25, 75])
    return {"mean": v.mean(), "cv": v.std(ddof=1) / v.mean(),
            "iqr_over_med": (q3 - q1) / np.median(v),
            "span": v.max() - v.min(), "rel_span": (v.max() - v.min()) / v.mean()}


print(f"\n{'band':>6s} {'σ(µm)':>8s} {'mean':>8s} {'CV':>8s} {'IQR/med':>9s} "
      f"{'CV amp':>8s} {'IQR amp':>8s}")
S0 = stats(overall)
ROWS = {}
for t in ts:
    b = np.array([E[e]["band_pcc"][str(t)] for e in E])
    s = stats(b)
    ROWS[t] = s
    print(f"{t:>6d} {sig[t]:>8.0f} {s['mean']:>8.4f} {s['cv']:>8.4f} "
          f"{s['iqr_over_med']:>9.4f} {s['cv']/S0['cv']:>8.2f}x "
          f"{s['iqr_over_med']/S0['iqr_over_med']:>7.2f}x")
print(f"{'全谱':>6s} {'--':>8s} {S0['mean']:>8.4f} {S0['cv']:>8.4f} "
      f"{S0['iqr_over_med']:>9.4f} {'1.00x':>8s} {'1.00x':>8s}")

t0 = ts[0]
b0 = np.array([E[e]["band_pcc"][str(t0)] for e in E])
amp_cv = ROWS[t0]["cv"] / S0["cv"]
amp_iq = ROWS[t0]["iqr_over_med"] / S0["iqr_over_med"]
print(f"\n最细带 (σ={sig[t0]:.0f}µm) 放大: CV {amp_cv:.2f}×   IQR/med {amp_iq:.2f}×")

# 留一：放大倍数是否由某个编码器独扛
print("\n留一稳健性（去掉每个编码器后的 CV 放大倍数）:")
loo = []
for i, e in enumerate(encs):
    keep = [j for j in range(len(encs)) if j != i]
    a = stats(overall[keep]); b = stats(b0[keep])
    loo.append(b["cv"] / a["cv"])
    print(f"  -{e:<16s} {loo[-1]:.2f}×")
loo = np.array(loo)
print(f"  留一范围 {loo.min():.2f}–{loo.max():.2f}×  (全量 {amp_cv:.2f}×)")

# 符号检验：细带排名 vs 全谱排名，成对比较有多少对在细带里差距更大
pairs = list(itertools.combinations(range(len(encs)), 2))
wid = sum(1 for i, j in pairs
          if abs(b0[i] - b0[j]) / ROWS[t0]["mean"] > abs(overall[i] - overall[j]) / S0["mean"])
print(f"\n成对相对差距在最细带更大的比例: {wid}/{len(pairs)} = {wid/len(pairs):.3f}")
rho = np.corrcoef(np.argsort(np.argsort(-overall)), np.argsort(np.argsort(-b0)))[0, 1]
print(f"整体 PCC 排名 vs 最细带排名 Spearman ρ = {rho:.4f}")

json.dump({"encoders": encs, "sigma_um": {str(t): sig[t] for t in ts},
           "overall_pcc": {e: E[e]["pcc_check"]["mine"] for e in encs},
           "band_pcc": {str(t): {e: E[e]["band_pcc"][str(t)] for e in encs} for t in ts},
           "spread": {str(t): ROWS[t] for t in ts}, "spread_overall": S0,
           "amp_finest": {"cv": amp_cv, "iqr": amp_iq,
                          "loo_cv_min": float(loo.min()), "loo_cv_max": float(loo.max())},
           "pairs_wider_in_fine_band": [wid, len(pairs)], "rank_rho": float(rho)},
          open("results/breadth_amp.json", "w"), indent=1)
print("\n已存 results/breadth_amp.json")

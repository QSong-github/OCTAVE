# -*- coding: utf-8 -*-
"""广度线：15 个编码器 vs 纯形态学下界(imageKNN 检索)，逐带。

只用带 PCC 跨编码器/跨下界比较（同样本、同图、同 t，可比）。
**不报"相对距下界差距的放大倍数"** —— 全谱差距近零(+0.0011)，除以它得到的
任何倍数都是被噪声分母放大出来的（全谱 gap 的跨编码器 CV = 27）。
真正可报的是差距随尺度的**符号翻转与抵消**。
"""
import json, glob
import numpy as np

FL = sorted(glob.glob("results/hest_floor/*.json"))
fs_pcc, fs_band, coh = {}, {}, {}
FENC, FK = None, None
for f in FL:
    dd = json.load(open(f))
    FENC, FK = dd["encoder"], dd["k"]
    for sid, v in dd["samples"].items():
        fs_pcc[sid] = v["pcc"]
        fs_band[sid] = {int(k): x for k, x in v["band_pcc"].items()}
        coh[sid] = dd["cohort"]
print(f"下界: {len(FL)} 队列 / {len(fs_pcc)} 样本  (encoder={FENC}, k={FK})")

E = {}
for f in sorted(glob.glob("results/hest_effres_*.json")):
    dd = json.load(open(f))
    if "_ps_" in f or dd["n_samples"] != 72:
        continue
    E[dd["encoder"]] = dd
encs = list(E)
ids = sorted(set(fs_pcc) & set(list(E.values())[0]["meta"]))
print(f"编码器 {len(encs)}；共有样本 {len(ids)}/72")
assert len(ids) == 72, "下界未覆盖全部样本，池化值不可比"

ts = sorted(int(k) for k in list(E.values())[0]["band_pcc"])
sig = {t: float(np.mean([E[e]["sigma_um"][str(t)] for e in E])) for t in ts}
FB = {t: float(np.nanmean([fs_band[s][t] for s in ids])) for t in ts}
F0 = float(np.nanmean([fs_pcc[s] for s in ids]))
E0 = np.array([E[e]["pcc_check"]["mine"] for e in encs])

print(f"\n整体 PCC: 下界 {F0:.4f}   编码器 {E0.mean():.4f} "
      f"(min {E0.min():.4f} {encs[int(E0.argmin())]}, max {E0.max():.4f} "
      f"{encs[int(E0.argmax())]})")
print(f"  ⇒ 15 个基础模型编码器 + ridge 只比纯 kNN 检索高 {E0.mean()-F0:+.4f} "
      f"({100*(E0.mean()-F0)/F0:+.2f}%)")
nb = int((E0 > F0).sum())
print(f"  ⇒ 整体分数高于下界的编码器: {nb}/{len(encs)}")

print(f"\n{'band':>5s} {'σ(µm)':>7s} {'floor':>8s} {'enc mean':>9s} {'gap':>9s} "
      f"{'gap %':>8s} {'>floor':>8s}")
ROWS = {}
for t in ts:
    b = np.array([E[e]["band_pcc"][str(t)] for e in encs])
    g = float(b.mean() - FB[t])
    nabove = int((b > FB[t]).sum())
    ROWS[t] = {"sigma_um": sig[t], "floor": FB[t], "enc_mean": float(b.mean()),
               "gap": g, "gap_pct": 100 * g / FB[t], "n_above_floor": nabove,
               "gaps": {e: float(x - FB[t]) for e, x in zip(encs, b)}}
    print(f"{t:>5d} {sig[t]:>7.0f} {FB[t]:>8.4f} {b.mean():>9.4f} {g:>+9.4f} "
          f"{100*g/FB[t]:>+7.2f}% {nabove:>5d}/15")
print(f"{'全谱':>5s} {'--':>7s} {F0:>8.4f} {E0.mean():>9.4f} {E0.mean()-F0:>+9.4f} "
      f"{100*(E0.mean()-F0)/F0:>+7.2f}% {nb:>5d}/15")

# 符号翻转点
gp = [(sig[t], ROWS[t]["gap"]) for t in ts]
flip = None
for (s0, g0), (s1, g1) in zip(gp, gp[1:]):
    if g0 > 0 >= g1:
        flip = float(np.exp(np.log(s0) + g0 / (g0 - g1) * (np.log(s1) - np.log(s0))))
        break
t0 = ts[0]
print(f"\n最细带 σ={sig[t0]:.0f}µm: 差距 {ROWS[t0]['gap']:+.4f} "
      f"({ROWS[t0]['gap_pct']:+.1f}%), {ROWS[t0]['n_above_floor']}/15 高于下界")
print(f"符号翻转于 σ ≈ {flip:.0f} µm；此后编码器**低于**纯检索下界，"
      f"最深 {min(r['gap_pct'] for r in ROWS.values()):+.1f}% "
      f"(σ={ROWS[min(ROWS, key=lambda t: ROWS[t]['gap_pct'])]['sigma_um']:.0f}µm)")
neg = [e for e, v in ROWS[t0]["gaps"].items() if v <= 0]
print(f"最细带不高于下界的编码器: {len(neg)}/15 {neg}")
print("\n结论口径：整体分数上编码器几乎就在形态学下界上(+0.44%)，但这是**抵消**的结果——"
      f"\n细带 {ROWS[t0]['gap_pct']:+.1f}%、粗带最低 "
      f"{min(r['gap_pct'] for r in ROWS.values()):+.1f}%，标量把两者相抵。"
      "\n不得报『相对放大 N 倍』：全谱差距近零，任何以它为分母的倍数都无意义。")

json.dump({"floor_encoder": FENC, "floor_k": FK, "n_samples": len(ids),
           "encoders": encs, "floor_overall": F0,
           "enc_overall": {e: float(x) for e, x in zip(encs, E0)},
           "n_above_floor_overall": nb, "rows": {str(t): ROWS[t] for t in ts},
           "gap_overall": float(E0.mean() - F0),
           "gap_pct_overall": float(100 * (E0.mean() - F0) / F0),
           "sign_flip_sigma_um": flip,
           "note": "不得用 gap/gap_overall 作放大倍数：分母近零"},
          open("results/breadth_floor.json", "w"), indent=1)
print("\n已存 results/breadth_floor.json")

# -*- coding: utf-8 -*-
"""配对形态学下界矩阵：每个编码器 vs 它**自己特征**的 kNN 检索下界，逐带。

纪律：
  - 只做同特征配对。跨编码器与下界比会把特征差异混进来（§42 就是栽在这上面）。
  - 主结论一律在**队列层级**（n=10，精确双侧符号检验，最小可得 P=0.00195）。
    72 个样本里 47 个来自 CCRCC+PRAD，池化均值基本是这两个队列，不作主结论。
  - 只比带 PCC（同样本、同图、同 t）。不跨估计量、不跨目标比 σ。
  - **不报任何「细带相对优势 ÷ 整体相对优势」的倍数**：整体优势可以近零，
    比值会炸出无意义的大数（§42 里的 23.9×、早期检查里 conch_v15 的 −21.1× 都是这个雷）。
    「细带优势更大」一律用**差的符号**做队列层级检验，不用比值。
"""
import json, glob, os
import numpy as np
from math import comb


def signp(k, n):
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


ENC = sorted(os.path.basename(d).replace("hest_floor_", "")
             for d in glob.glob("results/hest_floor_*") if os.path.isdir(d))
print(f"配对下界可用编码器 {len(ENC)}: {ENC}")

PS = {}
for f in sorted(glob.glob("results/hest_effres_ps_*.json")):
    d = json.load(open(f))
    PS[d["encoder"]] = d
print(f"逐样本带 PCC 可用编码器 {len(PS)}: {sorted(PS)}")

USE = sorted(set(ENC) & set(PS))
print(f"两者兼有、可做逐带配对检验的编码器 {len(USE)}: {USE}\n")
if not USE:
    print("[停] 还没有编码器同时具备配对下界与逐样本带 PCC。")
    raise SystemExit(0)

OUT = {}
for e in USE:
    fl_pcc, fl_band, coh = {}, {}, {}
    ok = True
    for f in sorted(glob.glob(f"results/hest_floor_{e}/*.json")):
        d = json.load(open(f))
        for sid, v in d["samples"].items():
            fl_pcc[sid] = v["pcc"]
            fl_band[sid] = {str(k): x for k, x in v["band_pcc"].items()}
            coh[sid] = d["cohort"]
    ps_band = PS[e]["per_sample_band_pcc"]
    ps_pcc = PS[e]["per_sample_pcc"]
    ids = sorted(set(fl_pcc) & set(ps_band))
    if len(ids) != 72:
        print(f"  {e}: 仅 {len(ids)}/72 样本配对成功，跳过（下界 {len(fl_pcc)}，"
              f"逐样本 {len(ps_band)}）")
        continue
    ts = sorted(int(t) for t in PS[e]["sigma_um"])
    sig = {t: PS[e]["sigma_um"][str(t)] for t in ts}

    # 相对量一律在**队列层级**算：分母是队列内的下界中位数，远离零。
    # 逐样本除法是禁区 —— 下界的带 PCC 可以为负或近零，逐样本比值会炸到 1e6
    # （今天已在三处踩到同一个雷）。队列本来就是检验单元，正好一致。
    FLOOR_MIN = 0.02

    def by_cohort(vals):
        d = {}
        for s in ids:
            d.setdefault(coh[s], []).append(vals[s])
        return d

    cg_o = by_cohort({s: ps_pcc[s] - fl_pcc[s] for s in ids})
    cf_o = by_cohort({s: fl_pcc[s] for s in ids})
    rel_o = {c: (float(np.median(cg_o[c])) / float(np.median(cf_o[c])))
             if float(np.median(cf_o[c])) > FLOOR_MIN else None for c in cg_o}

    rows = {}
    for t in ts:
        gp = by_cohort({s: ps_band[s][str(t)] - fl_band[s][str(t)] for s in ids})
        fp = by_cohort({s: fl_band[s][str(t)] for s in ids})
        med = {c: float(np.median(gp[c])) for c in gp}
        fmed = {c: float(np.median(fp[c])) for c in fp}
        rel = {c: (med[c] / fmed[c]) if fmed[c] > FLOOR_MIN else None for c in med}
        k = sum(1 for v in med.values() if v > 0); n = len(med)
        # 「细带的相对优势大于整体」：只在两边分母都合格的队列上比，用差的符号
        cmp_ok = [c for c in rel if rel[c] is not None and rel_o.get(c) is not None]
        kd = sum(1 for c in cmp_ok if rel[c] > rel_o[c])
        nlow = sum(1 for c in fmed if fmed[c] <= FLOOR_MIN)
        rows[t] = {"sigma_um": sig[t],
                   "pooled_gap": float(np.mean([ps_band[s][str(t)] - fl_band[s][str(t)]
                                                for s in ids])),
                   "pooled_floor": float(np.mean([fl_band[s][str(t)] for s in ids])),
                   "cohort_rel_median": (float(np.median([v for v in rel.values()
                                                          if v is not None]))
                                         if any(v is not None for v in rel.values()) else None),
                   "n_cohorts_floor_too_low": nlow,
                   "n_samples_win": int(sum(1 for s in ids
                                            if ps_band[s][str(t)] > fl_band[s][str(t)])),
                   "n_cohorts_win": k, "n_cohorts": n, "P_sign": signp(k, n),
                   "n_cohorts_band_gt_overall": kd, "n_cmp": len(cmp_ok),
                   "P_band_gt_overall": signp(kd, len(cmp_ok)) if cmp_ok else None,
                   "cohort_median": med, "cohort_rel": rel}
    d0 = {s: ps_pcc[s] - fl_pcc[s] for s in ids}
    by0 = {}
    for s in ids:
        by0.setdefault(coh[s], []).append(d0[s])
    med0 = {c: float(np.median(v)) for c, v in by0.items()}
    k0 = sum(1 for v in med0.values() if v > 0); n0 = len(med0)
    OUT[e] = {"bands": rows, "overall": {
        "pooled_gap": float(np.mean(list(d0.values()))),
        "pooled_floor": float(np.mean([fl_pcc[s] for s in ids])),
        "n_samples_win": int(sum(1 for v in d0.values() if v > 0)),
        "n_cohorts_win": k0, "n_cohorts": n0, "P_sign": signp(k0, n0),
        "cohort_median": med0}}

    # 诊断：相对量随 σ 的变化，究竟是分子(差距)在动还是分母(下界)在动？
    #（三个已完成的编码器里，phikon_v2 的绝对差距几乎是平的，相对量的衰减
    #  完全来自下界从 0.10 涨到 0.43；若不报这一行，会把归一化效应当成物理效应。）
    from scipy.stats import spearmanr
    sv = np.array([rows[t]["sigma_um"] for t in ts])
    gv = np.array([rows[t]["pooled_gap"] for t in ts])
    fv = np.array([rows[t]["pooled_floor"] for t in ts])
    rho_g = float(spearmanr(sv, gv).statistic)
    rho_f = float(spearmanr(sv, fv).statistic)
    rho_a = float(spearmanr(sv, np.abs(gv)).statistic)
    OUT[e]["profile"] = {"rho_sigma_gap": rho_g, "rho_sigma_absgap": rho_a,
                         "rho_sigma_floor": rho_f,
                         "gap_min": float(gv.min()), "gap_max": float(gv.max()),
                         "gap_flat_ratio": float(np.abs(gv).max() / max(np.abs(gv).min(), 1e-9)),
                         "floor_range": [float(fv.min()), float(fv.max())],
                         "crosses_zero": bool((gv > 0).any() and (gv < 0).any())}

    o = OUT[e]["overall"]
    print(f"\n### {e}")
    print(f"  整体: 池化差 {o['pooled_gap']:+.4f} "
          f"({100*o['pooled_gap']/o['pooled_floor']:+.2f}%)  "
          f"样本 {o['n_samples_win']}/72  队列 {o['n_cohorts_win']}/{o['n_cohorts']}  "
          f"P={o['P_sign']:.5f}")
    pr = OUT[e]["profile"]
    print(f"  剖面: 绝对差距 {pr['gap_min']:+.4f}…{pr['gap_max']:+.4f}"
          f"{'（穿过零）' if pr['crosses_zero'] else ''}  "
          f"ρ(σ, 差距)={pr['rho_sigma_gap']:+.2f}  ρ(σ, |差距|)={pr['rho_sigma_absgap']:+.2f}  "
          f"ρ(σ, 下界)={pr['rho_sigma_floor']:+.2f}  下界 "
          f"{pr['floor_range'][0]:.3f}→{pr['floor_range'][1]:.3f}")
    print(f"        ⇒ 相对量随 σ 的变化主要由"
          f"{'分子(差距)' if abs(pr['rho_sigma_absgap']) > 0.7 else '分母(下界)'}驱动")
    print(f"  {'σ(µm)':>7s} {'池化差':>9s} {'队列相对中位':>13s} {'样本胜':>8s} "
          f"{'队列胜':>8s} {'P':>9s} {'带>整体':>9s} {'P':>9s} {'分母不合格':>10s}")
    for t in ts:
        r = rows[t]
        rm = f"{100*r['cohort_rel_median']:+.2f}%" if r["cohort_rel_median"] is not None else "n/a"
        pb = f"{r['P_band_gt_overall']:.5f}" if r["P_band_gt_overall"] is not None else "n/a"
        print(f"  {r['sigma_um']:>7.0f} {r['pooled_gap']:>+9.4f} {rm:>13s} "
              f"{r['n_samples_win']:>5d}/72 {r['n_cohorts_win']:>5d}/{r['n_cohorts']} "
              f"{r['P_sign']:>9.5f} {r['n_cohorts_band_gt_overall']:>4d}/{r['n_cmp']:<4d} "
              f"{pb:>9s} {r['n_cohorts_floor_too_low']:>10d}")

if OUT:
    ts0 = sorted(int(t) for t in PS[USE[0]]["sigma_um"])
    print(f"\n\n=== 跨编码器汇总：多少个编码器在该带**显著**高于自己的检索下界 (P<0.05) ===")
    print(f"{'σ(µm)':>7s} {'显著':>8s} {'队列胜中位':>11s} {'队列相对中位':>13s}")
    for t in ts0:
        vs = [OUT[e]["bands"][t] for e in OUT if t in OUT[e]["bands"]]
        if not vs:
            continue
        rr = [v["cohort_rel_median"] for v in vs if v["cohort_rel_median"] is not None]
        print(f"{vs[0]['sigma_um']:>7.0f} "
              f"{sum(1 for v in vs if v['P_sign'] < 0.05):>5d}/{len(vs)} "
              f"{np.median([v['n_cohorts_win'] for v in vs]):>10.1f} "
              f"{(f'{100*np.median(rr):+.2f}%' if rr else 'n/a'):>13s}")
    vs = [OUT[e]["overall"] for e in OUT]
    print(f"{'全谱':>7s} {sum(1 for v in vs if v['P_sign'] < 0.05):>5d}/{len(vs)} "
          f"{np.median([v['n_cohorts_win'] for v in vs]):>10.1f}")
    print("\n注：相对量一律取队列中位数之比（分母 > 0.02 才算），逐样本不做除法。")

json.dump(OUT, open("results/matched_floor_matrix.json", "w"), indent=1)
print(f"\n已存 results/matched_floor_matrix.json ({len(OUT)} 个编码器)")

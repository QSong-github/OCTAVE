# -*- coding: utf-8 -*-
"""HEST（~100 µm）上的 OCTAVE 式差距：ridge（官方 α，hest_effres_ps）与 domain oracle（hest_oracle_bands）的标量差距与最细带差距。
Δ_scalar = (PCC_r − PCC_o)/PCC_r，Δ_fine = (β1_r − β1_o)/β1_r（与正文 Xenium 定义一致；oracle 高于 ridge 时为负）。
逐编码器：样本→队列中位数→10 队列符号计数；再跨 57 编码器汇总。"""
import json, glob, os, numpy as np
R = os.environ.get("S4ST_RESULTS", "results")
encs = sorted(f.split("hest_oracle_bands_")[1][:-5] for f in glob.glob(f"{R}/hest_oracle_bands_*.json"))
out = []
for e in encs:
    O = json.load(open(f"{R}/hest_oracle_bands_{e}.json"))["samples"]
    sel = f"{R}/hest_effres_ps_selfull_{e}.json"  # 统一基线：留一队列选 α 的 ridge（优先）
    P = json.load(open(sel)) if os.path.exists(sel) else json.load(open(f"{R}/hest_effres_ps_{e}.json"))
    if "per_sample_band_pcc" not in P:
        alt = f"{R}/hest_effres_ps_full_{e}.json"
        if not os.path.exists(alt): continue
        P = json.load(open(alt))
    pb, pp, ps = P["per_sample_band_pcc"], P["per_sample_pcc"], P["per_sample_sigma_um"]
    coh = {}
    for s, o in O.items():
        if s not in pb: continue
        r1, o1 = pb[s]["1"], o["band_pcc_oracle"]["1"]; rp, op = pp[s], o["oracle_pcc"]
        coh.setdefault(o["cohort"], []).append(dict(ds=(rp - op) / rp, df=(r1 - o1) / r1, b1r=r1, b1o=o1, pr=rp, po=op, sig1=o["sigma_um"]["1"], sig1_r=ps[s]["1"], pitch=o["pitch_um"]))
    cm = {c: {k: float(np.median([d[k] for d in v])) for k in v[0]} for c, v in coh.items()}
    C = list(cm.values())
    out.append(dict(enc=e, n_coh=len(C), ds=float(np.median([c["ds"] for c in C])), df=float(np.median([c["df"] for c in C])),
                    oracle_above_scalar=int(sum(c["ds"] < 0 for c in C)), oracle_above_fine=int(sum(c["df"] < 0 for c in C)), fine_gap_larger=int(sum(abs(c["df"]) > abs(c["ds"]) for c in C)),
                    b1r=float(np.median([c["b1r"] for c in C])), b1o=float(np.median([c["b1o"] for c in C])), pr=float(np.median([c["pr"] for c in C])), po=float(np.median([c["po"] for c in C])),
                    sig1=float(np.median([c["sig1"] for c in C])), sig_check=float(max(abs(c["sig1"] - c["sig1_r"]) for c in C)), pitch=float(np.median([c["pitch"] for c in C]))))
A = {k: np.array([o[k] for o in out]) for k in out[0] if k != "enc"}
print(f"编码器 {len(out)}；σ1 中位 {np.median(A['sig1']):.1f} µm（间距 {np.median(A['pitch']):.1f} µm；与 ridge 文件的 σ1 最大差 {A['sig_check'].max():.2e}）")
print(f"标量：ridge {np.median(A['pr']):.3f}  oracle {np.median(A['po']):.3f}  Δ_scalar 中位 {100*np.median(A['ds']):.1f}%  oracle 高于 ridge 的队列数 中位 {np.median(A['oracle_above_scalar']):.0f}/10")
print(f"最细带：β1 ridge {np.median(A['b1r']):.3f}  oracle {np.median(A['b1o']):.3f}  Δ_fine 中位 {100*np.median(A['df']):.1f}% [{100*A['df'].min():.1f}, {100*A['df'].max():.1f}]  oracle 高于 ridge 的队列数 中位 {np.median(A['oracle_above_fine']):.0f}/10 [{A['oracle_above_fine'].min()}, {A['oracle_above_fine'].max()}]")
print(f"|Δ_fine|>|Δ_scalar| 的队列数 中位 {np.median(A['fine_gap_larger']):.0f}/10；编码器数 ≥9/10: {(A['fine_gap_larger']>=9).sum()}/{len(out)}")
n_sel = sum(os.path.exists(f"{R}/hest_effres_ps_selfull_{e}.json") for e in encs); print(f"使用选 α ridge 的编码器: {n_sel}/{len(encs)}")
json.dump(out, open(f"{R}/hest_octave_delta.json", "w"), indent=1)
for o in out[:6]: print(f"   {o['enc']:16s} PCC r/o {o['pr']:.3f}/{o['po']:.3f}  β1 r/o {o['b1r']:.3f}/{o['b1o']:.3f}  Δs {100*o['ds']:+.0f}%  Δf {100*o['df']:+.0f}%  oracle>ridge: 标量 {o['oracle_above_scalar']}/10 最细带 {o['oracle_above_fine']}/10")

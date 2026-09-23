# -*- coding: utf-8 -*-
"""图 4 的 c–f：固定评测栅格下 PCC 与 σ 的最优点不一致，以及划分几何的敏感度对比。

正文此前从未陈述这四板的结果，本脚本把它们算出来并冻结。
"""
import glob, json, os, re
import numpy as np
R = "/path/to/project/results"
BINS = [8, 16, 32, 64]
GR = [16, 8, 4, 2]


def J(p):
    try: return json.load(open(p))
    except Exception: return None


def spec(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    if "Human_Lung_Cancer_FFPE" in n: return "Lung"   # v1 与 Prime 5K 同一供体同一组织块，按一个标本计
    return m.group(1) if m else n


# ── c：固定 16 µm 评测栅格下的 PCC，与跟随栅格下的 σ，各自在哪个预测分箱最好
rows = []
for f in sorted(glob.glob(R + "/downstream_*.json")):
    bn = os.path.basename(f)
    if "downstream2" in bn or bn == "downstream_summary.json": continue
    n = bn[len("downstream_"):-5]
    d = J(f)
    if not d or "scales" not in d: continue
    for b in BINS:
        q = J("%s/xenium_multi/%s_bin%d.json" % (R, n, b)) if b != 16 else J("%s/xenium/%s.json" % (R, n))
        fx = d["scales"].get(str(b), {}).get("pcc")
        if q and fx is not None:
            rows.append(dict(region=n, specimen=spec(n), bin_um=b, pcc_fixed=fx,
                             eq_sigma=q.get("eq_sigma"), eq_flag=q.get("eq_flag")))
P = {b: [r["pcc_fixed"] for r in rows if r["bin_um"] == b] for b in BINS}
S = {b: [r["eq_sigma"] for r in rows if r["bin_um"] == b and r["eq_flag"] == "ok"
         and r["eq_sigma"] is not None and np.isfinite(r["eq_sigma"])] for b in BINS}
mp = {b: float(np.median(P[b])) for b in BINS if P[b]}
ms = {b: float(np.median(S[b])) for b in BINS if S[b]}
b_pcc = max(mp, key=mp.get); b_sig = min(ms, key=ms.get)
print("面板 c（n=%d 区域）" % len(set(r["region"] for r in rows)))
for b in BINS:
    print("  预测分箱 %2d µm: 固定栅格 PCC %.4f   σ %.1f µm" % (b, mp.get(b, np.nan), ms.get(b, np.nan)))
print("  ⇒ PCC 在 %d µm 最高，σ 在 %d µm 最好 —— 两个判据的最优点%s"
      % (b_pcc, b_sig, "不一致" if b_pcc != b_sig else "一致"))

# ── d–f：空间块划分几何 16x16 → 2x2
pr = {}
for g in GR:
    for f in glob.glob("%s/proto_g%d/*.json" % (R, g)):
        k = J(f)
        if k: pr.setdefault(k["name"].replace("_g%d" % g, ""), {})[g] = k
full = {k: v for k, v in pr.items() if all(g in v for g in GR)}
vp = {g: [full[k][g]["pcc"] for k in full] for g in GR}
vs = {g: [full[k][g]["eq_sigma"] for k in full
          if full[k][g].get("eq_flag") == "ok" and np.isfinite(full[k][g]["eq_sigma"])] for g in GR}
mpg = {g: float(np.median(vp[g])) for g in GR}
msg = {g: float(np.median(vs[g])) for g in GR}
rp = 100 * (mpg[16] - mpg[2]) / mpg[16]
rs = 100 * (msg[2] - msg[16]) / msg[16]
print("\n面板 d–f（n=%d 区域）" % len(full))
for g in GR:
    print("  %2dx%-2d 块 CV: PCC %.4f   σ %.1f µm" % (g, g, mpg[g], msg[g]))
print("  16x16 → 2x2: PCC 变化 %.2f%%，σ 变化 %.2f%%  ⇒ σ 的敏感度是 PCC 的 %.1f 倍" % (rp, rs, rs / rp))

out = dict(panel_c=dict(n_regions=len(set(r["region"] for r in rows)), bins_um=BINS,
                        pcc_fixed_grid=mp, sigma_follow_grid=ms,
                        best_bin_pcc=b_pcc, best_bin_sigma=b_sig, by_region=rows),
           panel_def=dict(n_regions=len(full), grids=GR, pcc=mpg, sigma=msg,
                          pcc_change_pct=rp, sigma_change_pct=rs, sensitivity_ratio=rs / rp,
                          by_region={k: {str(g): dict(pcc=full[k][g]["pcc"],
                                                      eq_sigma=full[k][g]["eq_sigma"],
                                                      eq_flag=full[k][g].get("eq_flag"))
                                         for g in GR} for k in full}))
json.dump(out, open(R + "/split_sensitivity.json", "w"), indent=1)
print("\n-> %s/split_sensitivity.json" % R)

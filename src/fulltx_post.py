# -*- coding: utf-8 -*-
"""
全转录组结果的三项后续分析 —— 全部基于 fulltx 已存的 npz，不重跑模型。

fulltx (38983169) 留下三个必须澄清的问题：
 ① PCC~Moran 全谱 r=0.863/0.896，但 PCC~噪声天花板也有 0.796/0.830。
    两者互相纠缠 ⇒ 必须偏相关才能说清「指标在按基因有多空间给分」还是
    「在按基因有多可测给分」。
 ② 等价 σ 在 top-50→top-2000（基因数 40×）几乎不动（77→85µm），但 5000 之后
    升到 100/130/161µm。同期天花板从 0.249 塌到 0.156 ⇒ 需要检验 σ 的上升
    是不是纯粹的测量地板效应，而非模型真的变粗。按天花板分层重算 σ。
 ③ Moran 膨胀的绝对差值近似常数（+0.65~+0.80）⇒ 需要直接报预测 Moran 的分布，
    确认「预测的空间自相关几乎是常数、与真值无关」。
"""
import json, numpy as np
from scipy.stats import pearsonr, spearmanr
RES = "/blue/qsong1/wang.qing/systema4ST/results"
SL = ["P2", "P5"]

def partial(x, y, z):
    """x~y 控制 z 的偏相关（对 z 做线性残差化）。"""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    r = lambda v: v - np.polyval(np.polyfit(z, v, 1), z)
    return pearsonr(r(x), r(y))[0], spearmanr(r(x), r(y))[0]

D = {s: np.load(f"{RES}/fulltx_{s}.npz", allow_pickle=True) for s in SL}
OUT = {}
print("="*88); print("① 偏相关：指标在按「有多空间」给分，还是按「有多可测」给分？"); print("="*88)
print(f"{'':6s}{'PCC~Moran':>22s}{'PCC~天花板':>22s}{'PCC~Moran|天花板':>24s}{'PCC~天花板|Moran':>24s}")
for s in SL:
    d = D[s]; p, I, c = d["pcc"], d["I_true"], d["ceil"]
    m = np.isfinite(c) & (c > 0.02)                       # 天花板过低的基因无信息
    a = pearsonr(I[m], p[m])[0]; b = pearsonr(c[m], p[m])[0]
    pa = partial(p[m], I[m], c[m]); pb = partial(p[m], c[m], I[m])
    OUT[s] = dict(n=int(m.sum()), r_moran=float(a), r_ceil=float(b),
                  pr_moran=float(pa[0]), pr_ceil=float(pb[0]),
                  sr_moran=float(pa[1]), sr_ceil=float(pb[1]))
    print(f"{s:6s}{a:>22.3f}{b:>22.3f}{pa[0]:>24.3f}{pb[0]:>24.3f}")
print(f"\n  Moran 与天花板本身的相关: " +
      "  ".join(f"[{s}] r={pearsonr(D[s]['I_true'], np.nan_to_num(D[s]['ceil']))[0]:.3f}" for s in SL))

print(f"\n{'='*88}"); print("② 等价 σ 按噪声天花板分层 —— σ 的上升是测量地板还是模型变粗？"); print("="*88)
print(f"{'天花板区间':>14s}{'基因数':>8s}{'PCC':>9s}{'等价σ':>9s}{'真实Moran':>11s}{'Moran差':>10s}")
BINS = [(0.60, 1.01), (0.45, 0.60), (0.30, 0.45), (0.20, 0.30), (0.10, 0.20), (0.0, 0.10)]
lay = {}
for lo, hi in BINS:
    row = []
    for s in SL:
        d = D[s]; sg = d["sigma"]; lad = d["ladder"]; c = np.nan_to_num(d["ceil"])
        idx = np.where((c >= lo) & (c < hi))[0]
        if len(idx) < 30: continue
        v = float(np.nanmean(d["pcc"][idx])); curve = lad[:, idx].mean(1)
        pts = sorted(zip(sg, curve)); e = np.nan
        if v >= pts[0][1]: e = pts[0][0]
        else:
            for (a0, v0), (a1, v1) in zip(pts, pts[1:]):
                if v0 >= v >= v1: e = a0 + (v0-v)/max(v0-v1, 1e-12)*(a1-a0); break
        row.append((len(idx), v, e, float(np.nanmean(d["I_true"][idx])),
                    float(np.nanmedian(d["I_pred"][idx]-d["I_true"][idx]))))
    if not row: continue
    g = lambda i: np.nanmean([r[i] for r in row])
    lay[f"{lo:.2f}-{hi:.2f}"] = dict(n=g(0), pcc=g(1), eq=g(2), moran=g(3), diff=g(4))
    print(f"{f'{lo:.2f}–{hi:.2f}':>14s}{g(0):>8.0f}{g(1):>9.4f}" +
          (f"{g(2):>9.0f}" if np.isfinite(g(2)) else f"{'右删失':>9s}") +
          f"{g(3):>11.3f}{g(4):>+10.3f}")

print(f"\n{'='*88}"); print("③ 预测的 Moran's I 分布 —— 是否几乎与真值无关"); print("="*88)
print(f"{'':6s}{'':>10s}{'p10':>9s}{'p25':>9s}{'中位':>9s}{'p75':>9s}{'p90':>9s}{'四分位距':>11s}")
for s in SL:
    d = D[s]
    for nm, v in (("真实 I", d["I_true"]), ("预测 I", d["I_pred"])):
        q = np.nanpercentile(v, [10, 25, 50, 75, 90])
        print(f"{s:6s}{nm:>10s}" + "".join(f"{x:>9.3f}" for x in q) + f"{q[3]-q[1]:>11.3f}")
    print(f"{'':6s}{'相关':>10s}  预测 I ~ 真实 I: r={pearsonr(d['I_true'], d['I_pred'])[0]:.3f} "
          f"ρ={spearmanr(d['I_true'], d['I_pred'])[0]:.3f}")

print(f"\n{'='*88}"); print("④ 逐基因等价 σ（全谱）与基因属性的关系"); print("="*88)
for s in SL:
    d = D[s]; sg = d["sigma"]; lad = d["ladder"]
    eq = np.full(lad.shape[1], np.nan)
    for j in range(lad.shape[1]):
        v = d["pcc"][j]; pts = sorted(zip(sg, lad[:, j]))
        if v >= pts[0][1]: eq[j] = pts[0][0]; continue
        for (a0, v0), (a1, v1) in zip(pts, pts[1:]):
            if v0 >= v >= v1: eq[j] = a0 + (v0-v)/max(v0-v1, 1e-12)*(a1-a0); break
    ok = np.isfinite(eq)
    print(f"  [{s}] 可解析 {ok.sum()}/{len(eq)} ({ok.mean():.1%})  中位 σ={np.nanmedian(eq):.0f}µm")
    for nm, v in (("Moran", d["I_true"]), ("天花板", np.nan_to_num(d["ceil"])), ("PCC", d["pcc"])):
        m = ok & np.isfinite(v)
        print(f"        σ ~ {nm:8s} r={pearsonr(v[m], eq[m])[0]:+.3f}  ρ={spearmanr(v[m], eq[m])[0]:+.3f}")
    OUT[s]["eq_median"] = float(np.nanmedian(eq)); OUT[s]["eq_resolved"] = float(ok.mean())
json.dump({"partial": OUT, "by_ceiling": lay}, open(f"{RES}/fulltx_post.json", "w"),
          indent=2, ensure_ascii=False)
print(f"\n已存 {RES}/fulltx_post.json")

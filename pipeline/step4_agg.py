#!/usr/bin/env python
"""第四步的裁决：下游落差站在标量一边还是 β₁ 一边。

口径与全文一致：逐区域算量 → 样本内取中位 → 跨 8 个样本做精确符号检验。
不做逐区域比值的平均（近零分母），比值只在样本层级形成一次。
"""
import glob, json, os, re
import numpy as np
from math import comb
R = "/path/to/systema4ST/results"


def spec(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    return m.group(1) if m else n


def signp(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


D = [json.load(open(f)) for f in sorted(glob.glob(R + "/oracle_ds_*.json"))]
print("区域 %d 个 / 样本 %d 个\n" % (len(D), len({spec(d["name"]) for d in D})))
KEYS = ["svg_top_jaccard", "svg_rank_rho", "hotspot_jaccard",
        "coloc_preserve", "hotspot_recall_selectivity", "boundary_shift_um"]

# 每区域：标量落差、最细带落差、各下游落差
byspec = {}
for d in D:
    s, f = d["delta_scalar"], d["delta_fine"]
    byspec.setdefault(spec(d["name"]), []).append((s, f, d["readout_gaps"]))
SP_ = sorted(byspec)
sc = np.array([np.median([x[0] for x in byspec[k]]) for k in SP_])
fi = np.array([np.median([x[1] for x in byspec[k]]) for k in SP_])
print("样本层级：标量落差中位 %.1f%%  最细带落差中位 %.1f%%  比值 %.2f\n"
      % (np.median(sc), np.median(fi), np.median(fi) / np.median(sc)))

print("%-28s %9s %9s %9s   %-11s %-11s" %
      ("下游读数", "落差中位", "/标量", "/最细带", "> 标量", "> 最细带"))
rows, pooled = {}, []
for k in KEYS:
    g = []
    for kk in SP_:
        v = [x[2].get(k) for x in byspec[kk] if x[2].get(k) is not None]
        # 两侧都为 0 的度量（界面位移在亚 bin 处恒为 0）无定义，剔除
        v = [x for x in v if abs(x) > 1e-12]
        g.append(np.median(v) if v else np.nan)
    g = np.array(g, float)
    m = np.isfinite(g)
    if m.sum() < 5:
        print("%-28s （有效样本 %d 个，不足以检验）" % (k, int(m.sum()))); continue
    gs, ss, ff = g[m], sc[m], fi[m]
    n = len(gs)
    k1 = int((gs > ss).sum()); k2 = int((gs > ff).sum())
    rows[k] = dict(n=n, gap_median=float(np.median(gs)),
                   over_scalar=k1, P_over_scalar=signp(k1, n),
                   over_fine=k2, P_over_fine=signp(k2, n),
                   ratio_to_scalar=float(np.median(gs / ss)),
                   ratio_to_fine=float(np.median(gs / ff)))
    pooled += list(gs / ss)
    print("%-28s %8.1f%% %9.2f %9.2f   %-11s %-11s"
          % (k, np.median(gs), np.median(gs / ss), np.median(gs / ff),
             "%d/%d P=%.4f" % (k1, n, signp(k1, n)),
             "%d/%d P=%.4f" % (k2, n, signp(k2, n))))

print("\n合计：下游落差 / 标量落差 的中位 = %.2f（%d 个样本×读数）" % (np.median(pooled), len(pooled)))
nk = [k for k in rows if rows[k]["P_over_scalar"] < 0.05 and rows[k]["over_scalar"] > rows[k]["n"] / 2]
nf = [k for k in rows if rows[k]["P_over_fine"] < 0.05 and rows[k]["over_fine"] > rows[k]["n"] / 2]
print("显著大于标量落差的读数: %d/%d %s" % (len(nk), len(rows), nk))
print("显著大于最细带落差的读数: %d/%d %s" % (len(nf), len(rows), nf))
json.dump(dict(specimens=SP_, delta_scalar=sc.tolist(), delta_fine=fi.tolist(),
               readouts=rows, pooled_ratio_median=float(np.median(pooled))),
          open(R + "/step4_downstream.json", "w"), indent=1)
print("-> %s/step4_downstream.json" % R)

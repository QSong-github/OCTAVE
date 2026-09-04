# -*- coding: utf-8 -*-
"""对账：hest_effres_ps_*.json 是否与现有 hest_effres_*.json 完全一致。

ps 版只在原脚本上加了「额外保存逐样本值」，算法一行未改。若池化后的 band_pcc、
sigma_um、pcc_check 有任何非零差异，说明重算不是同一件事，逐样本值不能拿来
和现有结论混用。同时验证：逐样本值 nanmean 回去必须等于池化值。
"""
import json, glob, os
import numpy as np

bad, ok = [], []
for f in sorted(glob.glob("results/hest_effres_ps_*.json")):
    e = os.path.basename(f).replace("hest_effres_ps_", "").replace(".json", "")
    ref_p = f"results/hest_effres_{e}.json"
    if not os.path.exists(ref_p):
        bad.append((e, "缺参照文件")); continue
    new, ref = json.load(open(f)), json.load(open(ref_p))
    ts = sorted(int(t) for t in ref["band_pcc"])
    dmax_b = max(abs(new["band_pcc"][str(t)] - ref["band_pcc"][str(t)]) for t in ts)
    dmax_s = max(abs(new["sigma_um"][str(t)] - ref["sigma_um"][str(t)]) for t in ts)
    dpcc = abs(new["pcc_check"]["mine"] - ref["pcc_check"]["mine"])
    nsm = new["n_samples"] == ref["n_samples"] == 72
    same_ids = set(new["meta"]) == set(ref["meta"])

    # 逐样本 nanmean 是否重现池化值
    ps = new["per_sample_band_pcc"]
    ids = sorted(ps)
    dmax_r = max(abs(float(np.nanmean([ps[s][str(t)] for s in ids])) - ref["band_pcc"][str(t)])
                 for t in ts)
    nnan = sum(1 for s in ids for t in ts if not np.isfinite(ps[s][str(t)]))

    row = (e, len(ids), dmax_b, dmax_s, dpcc, dmax_r, nnan, nsm and same_ids)
    (ok if (dmax_b < 1e-12 and dmax_s < 1e-9 and dpcc < 1e-12 and dmax_r < 1e-9
            and nsm and same_ids) else bad).append(row)

print(f"{'编码器':<16s}{'样本':>5s}{'Δband':>11s}{'Δσ':>11s}{'Δpcc':>11s}"
      f"{'Δ重现':>11s}{'NaN':>6s}{'样本集一致':>10s}")
for r in sorted(ok + [x for x in bad if len(x) == 8], key=lambda x: x[0]):
    e, n, a, b, c, d, nn, s = r
    print(f"{e:<16s}{n:>5d}{a:>11.2e}{b:>11.2e}{c:>11.2e}{d:>11.2e}{nn:>6d}"
          f"{('是' if s else '否'):>10s}")
for x in bad:
    if len(x) != 8:
        print(f"  [问题] {x}")
print(f"\n完全一致: {len(ok)}/{len(ok)+len(bad)}")
if len(ok) != len(ok) + len(bad):
    print("!! 有不一致，逐样本值不可与现有结论混用")

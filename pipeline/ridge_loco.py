# -*- coding: utf-8 -*-
"""岭回归 α 的留一队列选择（不偷看被留出的队列）。
对每个编码器：候选 α = 官方公式 ∪ {0.1,1,10,100,1e3,1e4,1e5}。对每个被留出的队列 c，
在其余队列的样本上按「相对官方的平均逐样本 PCC 增益」选 α*，把 α* 在 c 上的逐样本 PCC 记入结果。
输出 results/hest_rsel_ps_<enc>.json（与 hest_effres_ps_ 同结构，可直接喂 k_sens/cohort_spread），
以及 results/ridge_alpha_grid.json（逐编码器 × α 的样本均值与逐队列选中的 α）。"""
import json, glob, os, sys, numpy as np
R = "/blue/qsong1/wang.qing/systema4ST/results"
ALPHAS = ["0.1", "1", "10", "100", "1000", "10000", "100000"]
ENC = sys.argv[1].split(",")
grid = {}
for e in ENC:
    off = json.load(open(f"{R}/hest_effres_ps_{e}.json"))["per_sample_pcc"]
    P = {"official": off}
    for a in ALPHAS:
        f = f"{R}/ridge_alpha/hest_ra_{e}_a{a}.json"
        if os.path.exists(f): P[a] = json.load(open(f))["per_sample_pcc"]
    coh = {}
    for f in glob.glob(f"{R}/hest_floor_{e}/*.json"):
        d = json.load(open(f))
        for s in d["samples"]: coh[s] = d["cohort"]
    if not coh:   # 扩集编码器不跑检索地板；队列归属从块预言机文件取
        for s, v in json.load(open(f"{R}/hest_blocks_{e}.json"))["samples"].items(): coh[s] = v["cohort"]
    ids = sorted(set(off) & set(coh))
    assert len(ids) == 72, (e, len(ids))
    assert all(set(P[a]) >= set(ids) for a in P), (e, "α 文件样本不齐")
    cohorts = sorted(set(coh[s] for s in ids))
    sel, out = {}, {}
    for c in cohorts:
        others = [s for s in ids if coh[s] != c]
        gain = {a: float(np.mean([P[a][s] - off[s] for s in others])) for a in P}
        a_star = max(gain, key=gain.get); sel[c] = a_star
        for s in ids:
            if coh[s] == c: out[s] = P[a_star][s]
    json.dump({"encoder": e, "head": "ridge", "alpha_rule": "leave-one-cohort-out selection over official+grid",
               "selected_alpha_by_cohort": sel, "per_sample_pcc": out, "mean_pcc": float(np.mean(list(out.values())))},
              open(f"{R}/hest_rsel_ps_{e}.json", "w"), indent=1)
    grid[e] = {"mean_by_alpha": {a: float(np.mean([P[a][s] for s in ids])) for a in P}, "selected_alpha_by_cohort": sel,
               "loco_mean": float(np.mean(list(out.values()))), "official_mean": float(np.mean([off[s] for s in ids]))}
    print("%-14s 官方 %.4f | " % (e, grid[e]["official_mean"]) + " ".join("α=%s %.4f" % (a, grid[e]["mean_by_alpha"][a]) for a in ALPHAS if a in P)
          + " | 留一队列选 α 后 %.4f (%+.4f)  选中: %s" % (grid[e]["loco_mean"], grid[e]["loco_mean"] - grid[e]["official_mean"], sorted(set(sel.values()))))
json.dump(grid, open(f"{R}/ridge_alpha_grid.json", "w"), indent=1)
print("已存 ridge_alpha_grid.json 与 %d 个 hest_rsel_ps_*.json" % len(grid))

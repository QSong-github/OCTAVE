# -*- coding: utf-8 -*-
"""在不偷看测试队列的前提下选回归头：留一队列（leave-one-cohort-out）。
对每个编码器、每个被留出的队列 c：在其余 9 个队列上按「相对官方 ridge 的平均增益」选出头 h*，
再报 h* 在 c 上的测试 PCC。候选池排除三类：mlp_groupval（少用一片训练数据，有混杂）、
mlp_identity（优化对照）、mlp_fixed100（故意不正则的消融）。
同时报「全 10 队列取均值挑最优」的乐观参照，明确标注它偷看了测试队列。
输入：results/mlp_head_sweep*.json（按文件合并）；输出：results/head_selection.json。"""
import glob, json, os, sys
import numpy as np
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(BASE, "results")
sw = {}
for f in sorted(glob.glob(os.path.join(R, "mlp_head_sweep*.json"))):
    for e, d in json.load(open(f)).items():
        for c, v in d.items():
            sw.setdefault(e, {}).setdefault(c, {"n": v["n_train_sections"], "v": {}})["v"].update(v["variants"])
EXCL = {"mlp_groupval", "mlp_identity", "mlp_fixed100"}
FAM = {"ridge family": lambda h: h == "ridge" or h.startswith("ridge_"),
       "mlp family":   lambda h: h.startswith("mlp_"),
       "any head":     lambda h: True}
OUT = {}
for e in sorted(sw):
    cohs = sorted(sw[e])
    common = set.intersection(*[set(sw[e][c]["v"]) for c in cohs]) - EXCL
    te = {h: np.array([sw[e][c]["v"][h]["test_pcc"] for c in cohs]) for h in common}
    base = te["ridge"]
    print(f"\n{e}: {len(cohs)} 队列，候选头 {len(common)} 个；官方 ridge 均值 {base.mean():.4f}")
    OUT[e] = {"cohorts": cohs, "ridge_official_mean": float(base.mean()), "families": {}}
    for fam, ok in FAM.items():
        cand = sorted(h for h in common if ok(h))
        if len(cand) < 2: continue
        sel, val = [], []
        for i, c in enumerate(cohs):
            m = np.ones(len(cohs), bool); m[i] = False
            h = max(cand, key=lambda h: (te[h][m] - base[m]).mean())
            sel.append(h); val.append(float(te[h][i]))
        val = np.array(val)
        orc = max(cand, key=lambda h: te[h].mean())
        from collections import Counter
        cnt = Counter(sel)
        print(f"  {fam:<13s} 留一队列选头: 均值 {val.mean():.4f}  Δ vs 官方 {val.mean()-base.mean():+.4f}  高于官方 {(val>base).sum()}/{len(cohs)}  "
              f"选中: {dict(cnt)}   | 乐观参照(全队列挑最优,偷看) {orc} {te[orc].mean()-base.mean():+.4f}")
        OUT[e]["families"][fam] = {"loco_selected": dict(zip(cohs, sel)), "loco_test_pcc": dict(zip(cohs, val.tolist())),
                                   "loco_mean": float(val.mean()), "loco_delta_vs_ridge": float(val.mean() - base.mean()),
                                   "cohorts_above_ridge": int((val > base).sum()), "optimistic_best": orc,
                                   "optimistic_delta": float(te[orc].mean() - base.mean()), "candidates": cand}
json.dump(OUT, open(os.path.join(R, "head_selection.json"), "w"), indent=1)
print("\n已存 results/head_selection.json")

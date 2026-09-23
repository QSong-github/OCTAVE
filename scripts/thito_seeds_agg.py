# -*- coding: utf-8 -*-
"""THItoGene 三次运行（原始未播种 + seed 1 + seed 2）的汇总。
输入：results/thitogene_hest.json（原始合并结果，样本→{cohort, folds, pcc}）与 results/thito_seeds/thitogene_{COH}_f{FOLD}_s{SEED}.json（逐折）。
输出：每次运行 72 样本的均值、队列级均值、三次的均值±标准差；与 57 个编码器 ridge 的样本均值比较（methods_vs_floor.json 的 _reference.encoder_mean）。"""
import json, glob, os, numpy as np
R = os.environ.get("OCTAVE_RESULTS", "results")
runs = {"original": json.load(open(f"{R}/thitogene_hest.json"))}
for seed in (1, 2):
    m = {}
    for f in glob.glob(f"{R}/thito_seeds/thitogene_*_s{seed}.json"): m.update(json.load(open(f)))
    runs[f"seed{seed}"] = m
ref = json.load(open(f"{R}/methods_vs_floor.json"))["_reference"]["encoder_mean"]; enc_means = np.array(list(ref.values()))
out = {}
for k, m in runs.items():
    v = np.array([x["pcc"] for x in m.values()]); coh = {}
    for s, x in m.items(): coh.setdefault(x["cohort"], []).append(x["pcc"])
    out[k] = dict(n_samples=len(v), mean=float(v.mean()) if len(v) else None, cohort_means={c: float(np.mean(p)) for c, p in coh.items()},
                  n_encoders_above=int((enc_means > v.mean()).sum()) if len(v) else None)
    print(f"{k:9s} n={len(v):2d} 样本均值 {v.mean() if len(v) else float('nan'):.4f}  高于它的编码器数 {out[k]['n_encoders_above']}/{len(enc_means)}")
full = [k for k in out if out[k]["n_samples"] == 72]
if len(full) >= 2:
    mm = np.array([out[k]["mean"] for k in full]); print(f"完整运行 {len(full)} 次：均值 {mm.mean():.4f} ± {mm.std(ddof=1):.4f}（范围 {mm.min():.4f}–{mm.max():.4f}）")
    out["summary"] = dict(runs=full, mean=float(mm.mean()), sd=float(mm.std(ddof=1)), min=float(mm.min()), max=float(mm.max()))
    # 逐样本的运行间差异
    common = set.intersection(*[set(runs[k]) for k in full]); d = np.array([[runs[k][s]["pcc"] for k in full] for s in sorted(common)])
    print(f"逐样本运行间标准差 中位 {np.median(d.std(1, ddof=1)):.3f}，最大 {d.std(1, ddof=1).max():.3f}")
json.dump(out, open(f"{R}/thitogene_seeds.json", "w"), indent=1)

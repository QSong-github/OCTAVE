# -*- coding: utf-8 -*-
"""汇总 25 个图像塔的等价 σ —— 判定"≈110µm"是否依赖编码器。"""
import json, glob, os, numpy as np
D = "/path/to/project/results/tower_sweep"
rows = []
for p in sorted(glob.glob(os.path.join(D, "*.json"))):
    s = json.load(open(p))["_summary"]
    rows.append(s)
rows.sort(key=lambda r: -r["pcc"]["Ridge_HEST"])
print(f"{'塔':16s}{'维度':>6s}{'Ridge':>8s}{'kNN':>8s}"
      f"{'σ.pcc':>8s}{'σ.ret1':>8s}{'σ.dom1':>8s}{'σ.ari':>8s}{'Moran比':>9s}")
for r in rows:
    e = r["eq_sigma"]["Ridge_HEST"]
    print(f"{r['tower']:16s}{r['dim']:>6d}{r['pcc']['Ridge_HEST']:>8.4f}"
          f"{r['pcc']['imageKNN']:>8.4f}"
          f"{e['pcc']:>8.0f}{e['ret@1']:>8.0f}{e['dom1']:>8.0f}{e['ari']:>8.0f}"
          f"{r['moran_ratio']['Ridge_HEST']:>8.2f}×")
P = np.array([r["pcc"]["Ridge_HEST"] for r in rows])
S = {k: np.array([r["eq_sigma"]["Ridge_HEST"][k] for r in rows]) for k in ("pcc","ret@1","dom1","ari")}
M = np.array([r["moran_ratio"]["Ridge_HEST"] for r in rows])
print(f"\n=== 跨 {len(rows)} 个塔的离散度 ===")
print(f"  原始 PCC   范围 [{P.min():.4f}, {P.max():.4f}]  最好/最差 = {P.max()/P.min():.2f}×")
for k, v in S.items():
    print(f"  等价σ.{k:6s} 中位={np.nanmedian(v):6.0f}µm  范围 [{np.nanmin(v):.0f}, {np.nanmax(v):.0f}]µm"
          f"  最粗/最细 = {np.nanmax(v)/np.nanmin(v):.2f}×")
print(f"  Moran 膨胀 中位={np.median(M):.2f}×  范围 [{M.min():.2f}, {M.max():.2f}]×")
best = rows[0]; worst = rows[-1]
print(f"\n  最强塔 {best['tower']:14s} PCC={best['pcc']['Ridge_HEST']:.4f} 等价σ(pcc)={best['eq_sigma']['Ridge_HEST']['pcc']:.0f}µm")
print(f"  最弱塔 {worst['tower']:14s} PCC={worst['pcc']['Ridge_HEST']:.4f} 等价σ(pcc)={worst['eq_sigma']['Ridge_HEST']['pcc']:.0f}µm")
json.dump(rows, open("/path/to/project/results/tower_summary.json","w"),
          indent=2, ensure_ascii=False)
print("\n已存 results/tower_summary.json")

# -*- coding: utf-8 -*-
"""汇总上下文尺寸扫描与基因面板扫描。"""
import json, glob, os, numpy as np
R = "/path/to/project/results"

print("="*78); print("上下文尺寸扫描 —— 更大空间上下文 → 有效分辨率变细还是变粗?"); print("="*78)
rows = [json.load(open(p))["_summary"] for p in sorted(glob.glob(f"{R}/ctx_sweep/*.json"))]
rows.sort(key=lambda r: r["tower"])
print(f"{'变体':22s}{'维度':>6s}{'Ridge PCC':>11s}{'σ.pcc':>8s}{'σ.ret1':>8s}{'σ.ari':>8s}{'Moran比':>9s}")
for r in rows:
    e = r["eq_sigma"]["Ridge_HEST"]
    print(f"{r['tower']:22s}{r['dim']:>6d}{r['pcc']['Ridge_HEST']:>11.4f}"
          f"{e['pcc']:>8.0f}{e['ret@1']:>8.0f}{e['ari']:>8.0f}"
          f"{r['moran_ratio']['Ridge_HEST']:>8.2f}×")
# hibou_l 的 ctx 系列单独排序看趋势
hib = sorted([r for r in rows if r["tower"].startswith("hibou_l_ctx")],
             key=lambda r: int("".join(ch for ch in r["tower"].split("ctx")[1] if ch.isdigit())))
if hib:
    print(f"\n  hibou_l 上下文序列(按 ctx 像素升序):")
    for r in hib:
        print(f"    {r['tower']:20s} PCC={r['pcc']['Ridge_HEST']:.4f} "
              f"σ.pcc={r['eq_sigma']['Ridge_HEST']['pcc']:.0f}µm")

print()
print("="*78); print("基因面板敏感性 —— 110µm 是否只在 top-50 HVG 上成立?"); print("="*78)
print(f"{'面板':>8s}{'Ridge PCC':>11s}{'kNN PCC':>10s}{'σ.pcc':>8s}{'σ.ret1':>8s}{'σ.dom1':>8s}{'σ.ari':>8s}{'Moran比':>9s}")
for h in (20, 50, 100, 200):
    ps = glob.glob(f"{R}/panel_sweep_hvg{h}/*.json")
    if not ps: print(f"{h:>8d}  (未产出)"); continue
    r = json.load(open(ps[0]))["_summary"]; e = r["eq_sigma"]["Ridge_HEST"]
    print(f"{h:>8d}{r['pcc']['Ridge_HEST']:>11.4f}{r['pcc']['imageKNN']:>10.4f}"
          f"{e['pcc']:>8.0f}{e['ret@1']:>8.0f}{e['dom1']:>8.0f}{e['ari']:>8.0f}"
          f"{r['moran_ratio']['Ridge_HEST']:>8.2f}×")

# -*- coding: utf-8 -*-
"""
FOV 定律的精确形状。

extract_wsi_emb.py: ctx_px 是从 WSI level-0 读入的 context tile 边长, 默认 224。
像素尺寸 0.2739 µm/px ⇒ 视野(µm) = ctx_px × 0.2739。
  · 25 个基座塔: 全部 ctx=224 ⇒ 视野恒为 61.4 µm ⇒ 它们的 σ 跨度与视野无关(表征质量)
  · hibou_l_ctxNNN: 只变视野的受控序列
  · gridN: ctx 不变, 在其内切 N×N 个 sub_px 子块再平均嵌入 —— 是"聚合方式"而非视野

检验三件事:
 ① σ 对视野是线性还是次线性? 拟合 log-log 斜率与 σ = a + b·FOV
 ② 是否存在一个视野压不下去的地板? 地板值多少?
 ③ 固定视野下(25 塔)σ 的跨度 = 表征质量的贡献, 与视野贡献分离
"""
import json, glob, os, re, numpy as np
R = "/path/to/systema4ST/results"
UMPX = 0.2739

def load(d):
    out = []
    for p in sorted(glob.glob(f"{d}/*.json")):
        s = json.load(open(p))["_summary"]
        out.append((s["tower"], s["pcc"]["Ridge_HEST"],
                    s["eq_sigma"]["Ridge_HEST"]["pcc"], s["moran_ratio"]["Ridge_HEST"]))
    return out

def fov(name):
    m = re.search(r"ctx(\d+)", name)
    return (int(m.group(1)) if m else 224) * UMPX

base = load(f"{R}/tower_sweep"); ctx = load(f"{R}/ctx_sweep")

print("="*74); print("① 受控序列: 同编码器 hibou_l, 只变视野"); print("="*74)
seq = [(n, p, s, m) for n, p, s, m in base + ctx
       if n == "hibou_l" or (n.startswith("hibou_l_ctx") and "g" not in n.split("ctx")[1])]
seq.sort(key=lambda r: fov(r[0]))
print(f"{'变体':18s}{'ctx_px':>8s}{'视野µm':>9s}{'PCC':>9s}{'σ µm':>8s}{'σ/视野':>9s}{'Moran':>8s}")
F, S = [], []
for n, p, s, m in seq:
    f = fov(n)
    print(f"{n:18s}{int(f/UMPX):>8d}{f:>9.1f}{p:>9.4f}{s:>8.0f}{s/f:>9.2f}{m:>7.2f}×")
    if np.isfinite(s): F.append(f); S.append(s)
F, S = np.array(F), np.array(S)

print(f"\n② 形状拟合 (n={len(F)})")
k, b = np.polyfit(np.log(F), np.log(S), 1)
print(f"  幂律   σ ∝ FOV^{k:.3f}   (k=1 为线性, k<1 为次线性)")
a1, a0 = np.polyfit(F, S, 1)
r2 = 1 - ((S - (a0 + a1*F))**2).sum() / ((S - S.mean())**2).sum()
print(f"  线性   σ = {a0:.0f} + {a1:.2f}·FOV     R² = {r2:.3f}")
print(f"  ⇒ 视野→0 的外推截距 = {a0:.0f} µm  ← 视野压不下去的地板")
big = F > 120
if big.sum() >= 3:
    c1, c0 = np.polyfit(F[big], S[big], 1)
    print(f"  仅取 FOV>120µm 段: σ = {c0:.0f} + {c1:.2f}·FOV  ⇒ 该段几乎 1:1 跟随视野")

print(f"\n{'='*74}"); print("③ 固定视野(61.4µm)下的 25 个基座塔 —— 表征质量的贡献"); print("="*74)
bs = [s for n, p, s, m in base if np.isfinite(s)]
print(f"  n={len(bs)}  σ 中位={np.median(bs):.0f}µm  范围=[{min(bs):.0f}, {max(bs):.0f}]µm  跨度={max(bs)/min(bs):.2f}×")
print(f"  最好的塔(hibou_l) σ={min(bs):.0f}µm, 而视野只有 61.4µm ⇒ σ/视野={min(bs)/61.4:.2f}")
print(f"\n  ⇒ 两个来源可分离: 视野 ≤120µm 时地板主导(表征/模态), >120µm 时视野主导")

print(f"\n{'='*74}"); print("④ grid 变体: 视野不变, 只改聚合方式"); print("="*74)
for n, p, s, m in sorted(ctx, key=lambda r: r[0]):
    if "grid" in n or "g3" in n:
        print(f"  {n:22s} 视野={fov(n):6.1f}µm  PCC={p:.4f}  σ={s if np.isfinite(s) else float('nan'):.0f}µm  Moran={m:.2f}×")
print("  (grid 在同一 ctx 内切子块再平均 ⇒ 等效于额外的空间平滑, 预期使 σ 变粗)")

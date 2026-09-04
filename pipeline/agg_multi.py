import json, glob, os
import numpy as np

rows = {}   # sample -> {bin: (pcc, sigma, flag)}
for f in glob.glob("results/xenium/*.json"):
    d = json.load(open(f))
    rows.setdefault(d["name"], {})[16] = (d["pcc"], d["eq_sigma"], d.get("eq_flag"))
for f in glob.glob("results/xenium_multi/*.json"):
    d = json.load(open(f))
    n = d["name"]
    base, b = n.rsplit("_bin", 1)
    rows.setdefault(base, {})[int(b)] = (d["pcc"], d["eq_sigma"], d.get("eq_flag"))

BINS = [8, 16, 32, 64]
full = {k: v for k, v in rows.items() if all(b in v for b in BINS)}
print(f"四个尺度齐全的样本: {len(full)}/{len(rows)}\n")

hdr = "样本".ljust(42)
for b in BINS: hdr += f"σ@{b}µm".rjust(10)
print(hdr); print("-" * (42 + 10 * len(BINS)))
for k in sorted(full):
    line = k[:40].ljust(42)
    for b in BINS:
        s = full[k][b][1]
        line += (f"{s:.0f}" if isinstance(s, (int, float)) else str(s)).rjust(10)
    print(line)

print()
for b in BINS:
    sig = [full[k][b][1] for k in full if isinstance(full[k][b][1], (int, float))]
    pcc = [full[k][b][0] for k in full if isinstance(full[k][b][0], (int, float))]
    if sig:
        print(f"分箱 {b:2d}µm   σ 中位={np.median(sig):6.1f}µm  范围=[{min(sig):.0f},{max(sig):.0f}]"
              f"   PCC 中位={np.median(pcc):.4f}   n={len(sig)}")

# 关键判据：σ 与分箱边长的比值。若 σ 纯粹是栅格产物，该比值应恒定。
print("\nσ / 分箱边长（若 σ 只是栅格的产物，此列应为常数）")
for b in BINS:
    r = [full[k][b][1] / b for k in full if isinstance(full[k][b][1], (int, float))]
    if r: print(f"  {b:2d}µm: 中位 {np.median(r):5.2f}×   范围 [{min(r):.2f}, {max(r):.2f}]")

print("\nσ 相对 16µm 基准的倍数（同一张片内比较）")
for b in BINS:
    if b == 16: continue
    r = [full[k][b][1] / full[k][16][1] for k in full
         if isinstance(full[k][b][1], (int, float)) and isinstance(full[k][16][1], (int, float)) and full[k][16][1]]
    if r: print(f"  {b:2d}µm / 16µm: 中位 {np.median(r):5.2f}×   范围 [{min(r):.2f}, {max(r):.2f}]")

"""下游影响·第二批的 16 片聚合。

判据先于结果写定：
  · 共定位保持 —— 计单调下降的片数
  · 边界偏移   —— 物理单位；细尺度为 0 是栅格量化（下限 = bin 间距 16µm），非失效
  · 尺寸选择性 —— 斜率 = 召回(≥200µm) − 召回(<50µm)。
                  有分辨能力的预测应当「大结构更好找」⇒ 斜率 > 0；
                  斜率消失 = 预测已无法区分结构尺寸，这是分辨率损失的签名。
"""
import json, glob, os
import numpy as np

B = ["8", "16", "32", "64"]
SZ = ["<50µm", "50-100µm", "100-200µm", "≥200µm"]
rows = []
for f in sorted(glob.glob("results/downstream2_*.json")):
    d = json.load(open(f))
    sc = d.get("scales", {})
    if not all(b in sc for b in B):
        continue
    rows.append((os.path.basename(f)[len("downstream2_"):-5], sc))
print(f"四档齐全的片: {len(rows)}\n")

def col(key):
    return {b: [s[b][key] for _, s in rows if isinstance(s[b].get(key), (int, float))] for b in B}

for lab, key in [("共定位保持", "coloc_preserve"), ("热点 Jaccard", "hotspot_jaccard"),
                 ("边界偏移 (µm)", "boundary_shift_um")]:
    c = col(key)
    print(f"【{lab}】")
    print("       " + "".join(f"bin{b}".rjust(11) for b in B))
    print("  中位 " + "".join(f"{np.median(c[b]):11.3f}" for b in B))
    print("  范围 " + "".join(f"[{min(c[b]):.2f},{max(c[b]):.2f}]".rjust(11) for b in B))
    mono = sum(1 for _, s in rows if all(s[B[i]][key] >= s[B[i+1]][key] - 1e-9 for i in range(3)))
    if key != "boundary_shift_um":
        print(f"  单调下降的片: {mono}/{len(rows)}")
    print()

print("【尺寸选择性：召回率随热点尺寸】")
print("       " + "".join(s.rjust(11) for s in SZ) + "      斜率")
slopes = {}
for b in B:
    vals = {s: [] for s in SZ}
    sl = []
    for _, sc in rows:
        r = sc[b].get("hotspot_recall_by_size") or {}
        for s in SZ:
            if s in r: vals[s].append(r[s][0])
        if SZ[0] in r and SZ[-1] in r:
            sl.append(r[SZ[-1]][0] - r[SZ[0]][0])
    slopes[b] = sl
    print(f"bin{b:<3} " + "".join(f"{np.median(vals[s]):11.3f}" if vals[s] else "          —" for s in SZ)
          + (f"{np.median(sl):+10.3f}" if sl else "         —"))

print(f"\n斜率 = 召回(≥200µm) − 召回(<50µm)；>0 表示「大结构更好找」")
for b in B:
    sl = slopes[b]
    if not sl: continue
    pos = sum(1 for x in sl if x > 0.02)
    print(f"  bin{b:<3} 中位 {np.median(sl):+.3f}   斜率>0.02 的片: {pos}/{len(sl)}")
if slopes["16"] and slopes["64"]:
    print(f"\n⇒ bin16 斜率 {np.median(slopes['16']):+.3f} → bin64 斜率 {np.median(slopes['64']):+.3f}")

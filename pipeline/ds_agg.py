"""评测栅格效应：同一批预测，评测栅格跟随 vs 固定，结论是否反向。

§22 让评测栅格跟着预测栅格走，得到「调粗 → PCC 涨」；
本分析把四个分箱的预测统一映射回 16µm 栅格再评分，检验方向是否反转。
bin16 一行两种口径应当相同（该处两种协议重合）—— 这是内建的自洽锚点。
"""
import json, glob, os
import numpy as np

BINS = [8, 16, 32, 64]
rows, anchors = [], []
for f in sorted(glob.glob("results/downstream_*.json")):
    n = os.path.basename(f)[len("downstream_"):-5]
    d = json.load(open(f))
    sc = d.get("scales", {})
    if not all(str(b) in sc for b in BINS):
        continue
    fixed = [sc[str(b)]["pcc"] for b in BINS]
    own = []
    for b in BINS:
        p = f"results/xenium_multi/{n}_bin{b}.json" if b != 16 else f"results/xenium/{n}.json"
        own.append(json.load(open(p))["pcc"] if os.path.exists(p) else np.nan)
    if not np.isfinite(own).all():
        continue
    rows.append((n, own, fixed,
                 [sc[str(b)]["ari"] for b in BINS],
                 [sc[str(b)]["edge_ratio"] for b in BINS],
                 [sc[str(b)]["corr_len_um"] for b in BINS],
                 d.get("truth_corr_len_um")))
    anchors.append(abs(own[1] - fixed[1]))

print(f"四档齐全的样本: {len(rows)}\n")
print("自洽锚点（bin16 两种口径之差）: 中位 %.5f  最大 %.5f" % (np.median(anchors), max(anchors)))
print("  —— 该处两种协议重合，差值应≈0\n")

hdr = "样本".ljust(38) + "".join(f"b{b}".rjust(8) for b in BINS) + "  变化"
print("【评测栅格跟随预测栅格】(§22 口径)"); print(hdr); print("-" * len(hdr))
for n, own, *_ in rows:
    print(n[:36].ljust(38) + "".join(f"{v:8.4f}" for v in own)
          + f"  {100*(own[-1]/own[0]-1):+6.1f}%")
print("\n【评测栅格固定 16µm】"); print(hdr); print("-" * len(hdr))
for n, _, fx, *_ in rows:
    print(n[:36].ljust(38) + "".join(f"{v:8.4f}" for v in fx)
          + f"  {100*(fx[-1]/fx[0]-1):+6.1f}%")

so = [100*(r[1][-1]/r[1][0]-1) for r in rows]
sf = [100*(r[2][-1]/r[2][0]-1) for r in rows]
print(f"\n8→64µm 的 PCC 变化:")
print(f"  跟随栅格  中位 {np.median(so):+.1f}%   范围 [{min(so):+.1f}, {max(so):+.1f}]")
print(f"  固定栅格  中位 {np.median(sf):+.1f}%   范围 [{min(sf):+.1f}, {max(sf):+.1f}]")
print(f"  摆动      中位 {np.median(np.array(so)-np.array(sf)):.1f} 个百分点")
print(f"  方向反转的样本: {sum(1 for a,b in zip(so,sf) if a>0>b)}/{len(rows)}")

print(f"\n【下游后果，固定栅格】各档中位")
print("      " + "".join(f"b{b}".rjust(9) for b in BINS))
for lab, i in [("PCC", 2), ("ARI", 3), ("边界保真", 4), ("自相关长", 5)]:
    v = [np.median([r[i][k] for r in rows]) for k in range(4)]
    print(lab.ljust(8) + "".join(f"{x:9.3f}" for x in v))
tl = [r[6] for r in rows if r[6] and r[6] == r[6]]
if tl: print(f"真值自相关长中位 {np.median(tl):.0f}µm")
print("\n注：bin8 的边界保真被计数噪声压低（每 bin 约 1 细胞，真值梯度含大量随机涨落，")
print("    抬高了比值的分母）。粗端（bin64）的下降才是分辨率损失。该指标需按可达上限归一。")

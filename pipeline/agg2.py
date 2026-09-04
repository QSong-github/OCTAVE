import json, glob, numpy as np
rows = {}
for f in glob.glob("results/xenium/*.json"):
    d = json.load(open(f)); rows.setdefault(d["name"], {})[16] = d
for f in glob.glob("results/xenium_multi/*.json"):
    d = json.load(open(f)); b, n = d["name"].rsplit("_bin", 1)[::-1]
    rows.setdefault(d["name"].rsplit("_bin", 1)[0], {})[int(d["name"].rsplit("_bin", 1)[1])] = d

BINS = [8, 16, 32, 64]
print("eq_flag 分布（σ 是否可信）")
for b in BINS:
    fl = [rows[k][b].get("eq_flag") for k in rows if b in rows[k]]
    from collections import Counter
    print(f"  {b:2d}µm: {dict(Counter(fl))}")

def ok(d):
    s = d.get("eq_sigma")
    return isinstance(s, (int, float)) and s == s and d.get("eq_flag") == "ok"

print("\n只用 eq_flag=ok 的样本")
hdr = "分箱".ljust(8) + "n".rjust(4) + "σ中位".rjust(10) + "σ范围".rjust(16) + "PCC中位".rjust(11) + "σ/边长".rjust(9)
print(hdr); print("-" * len(hdr))
med = {}
for b in BINS:
    v = [rows[k][b] for k in rows if b in rows[k] and ok(rows[k][b])]
    if not v: continue
    s = [x["eq_sigma"] for x in v]; p = [x["pcc"] for x in v]
    med[b] = (np.median(s), np.median(p), len(s))
    print(f"{b:2d}µm".ljust(8) + str(len(s)).rjust(4) + f"{np.median(s):.1f}".rjust(10)
          + f"[{min(s):.0f},{max(s):.0f}]".rjust(16) + f"{np.median(p):.4f}".rjust(11)
          + f"{np.median(s)/b:.2f}x".rjust(9))

print("\n配对比较（同一张片，两个尺度都 ok）")
for b in [8, 32, 64]:
    pr = [(rows[k][b]["eq_sigma"], rows[k][16]["eq_sigma"], rows[k][b]["pcc"], rows[k][16]["pcc"])
          for k in rows if b in rows[k] and 16 in rows[k] and ok(rows[k][b]) and ok(rows[k][16])]
    if not pr: continue
    rs = [a/c for a, c, _, _ in pr]; rp = [d/e for _, _, d, e in pr]
    print(f"  {b:2d}µm vs 16µm  n={len(pr)}  σ比 中位{np.median(rs):.2f}x  PCC比 中位{np.median(rp):.3f}x")

if 8 in med and 32 in med:
    print(f"\n分箱 8→32µm（4x 变粗）：σ {med[8][0]:.1f}→{med[32][0]:.1f}µm 仅 {med[32][0]/med[8][0]:.2f}x，"
          f"而 PCC {med[8][1]:.4f}→{med[32][1]:.4f} 涨 {(med[32][1]/med[8][1]-1)*100:.1f}%")

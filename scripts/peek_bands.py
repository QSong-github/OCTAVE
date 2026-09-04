import json, glob
import numpy as np
fs = sorted(glob.glob("results/blocks_xen_bands/*.json"))
print(f"已完成 {len(fs)} 个区域\n")
for f in fs:
    d = json.load(open(f))
    sig = {int(k): v for k, v in d["sigma_um"].items()}
    R = d["pred"]
    ts = sorted(int(t) for t in R["ridge"]["bands"])
    ov = 100 * (R["ridge"]["pcc"] - R["dom20"]["pcc"]) / R["ridge"]["pcc"]
    print(f"{d['name']}   整体缺口 {ov:.1f}%")
    print(f"{'sigma(um)':>10s} {'ridge':>8s} {'blocks':>8s} {'缺口%':>8s}")
    gaps = []
    for t in ts:
        r = R["ridge"]["bands"][str(t)]
        b = R["dom20"]["bands"][str(t)]
        g = 100 * (r - b) / r if r > 0 else float("nan")
        gaps.append(g)
        print(f"{sig[t]:>10.0f} {r:>8.4f} {b:>8.4f} {g:>7.1f}%")
    fin = [g for g in gaps if np.isfinite(g)]
    dec = all(fin[i] >= fin[i + 1] - 1e-9 for i in range(len(fin) - 1))
    print(f"  ⇒ 缺口从细到粗是否单调下降: {'是' if dec else '否'}"
          f"（细端 {fin[0]:.1f}% → 粗端 {fin[-1]:.1f}%）\n")

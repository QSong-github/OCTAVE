import json, glob
import numpy as np
fs = sorted(glob.glob("results/blocks_xen_bands/*.json"))
print(f"已完成 {len(fs)} 个区域\n")
print(f"{'区域':<34s}{'标量缺口':>9s}{'最细带':>8s}{'最粗带':>8s}"
      f"{'标量≈第几档':>12s}{'单调?':>7s}")
for f in fs:
    d = json.load(open(f))
    sig = {int(k): v for k, v in d["sigma_um"].items()}
    R = d["pred"]
    ts = sorted(int(t) for t in R["ridge"]["bands"])
    ov = 100 * (R["ridge"]["pcc"] - R["dom20"]["pcc"]) / R["ridge"]["pcc"]
    g = []
    for t in ts:
        r, b = R["ridge"]["bands"][str(t)], R["dom20"]["bands"][str(t)]
        g.append(100 * (r - b) / r if r > 0 else np.nan)
    g = np.array(g)
    # 标量缺口最接近哪一档？
    j = int(np.nanargmin(np.abs(g - ov)))
    dec = all(g[i] >= g[i + 1] - 1e-9 for i in range(len(g) - 1))
    print(f"{d['name'][:33]:<34s}{ov:>8.1f}%{g[0]:>7.1f}%{g[-1]:>7.1f}%"
          f"{('σ=%.0fµm(%d/%d)' % (sig[ts[j]], j + 1, len(ts))):>12s}"
          f"{('是' if dec else '否'):>7s}")

"""在已存的 band_pcc 曲线上插值求 ER(τ)，不必重跑 15 个塔。
检查点按 t 的 2 次幂走 ⇒ 相邻 σ 差 √2≈1.41×，直接取离散检查点会把小于该步长的塔间差异量化掉。"""
import json, glob
import numpy as np

TAUS = [0.1, 0.2, 0.3]
out = []
for f in sorted(glob.glob("results/hest_effres_*.json")):
    d = json.load(open(f))
    sig = {int(k): v for k, v in d["sigma_um"].items()}
    cur = {int(k): v for k, v in d["band_pcc"].items()}
    xs = sorted(sig, key=lambda c: sig[c])
    floor = sig[xs[0]]
    er, disc, cens = {}, {}, {}
    for tau in TAUS:
        hit = [sig[c] for c in xs if cur.get(c, -1) >= tau]
        disc[str(tau)] = float(min(hit)) if hit else None
        val = None
        for i, c in enumerate(xs):
            if cur.get(c, -1) >= tau:
                if i == 0:
                    val = float(sig[c])
                else:
                    p0, p1 = cur[xs[i - 1]], cur[c]
                    s0, s1 = np.log(sig[xs[i - 1]]), np.log(sig[c])
                    w = 0.0 if p1 == p0 else (tau - p0) / (p1 - p0)
                    val = float(np.exp(s0 + w * (s1 - s0)))
                break
        er[str(tau)] = val
        cens[str(tau)] = val is not None and abs(val - floor) < 1e-6
    d["eff_res_um_discrete"] = disc
    d["eff_res_um"] = er
    d["censored_at_floor"] = cens
    d["sigma_floor_um"] = float(floor)
    json.dump(d, open(f, "w"), indent=2, ensure_ascii=False)
    out.append((d["encoder"], (d.get("pcc_check") or {}).get("official"),
                er["0.1"], er["0.2"], er["0.3"], cens["0.1"]))

out.sort(key=lambda r: -(r[1] or 0))
print("塔".ljust(15) + "PCC".rjust(8) + "ER(.1)".rjust(9) + "ER(.2)".rjust(9) + "ER(.3)".rjust(9) + "  .1删失")
print("-" * 62)
for e, p, a, b, c, cz in out:
    fmt = lambda v: (f"{v:.1f}" if isinstance(v, (int, float)) else "—")
    print(e.ljust(15) + (f"{p:.4f}" if p else "—").rjust(8)
          + fmt(a).rjust(9) + fmt(b).rjust(9) + fmt(c).rjust(9) + ("  是" if cz else "  否"))

for i, lab in [(3, "ER(0.2)"), (4, "ER(0.3)")]:
    v = [r[i] for r in out if isinstance(r[i], (int, float))]
    if v:
        print(f"\n{lab}: 最好 {min(v):.1f}µm  最差 {max(v):.1f}µm  跨度 {max(v)/min(v):.2f}x"
              f"   (测量步长 √2=1.41x)")
pc = [r[1] for r in out if r[1]]
print(f"PCC   : 最好 {max(pc):.4f}  最差 {min(pc):.4f}  跨度 {max(pc)/min(pc):.2f}x")

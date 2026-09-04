import glob, json, os, numpy as np
from math import comb
R = "/blue/qsong1/wang.qing/systema4ST/results"
BINS = [8, 16, 32, 64]

def J(p):
    try: return json.load(open(p))
    except Exception: return None

def sign_p(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def spec(n):
    return n.rsplit("_", 1)[0] if n.startswith("Human_Breast_Biomarkers") else n

# ---- 1. 评测栅格 跟随 vs 固定 ----
rows = []
for f in sorted(glob.glob(f"{R}/downstream_*.json")):
    n = os.path.basename(f)[len("downstream_"):-5]
    d = J(f)
    if not d or not all(str(b) in d.get("scales", {}) for b in BINS): continue
    fx = [d["scales"][str(b)]["pcc"] for b in BINS]
    ow = [ (J(f"{R}/xenium_multi/{n}_bin{b}.json") if b != 16 else J(f"{R}/xenium/{n}.json") or {}).get("pcc", np.nan)
           if (J(f"{R}/xenium_multi/{n}_bin{b}.json") if b != 16 else J(f"{R}/xenium/{n}.json")) else np.nan
           for b in BINS ]
    if not np.isfinite(ow).all(): continue
    rows.append(dict(name=n, specimen=spec(n),
                     follow_pct=100*(ow[-1]/ow[0]-1), fixed_pct=100*(fx[-1]/fx[0]-1),
                     follow_curve=list(map(float, ow)), fixed_curve=list(map(float, fx))))
so = np.array([r["follow_pct"] for r in rows]); sf = np.array([r["fixed_pct"] for r in rows])
S = {}
for r in rows: S.setdefault(r["specimen"], []).append(r)
sp = {k: dict(follow_pct=float(np.median([x["follow_pct"] for x in v])),
              fixed_pct=float(np.median([x["fixed_pct"] for x in v])), n_regions=len(v)) for k, v in S.items()}
nrev_sp = sum(1 for v in sp.values() if v["follow_pct"] > 0 > v["fixed_pct"])
grid = dict(bins_um=BINS, n_regions=len(rows), n_specimens=len(sp),
    follow_median=float(np.median(so)), follow_lo=float(so.min()), follow_hi=float(so.max()),
    fixed_median=float(np.median(sf)), fixed_lo=float(sf.min()), fixed_hi=float(sf.max()),
    swing_pp=float(np.median(so) - np.median(sf)),
    n_regions_reverse=int(((so > 0) & (sf < 0)).sum()), n_specimens_reverse=nrev_sp,
    P_specimen=sign_p(nrev_sp, len(sp)), by_region=rows, by_specimen=sp)

# ---- 2. 块预言机占已训模型分数的比例（Xenium）----
sh = []
for f in sorted(glob.glob(f"{R}/blocks_xen/*.json")):
    d = J(f); P = d["pred"]
    def pc(k):
        v = P[k]
        return v["pcc"] if isinstance(v, dict) and "pcc" in v else (v if isinstance(v, float) else None)
    a, b = pc("dom20"), pc("ridge")
    if a and b: sh.append(dict(name=d["name"], specimen=spec(d["name"]), share=100*a/b))
    else: print("KEYS", d["name"], {k: (list(v.keys())[:6] if isinstance(v, dict) else type(v).__name__) for k, v in P.items()})
if sh:
    T = {}
    for r in sh: T.setdefault(r["specimen"], []).append(r["share"])
    ms = np.array([np.median(v) for v in T.values()])
    share = dict(n_regions=len(sh), n_specimens=len(T), by_region=sh,
                 by_specimen={k: float(np.median(v)) for k, v in T.items()},
                 median=float(np.median(ms)), lo=float(ms.min()), hi=float(ms.max()))
else:
    share = None

out = dict(grid_reversal=grid, block_oracle_share_xenium=share)
json.dump(out, open(f"{R}/grid_reversal.json", "w"), indent=1)
print("GRID  follow %+.1f%% [%+.1f, %+.1f]  fixed %+.1f%% [%+.1f, %+.1f]  swing %.1f pp  rev %d/%d regions %d/%d spec  P=%.4f"
      % (grid["follow_median"], grid["follow_lo"], grid["follow_hi"], grid["fixed_median"], grid["fixed_lo"], grid["fixed_hi"],
         grid["swing_pp"], grid["n_regions_reverse"], grid["n_regions"], grid["n_specimens_reverse"], grid["n_specimens"], grid["P_specimen"]))
if share: print("SHARE median %.1f%% [%.1f, %.1f] over %d specimens" % (share["median"], share["lo"], share["hi"], share["n_specimens"]))

# -*- coding: utf-8 -*-
"""复核意见 Q4：Xenium 各区域、各扩散带的拆半可靠性。对 16 µm 分箱 counts 做二项拆半（每个 count c 抽 h~Binom(c,1/2)），
两半各自 log1p，用与主流水线相同的算子（k=8、29 µm、lazy 1/2）做同样的带分解，逐基因算两半的带内相关 c_half，
Spearman–Brown 校正到全深度 c_full = 2c/(1+c)。基因集与主流水线一致（全场方差最高的 200 个）。"""
import argparse, json, os, sys, numpy as np, anndata as ad
from scipy import sparse
sys.path.insert(0, "/path/to/project/src")
from per_gene_xen import build_operator, per_gene_pcc
PREP = "/path/to/project/data/prepped_xen"; OUTD = "/path/to/project/results/xen_band_ceiling"
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--ngene", type=int, default=200)
ap.add_argument("--tmax", type=int, default=2048); ap.add_argument("--reps", type=int, default=3); a = ap.parse_args()
cps = [1]
while cps[-1] < a.tmax: cps.append(cps[-1] * 2)
A_ = ad.read_h5ad(f"{PREP}/{a.name}_bin16.h5ad"); px = float(A_.uns["px_per_um"]); xy = np.asarray(A_.obsm["pxl"], np.float64) / px
C = sparse.csr_matrix(A_.X); Yfull = np.log1p(np.asarray(C.todense(), np.float32))
gidx = np.argsort(-Yfull.var(0))[:a.ngene]; Cg = np.asarray(C[:, gidx].todense()); Cg = np.rint(Cg).astype(np.int64)
W = build_operator(xy)
def all_bands(M):
    out, prev, low, t = {}, M.copy(), M.copy(), 0
    for cp in cps:
        while t < cp: prev = W @ prev; t += 1
        out[cp] = np.asarray(low - prev, np.float32); low = prev.copy()
    return out
res = {str(cp): [] for cp in cps}; rel_share = {str(cp): [] for cp in cps}; scalar = []
rng = np.random.default_rng(0)
for r in range(a.reps):
    H = rng.binomial(Cg, 0.5); Ya = np.log1p(H.astype(np.float32)); Yb = np.log1p((Cg - H).astype(np.float32))
    scalar.append(float(np.nanmean(per_gene_pcc(Ya, Yb))))
    Ba, Bb = all_bands(Ya), all_bands(Yb)
    for cp in cps:
        pg = per_gene_pcc(Ba[cp], Bb[cp]); res[str(cp)].append(float(np.nanmean(pg)))
        # 可重复的带方差份额：两半带场的协方差 / 全深度带场方差
        Bf = 0.5 * (Ba[cp] + Bb[cp]) * 2  # 近似全深度带场：两半之和的 log1p 不可加，改用协方差口径
        ca = Ba[cp] - Ba[cp].mean(0); cb = Bb[cp] - Bb[cp].mean(0)
        cov = (ca * cb).mean(0); va = ca.var(0); vb = cb.var(0)
        rel_share[str(cp)].append(float(np.mean(cov / (np.sqrt(va * vb) + 1e-12))))
c_half = {k: float(np.mean(v)) for k, v in res.items()}; c_full = {k: 2 * v / (1 + v) for k, v in c_half.items()}
out = {"name": a.name, "n_bins": int(len(xy)), "n_genes": int(len(gidx)), "reps": a.reps, "cps": cps,
       "c_half": c_half, "c_full": c_full, "scalar_c_half": float(np.mean(scalar)), "scalar_c_full": float(2 * np.mean(scalar) / (1 + np.mean(scalar)))}
os.makedirs(OUTD, exist_ok=True); json.dump(out, open(f"{OUTD}/{a.name}.json", "w"), indent=1)
print(f"[{a.name}] scalar c_full={out['scalar_c_full']:.3f} | band1 c_half={c_half['1']:.3f} c_full={c_full['1']:.3f} | band2 c_full={c_full['2']:.3f} | coarsest c_full={c_full[str(cps[-1])]:.3f}", flush=True)

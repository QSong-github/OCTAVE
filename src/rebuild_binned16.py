# -*- coding: utf-8 -*-
"""Rebuild the parent project's data/binned_16um.h5ad inside this project .

The original binned file is no longer available. Its recipe is recorded in this project.
- Bins: the 10x square_016um bins of P2, then P5, in 10x order, as hd_prep.py writes them. The Hibou-L
  features computed on these bins match the parent's emb_hibou_l_P2.npy bit for bit.
- X: log1p of counts per 10k, with totals over all 18,085 genes (prep_bin).
- Genes: the 200 keys of results/per_gene_hibou_l.json. per_gene.py took all 200 genes of the parent file.
Check: per-gene mean and zero fraction against results/per_gene_hibou_l.json, which averages each over the two sections.
"""
import json, os, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
R = "/path/to/project"
OUT = f"{R}/data/vhd_legacy"
SLIDES = ["Visium_HD_Human_Colon_Cancer_P2", "Visium_HD_Human_Colon_Cancer_P5"]
PG = json.load(open(f"{R}/results/per_gene_hibou_l.json"))
panel = sorted(PG); assert len(panel) == 200


def norm_log(counts_csr):
    """prep_bin: counts per 10k over all genes, then log1p."""
    tot = np.asarray(counts_csr.sum(1)).ravel()
    tot[tot == 0] = 1.0
    Y = counts_csr.multiply((1e4 / tot)[:, None]).tocsr()
    Y.data = np.log1p(Y.data)
    return Y


Xs, pxl, sid, stats = [], [], [], {}
for s in SLIDES:
    a = ad.read_h5ad(f"{R}/data/prepped/{s}_16um.h5ad")
    names = np.asarray(a.var_names).astype(str)
    gi = {g: i for i, g in enumerate(names)}
    miss = [g for g in panel if g not in gi]; assert not miss, (s, miss)
    dup = [g for g in panel if f"{g}-1" in gi]
    print(f"[{s}] n={a.n_obs} genes={a.n_vars}; panel genes with a duplicated symbol: {dup}", flush=True)
    Y = np.asarray(norm_log(sparse.csr_matrix(a.X))[:, [gi[g] for g in panel]].todense(), np.float32)
    stats[s] = (Y.mean(0), (Y == 0).mean(0))
    Xs.append(Y); pxl.append(np.asarray(a.obsm["pxl"], np.float64)); sid += [s] * a.n_obs

b = ad.AnnData(X=np.concatenate(Xs), obs=pd.DataFrame({"slide_id": sid}, index=[str(i) for i in range(len(sid))]),
               var=pd.DataFrame({"gene": panel}, index=panel))
b.obsm["pxl"] = np.concatenate(pxl)
os.makedirs(OUT, exist_ok=True); b.write_h5ad(f"{OUT}/binned_16um.h5ad")
print(f"wrote {OUT}/binned_16um.h5ad {b.shape}", flush=True)

for k, j in (("mean", 0), ("zero", 1)):
    new = np.array([(stats[SLIDES[0]][j][i] + stats[SLIDES[1]][j][i]) / 2 for i in range(200)])
    old = np.array([PG[g][k] for g in panel]); d = np.abs(new - old); w = np.argsort(-d)[:5]
    print(f"{k}: max abs diff {d.max():.2e}, median {np.median(d):.2e}; worst " + ", ".join(f"{panel[i]} {old[i]:.5f}/{new[i]:.5f}" for i in w), flush=True)

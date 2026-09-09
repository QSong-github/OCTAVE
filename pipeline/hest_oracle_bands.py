# -*- coding: utf-8 -*-
"""第一份审稿意见：在 HEST（~100 µm 间距）上做 OCTAVE 式的 Δ_fine/Δ_scalar。
本脚本只补 domain oracle 的逐样本逐带 PCC；ridge 的逐带 PCC 已在 results/hest_effres_ps_{enc}.json（官方 α）。
算子、σ 标定、带分解与 hest_effres_ps.py 完全同一份代码（直接 import）；oracle 配方与 hest_blocks.py 逐参数一致。"""
import os, sys, json, numpy as np, anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
sys.path.insert(0, "/path/to/systema4ST/src")
import hest_effres_ps as H
B = getattr(H, "B", "/path/to/he2st/HEST/eval/bench_data"); EMB = getattr(H, "EMB", "/path/to/systema4ST/results/hest_emb")
LADDER = H.LADDER; R = "/path/to/systema4ST/results"
enc = sys.argv[1]; K = 20; tmax = 1024
cps = [1]
while cps[-1] < tmax: cps.append(cps[-1] * 2)
umpx = {k: v["umpx"] for k, v in json.load(open(LADDER)).items()}
def group_means(Y, lab):
    out = np.empty_like(Y)
    for c in np.unique(lab): m = lab == c; out[m] = Y[m].mean(0)
    return out
out = {}
for c in sorted(os.listdir(B)):
    if not os.path.isdir(os.path.join(B, c, "adata")): continue
    genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]
    for f in sorted(os.listdir(os.path.join(B, c, "adata"))):
        if not f.endswith(".h5ad"): continue
        sid = f[:-5]; ef = os.path.join(EMB, f"{sid}_{enc}.npz")
        if not os.path.exists(ef): continue
        z = np.load(ef, allow_pickle=True); X, bc = z["X"].astype(np.float32), z["bc"].tolist()
        a = ad.read_h5ad(os.path.join(B, c, "adata", f)); pos = {b: i for i, b in enumerate(a.obs_names.astype(str))}; sub = a[np.array([pos[b] for b in bc])]
        gi = [list(sub.var_names.astype(str)).index(g) for g in genes]
        Y = np.log1p(np.asarray(sub.X.todense() if sparse.issparse(sub.X) else sub.X, np.float64)[:, gi]).astype(np.float32)
        xy = np.asarray(sub.obsm["spatial"], np.float64) * umpx.get(sid, 1.0)
        n = Y.shape[0]; pc = PCA(n_components=min(50, X.shape[1], n - 1), random_state=0).fit_transform(X)
        lab = KMeans(n_clusters=min(K, n), n_init=4, random_state=0).fit_predict(pc); O = group_means(Y, lab)
        W, pitch, deg = H.build_operator(xy); sig = H.calibrate_sigma(W, xy, cps)
        M = np.concatenate([Y, O], 1).astype(np.float32); G = Y.shape[1]
        prev, low_prev, t = M.copy(), M.copy(), 0; bands = {}
        for cp in cps:
            while t < cp: prev = W @ prev; t += 1
            band = low_prev - prev; bands[str(cp)] = float(np.nanmean(H.per_gene_pcc(band[:, G:], band[:, :G]))); low_prev = prev.copy()
        out[sid] = {"cohort": c, "n": int(n), "oracle_pcc": float(np.nanmean(H.per_gene_pcc(O, Y))), "band_pcc_oracle": bands,
                    "sigma_um": {str(cp): float(sig[cp]) for cp in cps}, "pitch_um": float(pitch)}
        print(f"  {sid} n={n} oracle={out[sid]['oracle_pcc']:.4f} β1={bands['1']:.4f} σ1={sig[1]:.1f}µm", flush=True)
o = f"{R}/hest_oracle_bands_{enc}.json"; json.dump({"encoder": enc, "K": K, "cps": cps, "samples": out}, open(o, "w"), indent=1); print(f"{enc}: {len(out)} 样本 -> {o}")

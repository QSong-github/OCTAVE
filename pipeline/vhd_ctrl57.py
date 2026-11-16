# -*- coding: utf-8 -*-
"""Table 2 on one encoder set: Visium HD (P2, P5) partition controls for all 57 encoders , a copy of xen_ctrl57.py.
Same protocol as xen_controls.py and xen_inductive.py (K=20, top-200 genes, 16x16 spatial blocks, same fold thresholds and seeds).
--mode cheap: image oracle (check against blocks_xen_bands_<enc> dom20), random matched-size partition,
              learnable domain-level predictor (image partition, means from training blocks) -> results/xen_ctrl57/<tower>/<region>.json
--mode ind:   partition and means both learned from training blocks (PCA(50)+KMeans(20) refitted on each fold's training blocks)
              -> results/xen_ind57/<tower>/<region>.json
The coordinate partition does not use the image, so its PCC is taken from results/xen_controls."""
import argparse, json, os, sys, time, numpy as np, anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
ROOT = "/path/to/project"
sys.path.insert(0, f"{ROOT}/src")
from per_gene_xen import per_gene_pcc
PREP = f"{ROOT}/data/prepped_vhd"; EMB = f"{ROOT}/results/emb_vhd"
TOWERS = ("hibou_l biomedclip ciga clip_vitl14 conch_v1 conch_v15 ctranspath dinov2_base dinov2_giant dinov2_large dinov3_vitb16 dinov3_vith16 "
          "dinov3_vitl16 distillpath_is16 distillpath_ks16 genbio_pathfm gigapath gigapath_flash gpfm h0_mini hibou_b hoptimus0 hoptimus1 "
          "kaiko_vitb16 kaiko_vitb8 kaiko_vitl14 kaiko_vits16 keep litefm litefm_l litefm_s litevirchow2 lunit_r50_bt lunit_r50_moco "
          "lunit_r50_swav lunit_vits16 lunit_vits8 mascaret midnight12k mstar musk omiclip openmidnight path_foundation pathgen_clip "
          "pathryoshka_b phaet phikon phikon_v2 plip quiltnet retccl siglip2 uni_v1 uni_v2 virchow virchow2").split()
assert len(TOWERS) == 57 and len(set(TOWERS)) == 57
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--mode", choices=["cheap", "ind"], required=True)
ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--nchunk", type=int, default=1); ap.add_argument("--ngene", type=int, default=200); a = ap.parse_args()
OUTD = f"{ROOT}/results/" + ("vhd_ctrl57" if a.mode == "cheap" else "vhd_ind57")
towers = TOWERS[a.chunk::a.nchunk]
todo = [t for t in towers if not os.path.exists(f"{OUTD}/{t}/{a.name}.json") and os.path.exists(f"{EMB}/emb_{t}_{a.name}.npy")]   # skip encoders whose embedding is missing
if not todo: print(f"[{a.name}] chunk {a.chunk}: all done"); sys.exit(0)
A_ = ad.read_h5ad(f"{PREP}/{a.name}_bin16.h5ad"); xy = np.asarray(A_.obsm["pxl"], np.float64) / float(A_.uns["px_per_um"])
Y_all = np.log1p(np.asarray(sparse.csr_matrix(A_.X).todense(), np.float32)); gidx = np.argsort(-Y_all.var(0))[:a.ngene]; Y = Y_all[:, gidx]; del Y_all, A_
def folds(xy, grid=16):
    qx = np.quantile(xy[:, 0], np.linspace(0, 1, grid + 1)); qx[-1] += 1
    qy = np.quantile(xy[:, 1], np.linspace(0, 1, grid + 1)); qy[-1] += 1
    return (np.searchsorted(qx, xy[:, 0], "right") - 1) * grid + (np.searchsorted(qy, xy[:, 1], "right") - 1)
fold = folds(xy)
FOLDS = [f for f in np.unique(fold) if (fold == f).sum() >= 20 and (fold != f).sum() >= 2000]   # same thresholds as block_cv_predict
ok = np.isin(fold, FOLDS)                                                                       # bins where the ridge prediction is finite
def score(Pm): return float(np.nanmean(per_gene_pcc(Pm[ok], Y[ok])))
def group_means(lab):
    out = np.zeros_like(Y)
    for u in np.unique(lab): m = lab == u; out[m] = Y[m].mean(0)
    return out
def train_only_means(lab):
    Pt = np.full_like(Y, np.nan)
    for f in FOLDS:
        te = fold == f
        mu = {u: Y[(~te) & (lab == u)].mean(0) for u in np.unique(lab[te]) if ((~te) & (lab == u)).any()}
        g = Y[~te].mean(0); Pt[te] = np.stack([mu.get(u, g) for u in lab[te]])
    return Pt
def inductive(X):
    P = np.full_like(Y, np.nan)
    for f in FOLDS:
        te = fold == f; tr = ~te; Xtr = X[tr]; Ytr = Y[tr]
        pca = PCA(n_components=50, random_state=0).fit(Xtr); km = KMeans(n_clusters=20, n_init=4, random_state=0).fit(pca.transform(Xtr))
        ltr = km.labels_; lte = km.predict(pca.transform(X[te]))
        mu = np.stack([Ytr[ltr == u].mean(0) if (ltr == u).any() else Ytr.mean(0) for u in range(20)]); P[te] = mu[lte]
    return P
for t in todo:
    t0 = time.time()
    X = np.nan_to_num(np.load(f"{EMB}/emb_{t}_{a.name}.npy").astype(np.float32)); assert X.shape[0] == Y.shape[0], (t, X.shape, Y.shape)
    res = dict(name=a.name, tower=t, n=int(len(Y)), n_ok=int(ok.sum()), n_folds=len(FOLDS))
    if a.mode == "cheap":
        lab = KMeans(n_clusters=20, n_init=4, random_state=0).fit_predict(PCA(n_components=50, random_state=0).fit_transform(X))
        lab_rand = np.random.default_rng(0).permutation(lab)   # first draw of rng(0), as for K=20 in xen_controls.py
        lab_xy = KMeans(n_clusters=20, n_init=4, random_state=0).fit_predict(xy)   # coordinate partition, as in xen_controls.py
        res.update(oracle_image=score(group_means(lab)), oracle_random_matched=score(group_means(lab_rand)), trainonly_image=score(train_only_means(lab)), oracle_coord=score(group_means(lab_xy)))
    else:
        res.update(inductive=score(inductive(X)))
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(f"{OUTD}/{t}", exist_ok=True); json.dump(res, open(f"{OUTD}/{t}/{a.name}.json", "w"), indent=1)
    print(f"[{a.name}] {t}: " + ", ".join(f"{k} {v:.4f}" for k, v in res.items() if isinstance(v, float) and k != "seconds") + f" ({res['seconds']} s)", flush=True)

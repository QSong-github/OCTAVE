# -*- coding: utf-8 -*-
"""data for Figure 1 panel b, the 2 Visium HD windows of the earlier Figure 1 (results/repro_vhd/fig1_windows),
redrawn at full H&E resolution. Same fold ("half"), genes (the 50 with the highest variance among HEST's 200), ridge
(penalty 1e4) on frozen Hibou-L features, and domain oracle (K-means with K = 20 on 50 principal components of the
features, measured domain means), as in fig1_windows.py. Run with 32 threads so that K-means matches the earlier Figure 1.
Outputs in results/repro_vhd/fig1_v3: he_<TAG>.png (window at level 0, 1000 px), maps_<TAG>.npz (first principal
component of measured T, oracle D and model P on the 16 um bin grid of the window, and bands 1, 4, 7 of each),
numbers.json (window correlations as in the earlier Figure 1, per-gene PCC on the test half, band widths)."""
import os, sys, json
import numpy as np, anndata as ad
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
R = "/path/to/project"
sys.path.insert(0, f"{R}/src")
import st_encoder_bench as SEB, effres   # effres points SEB at the deleted parent project; overridden below
SEB.H5AD, SEB.EMBDIR = f"{R}/data/vhd_legacy/binned_16um.h5ad", f"{R}/data/vhd_legacy/emb"
import evaluate as E
from baselines import ridge_predict
from within_bench import make_split
from effres import PX_PER_UM, build_operator, calibrate_sigma
from ruler import group_means

OUT = f"{R}/results/repro_vhd/fig1_v3"; os.makedirs(OUT, exist_ok=True)
FOLD, BLOCK_UM, MARGIN_UM, HVG_POOL, HVG_EVAL, SIDE = "half", 320.0, 64.0, 200, 50, 900.0
SLIDES = {"P2": "Visium_HD_Human_Colon_Cancer_P2", "P5": "Visium_HD_Human_Colon_Cancer_P5"}
NUM = json.load(open(f"{R}/results/repro_vhd/fig1_windows/fig1_v2_numbers.json"))
TS = [0] + [2 ** i for i in range(7)]          # smoothing levels up to band 7
print(f"H5AD {SEB.H5AD}; OMP {os.environ.get('OMP_NUM_THREADS')}", flush=True)

a = ad.read_h5ad(SEB.H5AD)
X = np.nan_to_num(np.asarray(a.X, np.float32))
slide = a.obs["slide_id"].astype(str).values
pxl = np.asarray(a.obsm["pxl"], np.float64)
img = np.nan_to_num(np.concatenate([np.load(os.path.join(SEB.EMBDIR, "emb_hibou_l_P2.npy")),
                                    np.load(os.path.join(SEB.EMBDIR, "emb_hibou_l_P5.npy"))]).astype(np.float32))


def _r(u, v):
    u = u - u.mean(); v = v - v.mean()
    return float((u * v).sum() / (np.sqrt((u ** 2).sum() * (v ** 2).sum()) + 1e-12))


import openslide
out = {}
for TAG, SLIDE in SLIDES.items():
    m = np.where(slide == SLIDE)[0]
    xy = pxl[m] / PX_PER_UM[SLIDE]; Xs, IM = X[m], img[m]
    trm, tem = make_split(xy, FOLD, BLOCK_UM, MARGIN_UM)
    gidx = E.topk_hvg(Xs[trm], HVG_POOL)[-HVG_EVAL:]
    y_full = Xs[:, gidx]; yte = y_full[tem]
    RDG = ridge_predict(IM[trm], Xs[trm][:, gidx], IM[tem], 1e4)
    pc = PCA(n_components=50, random_state=0).fit_transform(IM)
    lab20 = KMeans(n_clusters=20, n_init=4, random_state=0).fit_predict(pc)
    DOM = group_means(y_full, lab20)[tem]
    pcc = {"ridge": float(E.per_gene_pcc(RDG, yte).mean()), "dom20": float(E.per_gene_pcc(DOM, yte).mean())}
    mu = yte.mean(0); Yc = yte - mu
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False); load = Vt[0]
    if (Yc @ load).mean() < 0: load = -load
    F = {"T": (Yc @ load).astype(np.float32), "P": ((RDG - mu) @ load).astype(np.float32), "D": ((DOM - mu) @ load).astype(np.float32)}
    xt = xy[tem]
    W = build_operator(xt, k=8, cut_um=29.0)
    sig = calibrate_sigma(W, xt, TS[1:])
    lev = {k: {} for k in F}; cur = {k: v.copy() for k, v in F.items()}; t = 0
    for tt in TS:
        while t < tt:
            cur = {k: W @ v for k, v in cur.items()}; t += 1
        for k in F: lev[k][tt] = cur[k].copy()
    band = {k: {i: lev[k][TS[i - 1]] - lev[k][TS[i]] for i in range(1, 8)} for k in F}
    x0, y0 = NUM[TAG]["best"]["x0"], NUM[TAG]["best"]["y0"]
    s_ = (xt[:, 0] >= x0) & (xt[:, 0] < x0 + SIDE) & (xt[:, 1] >= y0) & (xt[:, 1] < y0 + SIDE)
    r = dict(map_d=_r(F["D"][s_], F["T"][s_]), map_r=_r(F["P"][s_], F["T"][s_]),
             band_d=_r(band["D"][1][s_], band["T"][1][s_]), band_r=_r(band["P"][1][s_], band["T"][1][s_]))
    ref = NUM[TAG]["best"]
    print(f"[{TAG}] PCC ridge {pcc['ridge']:.4f} dom20 {pcc['dom20']:.4f} | window r: " +
          ", ".join(f"{k} {v:.3f} (reference {ref[k]:.3f})" for k, v in r.items()), flush=True)
    sub = xt[s_]; ix = np.round((sub[:, 0] - x0) / 16.0).astype(int); iy = np.round((sub[:, 1] - y0) / 16.0).astype(int)
    def grid(v):
        H = np.full((iy.max() + 1, ix.max() + 1), np.nan, np.float32); H[iy, ix] = v[s_]; return H
    arr = {}
    for k in F:
        arr[f"{k}_map"] = grid(F[k])
        for i in (1, 4, 7): arr[f"{k}_band{i}"] = grid(band[k][i])
    np.savez_compressed(f"{OUT}/maps_{TAG}.npz", **arr)
    sl = openslide.OpenSlide(f"{R}/data/visiumhd/{SLIDE}/{SLIDE}_PYRAMIDAL.tif")
    hp = pxl[m][tem][s_]; half = 8.0 * PX_PER_UM[SLIDE]
    bx0, by0, bx1, by1 = hp[:, 0].min() - half, hp[:, 1].min() - half, hp[:, 0].max() + half, hp[:, 1].max() + half
    w_, h_ = int(round(bx1 - bx0)), int(round(by1 - by0))
    sl.read_region((int(round(bx0)), int(round(by0))), 0, (w_, h_)).convert("RGB").resize(
        (1000, int(round(1000 * h_ / w_))), Image.LANCZOS).save(f"{OUT}/he_{TAG}.png")
    out[TAG] = dict(x0=x0, y0=y0, pcc=pcc, window_r=r, b214=ref, sigma_um={str(k): v for k, v in sig.items()},
                    grid=list(arr["T_map"].shape), n_bins=int(s_.sum()))
json.dump(out, open(f"{OUT}/numbers.json", "w"), indent=1)
print("done", flush=True)

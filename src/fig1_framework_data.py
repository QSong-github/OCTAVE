# -*- coding: utf-8 -*-
"""data for the framework figure (Figure 1), one 900 um window of Visium HD section P5.
The window, split, genes and model are those of the earlier Figure 1 (results/repro_vhd/fig1_windows): fold "half",
the 50 genes with the highest variance among HEST's 200, ridge (penalty 1e4) on frozen Hibou-L features trained on the
other half of the section. Outputs in results/repro_vhd/fig1_framework:
  he_window.png     H&E of the window at level 0, resized to 1200 px
  he_section.png    the whole section, longest side 1600 px, with the window box in window_box.json
  maps.npz          on the 16 um bin grid of the window: first principal component of measured (T) and predicted (P)
                    expression, its smoothing levels W^t (t = 0, 1, 2, ..., 2048) and its 12 bands
  numbers.json      per-gene PCC and band scores beta_1..beta_12 of the prediction on the whole test half, band widths"""
import os, sys, json
import numpy as np, anndata as ad
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

OUT = f"{R}/results/repro_vhd/fig1_framework"; os.makedirs(OUT, exist_ok=True)
TAG, SLIDE = "P5", "Visium_HD_Human_Colon_Cancer_P5"
FOLD, BLOCK_UM, MARGIN_UM, HVG_POOL, HVG_EVAL, SIDE = "half", 320.0, 64.0, 200, 50, 900.0
NUM = json.load(open(f"{R}/results/repro_vhd/fig1_windows/fig1_v2_numbers.json"))
x0, y0 = NUM[TAG]["best"]["x0"], NUM[TAG]["best"]["y0"]
TS = [0] + [2 ** i for i in range(12)]
print(f"H5AD {SEB.H5AD}; window {TAG} x0 {x0:.1f} y0 {y0:.1f}; OMP {os.environ.get('OMP_NUM_THREADS')}", flush=True)

a = ad.read_h5ad(SEB.H5AD)
X = np.nan_to_num(np.asarray(a.X, np.float32))
slide = a.obs["slide_id"].astype(str).values
pxl = np.asarray(a.obsm["pxl"], np.float64)
img = np.nan_to_num(np.concatenate([np.load(os.path.join(SEB.EMBDIR, "emb_hibou_l_P2.npy")),
                                    np.load(os.path.join(SEB.EMBDIR, "emb_hibou_l_P5.npy"))]).astype(np.float32))
m = np.where(slide == SLIDE)[0]
xy = pxl[m] / PX_PER_UM[SLIDE]; Xs, IM = X[m], img[m]
trm, tem = make_split(xy, FOLD, BLOCK_UM, MARGIN_UM)
gidx = E.topk_hvg(Xs[trm], HVG_POOL)[-HVG_EVAL:]
yte = Xs[tem][:, gidx]
RDG = ridge_predict(IM[trm], Xs[trm][:, gidx], IM[tem], 1e4)
pcc = float(E.per_gene_pcc(RDG, yte).mean())
print(f"per-gene PCC {pcc:.4f} (reference: {NUM[TAG]['pcc']['ridge']:.4f})", flush=True)

mu = yte.mean(0); Yc = yte - mu
_, _, Vt = np.linalg.svd(Yc, full_matrices=False); load = Vt[0]
if (Yc @ load).mean() < 0: load = -load
xt = xy[tem]
W = build_operator(xt, k=8, cut_um=29.0)
sig = calibrate_sigma(W, xt, TS[1:])

# band scores of the prediction on the whole test half (all 50 genes), and the smoothing levels of the PC1 maps
lev = {"T": {}, "P": {}}; beta = {}
VT, VP = yte.astype(np.float32).copy(), RDG.astype(np.float32).copy()
fT, fP = (Yc @ load).astype(np.float32), ((RDG - mu) @ load).astype(np.float32)
prevT, prevP, t = VT.copy(), VP.copy(), 0
for tt in TS:
    while t < tt:
        VT, VP, fT, fP = W @ VT, W @ VP, W @ fT, W @ fP; t += 1
    lev["T"][tt], lev["P"][tt] = fT.copy(), fP.copy()
    if tt > 0:
        beta[tt] = float(E.per_gene_pcc(prevP - VP, prevT - VT).mean())
        prevT, prevP = VT.copy(), VP.copy()
print("beta", {k: round(v, 3) for k, v in beta.items()}, flush=True)
print("sigma", {k: round(v, 1) for k, v in sig.items()}, flush=True)

s_ = (xt[:, 0] >= x0) & (xt[:, 0] < x0 + SIDE) & (xt[:, 1] >= y0) & (xt[:, 1] < y0 + SIDE)
sub = xt[s_]; ix = np.round((sub[:, 0] - x0) / 16.0).astype(int); iy = np.round((sub[:, 1] - y0) / 16.0).astype(int)
def grid(v):
    H = np.full((iy.max() + 1, ix.max() + 1), np.nan, np.float32); H[iy, ix] = v[s_]; return H
arr = {}
for k in ("T", "P"):
    for tt in TS:
        arr[f"{k}_lev{tt}"] = grid(lev[k][tt])
    for i in range(1, 13):
        arr[f"{k}_band{i}"] = grid(lev[k][TS[i - 1]] - lev[k][TS[i]])
np.savez_compressed(f"{OUT}/maps.npz", **arr)

# H&E: the window at level 0 and the whole section
import openslide
sl = openslide.OpenSlide(f"{R}/data/visiumhd/{SLIDE}/{SLIDE}_PYRAMIDAL.tif")
hp = pxl[m][tem][s_]; half = 8.0 * PX_PER_UM[SLIDE]
bx0, by0 = hp[:, 0].min() - half, hp[:, 1].min() - half
bx1, by1 = hp[:, 0].max() + half, hp[:, 1].max() + half
w_, h_ = int(round(bx1 - bx0)), int(round(by1 - by0))
win = sl.read_region((int(round(bx0)), int(round(by0))), 0, (w_, h_)).convert("RGB")
win.resize((1200, int(round(1200 * h_ / w_))), Image.LANCZOS).save(f"{OUT}/he_window.png")
W0, H0 = sl.dimensions; sc = 1600.0 / max(W0, H0)
sl.get_thumbnail((int(W0 * sc), int(H0 * sc))).convert("RGB").save(f"{OUT}/he_section.png")
th = Image.open(f"{OUT}/he_section.png"); fx, fy = th.size[0] / W0, th.size[1] / H0
allp = pxl[m]
json.dump(dict(window_px=[bx0, by0, bx1, by1], level0=[W0, H0], thumb=list(th.size),
               window_thumb=[bx0 * fx, by0 * fy, bx1 * fx, by1 * fy],
               tissue_thumb=[float(allp[:, 0].min() * fx), float(allp[:, 1].min() * fy), float(allp[:, 0].max() * fx), float(allp[:, 1].max() * fy)],
               um_per_px_window=float(SIDE / 1200.0 * (w_ / (SIDE * PX_PER_UM[SLIDE])))),
          open(f"{OUT}/window_box.json", "w"), indent=1)
json.dump(dict(tag=TAG, x0=x0, y0=y0, side_um=SIDE, n_bins_window=int(s_.sum()), grid_shape=list(arr["T_lev0"].shape),
               pcc=pcc, beta={str(k): v for k, v in beta.items()}, sigma_um={str(k): v for k, v in sig.items()},
               pc1_share=float((Yc @ load).var() / Yc.var(0).sum()), genes=[str(g) for g in np.asarray(a.var_names)[gidx]]),
          open(f"{OUT}/numbers.json", "w"), indent=1)
print("done", flush=True)

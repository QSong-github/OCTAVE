# -*- coding: utf-8 -*-
"""the thread-count note of Supp. "Two details affect reproduction", rerun on the rebuilt Visium HD inputs.
Observed (P5 half fold): K-means at 8 and 32 threads fell into different local optima, ARI 0.8318, about 11% of bins changed domain,
and D_domImg_k20 was 0.5237314 against 0.5213906. The partition is the one of src/ruler_fold.py: PCA(50) of the Hibou-L features of
the whole section, KMeans(20, n_init=4, random_state=0). The score is on the P5 half fold.
  --save     partition and score at the thread count set in OMP_NUM_THREADS
  --compare  ARI of the 8- and 32-thread partitions, and the share of bins that change domain after matching domains one to one"""
import os, sys, json, argparse, numpy as np, anndata as ad
R = "/path/to/project"
sys.path.insert(0, f"{R}/src")
import st_encoder_bench as SEB, effres
SEB.H5AD, SEB.EMBDIR = f"{R}/data/vhd_legacy/binned_16um.h5ad", f"{R}/data/vhd_legacy/emb"
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import evaluate as E
from within_bench import make_split
from ruler import group_means
from ruler_fold import BLOCK_UM, MARGIN_UM, HVG_POOL, HVG_EVAL
from effres import PX_PER_UM, SLIDES

OUT = f"{R}/results/repro_vhd/kmeans_threads"
os.makedirs(OUT, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--save", action="store_true"); ap.add_argument("--compare", action="store_true")
a_ = ap.parse_args()
s = [x for x in SLIDES if x.endswith("P5")][0]
if a_.save:
    nt = int(os.environ["OMP_NUM_THREADS"])
    a = ad.read_h5ad(SEB.H5AD)
    m = np.where(a.obs["slide_id"].astype(str).values == s)[0]
    X = np.nan_to_num(np.asarray(a.X, np.float32))[m]
    xy = np.asarray(a.obsm["pxl"], np.float64)[m] / PX_PER_UM[s]
    img = np.nan_to_num(np.load(f"{SEB.EMBDIR}/emb_hibou_l_P5.npy").astype(np.float32))
    assert len(img) == len(m), (len(img), len(m))
    trm, tem = make_split(xy, "half", BLOCK_UM, MARGIN_UM)
    y = X[:, E.topk_hvg(X[trm], HVG_POOL)[-HVG_EVAL:]]
    lab = KMeans(n_clusters=20, n_init=4, random_state=0).fit_predict(PCA(n_components=50, random_state=0).fit_transform(img))
    sc = float(E.per_gene_pcc(group_means(y, lab)[tem], y[tem]).mean())
    np.save(f"{OUT}/lab_t{nt}.npy", lab)
    json.dump(dict(threads=nt, D_domImg_k20=sc), open(f"{OUT}/score_t{nt}.json", "w"))
    print(f"threads {nt}: D_domImg_k20 {sc:.7f}", flush=True)
if a_.compare:
    l8, l32 = np.load(f"{OUT}/lab_t8.npy"), np.load(f"{OUT}/lab_t32.npy")
    C = np.zeros((20, 20), int); np.add.at(C, (l8, l32), 1)
    r, c = linear_sum_assignment(-C)
    res = dict(ari=float(adjusted_rand_score(l8, l32)), changed=float(1 - C[r, c].sum() / len(l8)),
               score_t8=json.load(open(f"{OUT}/score_t8.json"))["D_domImg_k20"],
               score_t32=json.load(open(f"{OUT}/score_t32.json"))["D_domImg_k20"])
    json.dump(res, open(f"{OUT}/compare.json", "w"), indent=1)
    print(res, flush=True)

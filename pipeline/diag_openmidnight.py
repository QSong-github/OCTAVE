# -*- coding: utf-8 -*-
"""为什么 OpenMidnight 的 K=20 域预言机异常低？在同一配方（PCA(50) 不标准化 → KMeans）上做诊断，并跑一个反事实（先 z-score 各维再 PCA）。"""
import os, sys, json, glob, numpy as np, h5py, anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
B = "/path/to/he2st/HEST/eval/bench_data"; EMB = "/path/to/systema4ST/results/hest_emb"
ENCS = ["openmidnight", "midnight12k", "hoptimus0", "dinov2_large", "ciga"]
SIDS = ["INT1", "INT24", "TENX117", "TENX148", "MEND151", "NCBI643"]     # CCRCC×2, SKCM, COAD, PRAD?, HCC
def per_gene_pcc(P, Y):
    P = P - P.mean(0); Y = Y - Y.mean(0)
    num = (P * Y).sum(0); den = np.sqrt((P ** 2).sum(0) * (Y ** 2).sum(0)) + 1e-12
    return float(np.mean(num / den))
def group_means(Y, lab):
    out = np.empty_like(Y)
    for c in np.unique(lab): m = lab == c; out[m] = Y[m].mean(0)
    return out
def find(sid):
    for c in sorted(os.listdir(B)):
        f = os.path.join(B, c, "adata", sid + ".h5ad")
        if os.path.exists(f): return c, f
    return None, None
def load_Y(c, f, bc):
    genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]
    a = ad.read_h5ad(f); pos = {b: i for i, b in enumerate(a.obs_names.astype(str))}
    sub = a[np.array([pos[b] for b in bc])]; gi = [list(sub.var_names.astype(str)).index(g) for g in genes]
    Y = np.asarray(sub.X.todense() if sparse.issparse(sub.X) else sub.X, np.float64)[:, gi]
    return np.log1p(Y).astype(np.float32)
def coherence(lab, coords, k=6):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords); idx = nn.kneighbors(coords, return_distance=False)[:, 1:]
    return float((lab[idx] == lab[:, None]).mean())
print("%-13s %-8s %5s | %6s %6s %5s | %5s %5s | %5s %5s | %6s | %6s %6s %6s" % (
    "encoder", "sample", "dim", "v1dim", "v5dim", "#big", "PC1%", "PC5%", "Neff", "coh20", "r_int", "orc20", "orcZ20", "orc200"), flush=True)
agg = {}
for sid in SIDS:
    c, f = find(sid)
    if f is None: print("跳过", sid); continue
    with h5py.File(os.path.join(B, c, "patches", sid + ".h5"), "r") as h:
        coords = np.asarray(h["coords"][:], np.float64); bck = "barcodes" if "barcodes" in h else "barcode"
        hbc = np.asarray(h[bck][:]).flatten().astype(str).tolist()
        inten = np.asarray(h["img"][:], np.float32).mean(axis=(1, 2, 3))            # 每片平均亮度
    for e in ENCS:
        ef = os.path.join(EMB, f"{sid}_{e}.npz")
        if not os.path.exists(ef): continue
        z = np.load(ef, allow_pickle=True); X, bc = z["X"].astype(np.float32), z["bc"].tolist()
        order = {b: i for i, b in enumerate(hbc)}; sel = np.array([order[b] for b in bc])
        co, it = coords[sel], inten[sel]
        Y = load_Y(c, f, bc)
        v = X.var(0); vs = np.sort(v)[::-1]; tot = v.sum()
        v1, v5 = vs[0] / tot, vs[:5].sum() / tot; nbig = int((v > 20 * np.median(v)).sum())
        pca = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1), random_state=0); pc = pca.fit_transform(X)
        pc1, pc5 = pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[:5].sum()
        lab = KMeans(n_clusters=20, n_init=4, random_state=0).fit_predict(pc)
        cnt = np.bincount(lab, minlength=20) / len(lab); neff = float(np.exp(-(cnt[cnt > 0] * np.log(cnt[cnt > 0])).sum()))
        coh = coherence(lab, co); r_int = float(abs(np.corrcoef(pc[:, 0], it)[0, 1]))
        orc20 = per_gene_pcc(group_means(Y, lab), Y)
        lab200 = KMeans(n_clusters=min(200, len(lab)), n_init=4, random_state=0).fit_predict(pc); orc200 = per_gene_pcc(group_means(Y, lab200), Y)
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-6); pcz = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1), random_state=0).fit_transform(Xz)
        labz = KMeans(n_clusters=20, n_init=4, random_state=0).fit_predict(pcz); orcz = per_gene_pcc(group_means(Y, labz), Y)
        print("%-13s %-8s %5d | %6.3f %6.3f %5d | %5.2f %5.2f | %5.1f %5.2f | %6.2f | %6.3f %6.3f %6.3f" % (
            e, sid, X.shape[1], v1, v5, nbig, pc1, pc5, neff, coh, r_int, orc20, orcz, orc200), flush=True)
        agg.setdefault(e, []).append((v1, pc1, neff, coh, r_int, orc20, orcz, orc200))
print("\n均值（%d 个样本）：" % len(SIDS))
for e, L in agg.items():
    a = np.mean(L, 0); print("%-13s v1dim %.3f PC1%% %.2f Neff %.1f coh20 %.2f r_int %.2f | orc20 %.3f  z-score后 %.3f  orc200 %.3f" % (e, *a))

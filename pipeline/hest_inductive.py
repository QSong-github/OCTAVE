# -*- coding: utf-8 -*-
"""归纳式分区预测器（HEST）：对每个测试样本，PCA+KMeans 只在同队列其它样本的特征上拟合，测试 spot 用该模型指派簇，簇均值来自训练样本。
与 hest_controls.py 的 trainonly_image（分区在测试样本特征上拟合，transductive）并列。基线：留一队列选 α 的 ridge（hest_rsel_ps）。"""
import os, sys, json, glob, warnings, numpy as np, anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
B = "/path/to/he2st/HEST/eval/bench_data"; EMB = "/path/to/project/results/hest_emb"; R = "/path/to/project/results"; K = 20
def per_gene_pcc(P, Y):
    P = P - P.mean(0); Y = Y - Y.mean(0); num = (P * Y).sum(0); den = np.sqrt((P ** 2).sum(0) * (Y ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"): r = num / den
    return float(np.nanmean(r))
enc = sys.argv[1]; out = {}
for c in sorted(os.listdir(B)):
    if not os.path.isdir(os.path.join(B, c, "adata")): continue
    genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]; S = {}
    for f in sorted(glob.glob(os.path.join(B, c, "adata", "*.h5ad"))):
        sid = os.path.basename(f)[:-5]; ef = os.path.join(EMB, f"{sid}_{enc}.npz")
        if not os.path.exists(ef): continue
        z = np.load(ef, allow_pickle=True); X, bc = z["X"].astype(np.float32), z["bc"].tolist()
        a = ad.read_h5ad(f); pos = {b: i for i, b in enumerate(a.obs_names.astype(str))}; sub = a[np.array([pos[b] for b in bc])]
        gi = [list(sub.var_names.astype(str)).index(g) for g in genes]
        S[sid] = (X, np.log1p(np.asarray(sub.X.todense() if sparse.issparse(sub.X) else sub.X, np.float64)[:, gi]).astype(np.float32))
    for sid, (X, Y) in S.items():
        tr = [s for s in S if s != sid]
        if not tr: out[sid] = dict(cohort=c, n=int(len(Y)), inductive=None); continue
        Xtr = np.concatenate([S[s][0] for s in tr]); Ytr = np.concatenate([S[s][1] for s in tr])
        pca = PCA(n_components=min(50, Xtr.shape[1], Xtr.shape[0] - 1), random_state=0).fit(Xtr); km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(pca.transform(Xtr))
        ltr = km.labels_; lte = km.predict(pca.transform(X)); g = Ytr.mean(0)
        mu = np.stack([Ytr[ltr == u].mean(0) if (ltr == u).any() else g for u in range(K)])
        out[sid] = dict(cohort=c, n=int(len(Y)), inductive=per_gene_pcc(mu[lte], Y), n_train=len(tr), clusters_used=int(len(np.unique(lte))))
        print(f"  {sid} n={len(Y)} inductive={out[sid]['inductive']:.4f} 用到簇 {out[sid]['clusters_used']}/20", flush=True)
json.dump({"encoder": enc, "K": K, "samples": out}, open(f"{R}/hest_inductive_{enc}.json", "w"), indent=1); print(f"{enc}: {len(out)} 样本")

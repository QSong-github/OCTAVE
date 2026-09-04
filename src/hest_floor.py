#!/usr/bin/env python
"""HEST 广度线的形态学下界（imageKNN）+ 带通曲线。

深度线用 imageKNN 当「纯形态学下界」——只按图像相似度检索训练点取平均，不学任何回归。
广度线一直缺这个下界，导致「距下界的差距」这个**不依赖极值**的统计只能在深度线上说。
本脚本补上，口径与 hest_effres.py 逐行一致（同划分、同基因、同 target、同扩散算子），
唯一区别是把 Ridge 换成 kNN 检索。

按队列并行：--cohort 一次跑一个，10 个队列同时跑。
"""
import os, sys, glob, json, argparse, warnings
import numpy as np, anndata as ad
from scipy.spatial import cKDTree
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
SRC = "/blue/qsong1/wang.qing/systema4ST/src"
sys.path.insert(0, SRC)
from hest_effres import build_operator, calibrate_sigma, per_gene_pcc, B, EMB, LADDER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--encoder", default="phikon_v2",
                    help="仅用于取图像嵌入做检索；下界本身不含任何训练")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--tmax", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.out = a.out or f"results/hest_floor/{a.cohort}.json"

    cps = [1]
    while cps[-1] < a.tmax:
        cps.append(cps[-1] * 2)
    umpx = {k: v["umpx"] for k, v in json.load(open(LADDER)).items()}

    c = a.cohort
    genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]
    cache = {}

    def get(sid):
        if sid in cache:
            return cache[sid]
        z = np.load(os.path.join(EMB, f"{sid}_{a.encoder}.npz"), allow_pickle=True)
        X, bc = z["X"], z["bc"].tolist()
        ad_ = ad.read_h5ad(os.path.join(B, c, "adata", f"{sid}.h5ad"))
        pos = {b: i for i, b in enumerate(ad_.obs_names.astype(str))}
        sub = ad_[np.array([pos[b] for b in bc])]
        gi = [list(sub.var_names.astype(str)).index(g) for g in genes]
        Y = np.asarray(sub.X.todense() if sparse.issparse(sub.X) else sub.X,
                       np.float32)[:, gi]
        Y = np.log1p(Y.astype(np.float64)).astype(np.float32)   # 官方口径：仅 log1p
        xy = np.asarray(sub.obsm["spatial"], np.float64) * umpx.get(sid, 1.0)
        cache[sid] = (X.astype(np.float32), Y, xy)
        return cache[sid]

    out = {"cohort": c, "encoder": a.encoder, "k": a.k, "samples": {}}
    nfold = len(glob.glob(os.path.join(B, c, "splits", "test_*.csv")))
    for kf in range(nfold):
        rd = lambda f: [l.split(",")[0] for l in
                        open(os.path.join(B, c, "splits", f)).read().splitlines()[1:]
                        if l.strip()]
        tr, te = rd(f"train_{kf}.csv"), rd(f"test_{kf}.csv")
        Xtr = np.concatenate([get(s)[0] for s in tr])
        Ytr = np.concatenate([get(s)[1] for s in tr])
        pipe = Pipeline([("sc", StandardScaler()),
                         ("pca", PCA(n_components=min(a.latent_dim, Xtr.shape[1],
                                                      Xtr.shape[0] - 1),
                                     random_state=0))]).fit(Xtr)
        Ztr = pipe.transform(Xtr)
        tree = cKDTree(Ztr)
        print(f"[{c} fold{kf}] train={len(tr)} test={te} Ztr={Ztr.shape}", flush=True)

        for s in te:
            Xs, Ys, xy = get(s)
            _, nn = tree.query(pipe.transform(Xs), k=min(a.k, len(Ztr)))
            P = Ytr[nn].mean(1).astype(np.float32)     # 纯检索平均，无回归
            pcc = float(np.nanmean(per_gene_pcc(P, Ys)))

            W, pitch, deg = build_operator(xy)
            sig = calibrate_sigma(W, xy, cps)
            M = np.concatenate([Ys, P], 1).astype(np.float32)
            G = Ys.shape[1]
            prev, low_prev, t = M.copy(), M.copy(), 0
            curve = {}
            for cp in cps:
                while t < cp:
                    prev = W @ prev; t += 1
                band = low_prev - prev
                curve[str(cp)] = float(np.nanmean(per_gene_pcc(band[:, G:], band[:, :G])))
                low_prev = prev.copy()
            out["samples"][s] = {"pcc": pcc, "n": int(Ys.shape[0]),
                                 "sigma_um": {str(k_): float(v) for k_, v in sig.items()},
                                 "band_pcc": curve}
            print(f"    {s}  imageKNN PCC={pcc:.4f}  band[min]={curve[str(cps[0])]:.4f}",
                  flush=True)

    os.makedirs("results/hest_floor", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"已存 {a.out}  ({len(out['samples'])} 样本)", flush=True)


if __name__ == "__main__":
    main()

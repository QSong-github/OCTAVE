# -*- coding: utf-8 -*-
"""
Systema 第 5 步的对应物 + 指标稳健性检验。

Systema 用 centroid accuracy(预测的扰动质心是否更靠近自己的真值质心而非别的扰动)
补充相关系数, 因为它阈值无关、尺度无关。ST 的对应物是【空间检索准确率】。

同时解决一个隐患: 本文迄今全部结论都建立在 Pearson 上。若换成完全不同性质的指标
(检索准确率是排序型、ARI 是聚类型、Moran's I 是空间自相关型)结论翻转, 那结论就是
Pearson 特有的假象。所以每个指标都在【同一把 A_coarse 阶梯】上重读一次等价 σ ——
若四个指标给出一致的等价 σ, 结论与指标无关; 若不一致, 必须如实报告。

四个指标:
  pcc        per-gene Pearson(领域主指标, 参照)
  ret@1      spot 检索: 预测谱是否最接近自己那个 spot 的真值(随机基线 1/n)
  dom@1      域检索: 预测谱是否最接近自己所属域的真值质心(随机基线 1/k)
  ari        对预测做 KMeans vs 对真值做 KMeans 的一致性
另报 Moran's I 膨胀: 若模型本质低通, 预测的空间自相关应系统性高于真值 ——
任何人拿预测 ST 去找空间可变基因都会被这个偏差污染。
"""
import os, sys, glob, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/path/to/he2st/HEST/src")
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hest.bench.st_dataset import load_adata
from hest.bench.trainer import train_test_reg
import evaluate as E
from hest_ladder import build_operator, calibrate_sigma

B = "/path/to/he2st/HEST/eval/bench_data"
EMB = "/path/to/systema4ST/results/hest_emb"
SAMPLES = "/path/to/systema4ST/results/hest_samples.json"
MAXRET = 4000          # 检索指标的子采样上限(O(n^2))


def zscore(A):
    A = A - A.mean(0); s = A.std(0); s[s < 1e-8] = 1.0
    return A / s


def ret_at_k(pred, truth, rng, ks=(1, 10)):
    """spot 检索: 每个预测谱在全部真值谱中的排名。行方向余弦。"""
    n = pred.shape[0]
    idx = rng.choice(n, min(MAXRET, n), replace=False)
    P, T = zscore(pred[idx]), zscore(truth[idx])
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-8
    S = P @ T.T
    self_s = np.diag(S).copy()
    rank = (S > self_s[:, None]).sum(1)          # 有多少个比自己更像
    return {f"ret@{k}": float((rank < k).mean()) for k in ks}, len(idx)


def dom_at1(pred, truth, lab):
    """域检索: 预测谱是否最接近自己所属域的真值质心。"""
    us = np.unique(lab)
    C = np.stack([truth[lab == u].mean(0) for u in us])
    d = ((pred[:, None, :] - C[None, :, :]) ** 2).sum(-1) if len(us) * pred.shape[0] < 4e7 else None
    if d is None:
        d = np.empty((pred.shape[0], len(us)), np.float32)
        for j in range(len(us)):
            d[:, j] = ((pred - C[j]) ** 2).sum(1)
    return float((us[d.argmin(1)] == lab).mean())


def morans_I(A, X):
    """逐基因 Moran's I。A: 二值邻接(对称)。"""
    Xc = X - X.mean(0)
    num = (Xc * (A @ Xc)).sum(0)
    den = (Xc ** 2).sum(0) + 1e-12
    return (X.shape[0] / A.sum()) * num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--tmax", type=int, default=8192)
    ap.add_argument("--k_dom", type=int, default=10)
    ap.add_argument("--out", default=None)
    a_ = ap.parse_args()
    a_.out = a_.out or f"results/hest_downstream_{a_.encoder}.json"
    cps = [1]
    while cps[-1] < a_.tmax:
        cps.append(cps[-1] * 2)
    info = {r["sid"]: r for r in json.load(open(SAMPLES))}
    rng = np.random.default_rng(0)
    out = {}

    for c in sorted(os.listdir(B)):
        if not os.path.isdir(os.path.join(B, c, "adata")):
            continue
        genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]
        cache = {}

        def get(sid):
            if sid not in cache:
                z = np.load(os.path.join(EMB, f"{sid}_{a_.encoder}.npz"), allow_pickle=True)
                X, bc = z["X"], z["bc"].tolist()
                Y = load_adata(os.path.join(B, c, "adata", f"{sid}.h5ad"), genes=genes,
                               barcodes=bc, normalize=True).values.astype(np.float32)
                import anndata as ad
                aa = ad.read_h5ad(os.path.join(B, c, "adata", f"{sid}.h5ad"))
                bidx = {b: i for i, b in enumerate(map(str, aa.obs_names))}
                keep = np.array([bidx[b] for b in bc])
                xy = np.asarray(aa.obsm["spatial"], np.float64)[keep] * info[sid]["umpx"]
                cache[sid] = (X, Y, xy)
            return cache[sid]

        for k in range(len(glob.glob(os.path.join(B, c, "splits", "test_*.csv")))):
            rd = lambda f: [l.split(",")[0] for l in open(os.path.join(B, c, "splits", f)
                            ).read().splitlines()[1:] if l.strip()]
            tr, te = rd(f"train_{k}.csv"), rd(f"test_{k}.csv")
            Xtr = np.concatenate([get(s)[0] for s in tr]); Ytr = np.concatenate([get(s)[1] for s in tr])
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("PCA", PCA(n_components=min(256, Xtr.shape[1], Xtr.shape[0] - 1),
                                         random_state=0))]).fit(Xtr)
            Ztr = pipe.transform(Xtr)
            for s in te:
                if s in out:
                    continue
                Xs, Ys, xy = get(s)
                _, dump = train_test_reg(Ztr, pipe.transform(Xs), Ytr, Ys, genes=genes, method="ridge")
                pred = np.asarray(dump["preds_all"], np.float32)
                W, deg, iso = build_operator(xy)
                A = (W > 0).astype(np.float32); A.setdiag(0); A.eliminate_zeros()
                sig = calibrate_sigma(W, xy, cps)
                lab = KMeans(n_clusters=min(a_.k_dom, max(2, Ys.shape[0] // 50)),
                             n_init=4, random_state=0).fit_predict(zscore(Ys))
                I_true = morans_I(A, Ys)

                def metrics(P):
                    r, nret = ret_at_k(P, Ys, np.random.default_rng(0))
                    lp = KMeans(n_clusters=len(np.unique(lab)), n_init=4,
                                random_state=0).fit_predict(zscore(P))
                    return dict(pcc=float(E.per_gene_pcc(P, Ys).mean()), **r,
                                dom1=dom_at1(P, Ys, lab), ari=float(adjusted_rand_score(lab, lp)),
                                moran=float(np.mean(morans_I(A, P))), n_ret=nret)

                m_pred = metrics(pred)
                ladder, cur, t = {}, Ys.copy(), 0
                for cp in cps:
                    while t < cp:
                        cur = W @ cur; t += 1
                    ladder[str(cp)] = metrics(cur)
                out[s] = dict(cohort=c, platform=info[s]["tech"], n=int(Ys.shape[0]),
                              sigma_um={str(x): sig[x] for x in cps},
                              moran_true=float(np.mean(I_true)),
                              pred=m_pred, ladder=ladder,
                              chance_ret=1.0 / m_pred["n_ret"],
                              chance_dom=1.0 / len(np.unique(lab)))
                print(f"{c:10s}{s:10s} n={Ys.shape[0]:6d} | pcc={m_pred['pcc']:.3f} "
                      f"ret@1={m_pred['ret@1']:.4f}(chance {1/m_pred['n_ret']:.1e}) "
                      f"dom1={m_pred['dom1']:.3f} ari={m_pred['ari']:.3f} "
                      f"MoranI 真={np.mean(I_true):.3f} 预测={m_pred['moran']:.3f}", flush=True)

    os.makedirs("results", exist_ok=True)
    json.dump(out, open(a_.out, "w"), indent=2, ensure_ascii=False)

    # ---- 四个指标各自读一次等价 σ
    def eqs(d, key):
        v = d["pred"][key]
        pts = sorted(((d["sigma_um"][c], d["ladder"][c][key]) for c in d["ladder"]), key=lambda t: t[0])
        if v >= pts[0][1]:
            return pts[0][0]
        for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
            if v0 >= v >= v1:
                return s0 + (v0 - v) / max(v0 - v1, 1e-12) * (s1 - s0)
        return np.nan

    print(f"\n=== 四个指标各自给出的等价 σ (编码器={a_.encoder}, n={len(out)}) ===")
    print(f"{'指标':10s}{'中位σ(µm)':>12s}{'四分位':>20s}{'落档外':>8s}")
    E4 = {}
    for key in ("pcc", "ret@1", "dom1", "ari"):
        v = np.array([eqs(d, key) for d in out.values()], float)
        E4[key] = v
        f = np.isfinite(v)
        print(f"{key:10s}{np.nanmedian(v):>12.0f}"
              f"{f'[{np.nanpercentile(v,25):.0f},{np.nanpercentile(v,75):.0f}]':>20s}"
              f"{int((~f).sum()):>8d}")
    from scipy.stats import spearmanr
    print(f"\n指标之间等价 σ 的一致性(Spearman):")
    ks = list(E4)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            m = np.isfinite(E4[ks[i]]) & np.isfinite(E4[ks[j]])
            if m.sum() > 5:
                print(f"  {ks[i]:6s} vs {ks[j]:6s}: {spearmanr(E4[ks[i]][m], E4[ks[j]][m])[0]:.3f} (n={m.sum()})")

    mt = np.array([d["moran_true"] for d in out.values()])
    mp = np.array([d["pred"]["moran"] for d in out.values()])
    print(f"\n=== Moran's I 膨胀 ===")
    print(f"  真值 {mt.mean():.3f}  预测 {mp.mean():.3f}  比值 {np.mean(mp/np.maximum(mt,1e-6)):.2f}×")
    print(f"  预测 > 真值 的样本: {int((mp > mt).sum())}/{len(mt)}")
    print(f"\n已存 {a_.out}")


if __name__ == "__main__":
    main()

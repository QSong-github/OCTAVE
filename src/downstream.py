#!/usr/bin/env python
"""下游危害：σ 差的预测在真实生物学判读上损失了什么。

这是本项目相对 PathoROB（Nat Commun 2026）唯一实质缺失的一块——
那篇的 Fig 4 证明非鲁棒表征导致**可辨认的诊断错误**，而我们此前只证明了「分辨率差距被低估」。

设计：评测栅格固定在 16 µm，只让预测的**有效分辨率**变。
四个分箱尺度（8/16/32/64 µm）的预测各自映射回同一批 16 µm bin（最近邻），
于是下游任务的输入维度、bin 数、真值完全相同，唯一变量是 σ。
（若直接在各自栅格上比，bin 数与邻接关系都变，任何差异都无法归因。）

三个度量，均不需要人工标注：
  ① 空间域一致性 —— 对预测谱做 Leiden 聚类，与真值聚类的 ARI
  ② 边界保真度 —— 在**真值梯度最大的那一成 bin** 上，|∇pred| / |∇truth|
                   完美锐利 = 1.0；被抹平 < 1.0。这直接量化「界面被糊掉多少」
  ③ 空间自相关长度 —— 预测的 Moran 相关随距离衰减到 1/e 的长度，与真值比
"""
import argparse, glob, json, os
import numpy as np
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import adjusted_rand_score

PREP = "/blue/qsong1/wang.qing/systema4ST/data/prepped_xen"
EMB = "/blue/qsong1/wang.qing/systema4ST/results/emb_xen"


def gene_names(a):
    """Xenium 的 prepped h5ad 不一定有 var["gene"] 列 —— 回退到 var_names。"""
    for c in ("gene", "gene_name", "feature_name", "symbol"):
        if c in a.var.columns:
            return np.asarray(a.var[c]).astype(str)
    return np.asarray(a.var_names).astype(str)


def topk_hvg(Y, k):
    return np.argsort(-Y.var(0))[:k]


def block_cv_predict(X, Y, xy, grid=16, seed=0):
    """与 nine.py 同协议的片内空间块 CV，返回全体 bin 的预测（未覆盖处为 nan）。"""
    qx = np.quantile(xy[:, 0], np.linspace(0, 1, grid + 1)); qx[-1] += 1
    qy = np.quantile(xy[:, 1], np.linspace(0, 1, grid + 1)); qy[-1] += 1
    fold = (np.searchsorted(qx, xy[:, 0], "right") - 1) * grid + \
           (np.searchsorted(qy, xy[:, 1], "right") - 1)
    P = np.full_like(Y, np.nan)
    for f in np.unique(fold):
        te = fold == f
        if te.sum() < 20 or (~te).sum() < 2000:
            continue
        mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-8
        m = Ridge(alpha=1e4).fit((X[~te] - mu) / sd, Y[~te])
        P[te] = m.predict((X[te] - mu) / sd).astype(np.float32)
    return P


def grad_mag(V, xy, k=8):
    """每个 bin 的空间梯度幅值：与 k 近邻的表达差 / 距离，取均值。"""
    d, idx = cKDTree(xy).query(xy, k=k + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    d = np.maximum(d, 1e-6)
    diff = np.abs(V[idx] - V[:, None, :]).mean(2)       # (n,k) 跨基因平均
    return (diff / d).mean(1)


def moran_length(V, xy, rings=(20, 40, 80, 160, 320)):
    """相关随距离衰减：各距离环上的空间自相关，返回衰减到 1/e 的长度（线性插值）。"""
    tree = cKDTree(xy)
    Vc = V - V.mean(0)
    denom = (Vc ** 2).sum(0) + 1e-12
    prev, out = 0.0, []
    for r in rings:
        pairs = tree.query_pairs(r, output_type="ndarray")
        if len(pairs) < 100:
            out.append(np.nan); continue
        if len(pairs) > 2_000_000:
            sel = np.random.default_rng(0).choice(len(pairs), 2_000_000, replace=False)
            pairs = pairs[sel]
        num = (Vc[pairs[:, 0]] * Vc[pairs[:, 1]]).sum(0)
        out.append(float(np.mean(num / denom * len(V) / len(pairs))))
    vals = np.array(out, float)
    ok = np.isfinite(vals)
    if ok.sum() < 2 or vals[ok][0] <= 0:
        return float("nan")
    tgt = vals[ok][0] / np.e
    rr = np.array(rings, float)[ok]; vv = vals[ok]
    for i in range(len(vv) - 1):
        if vv[i] >= tgt >= vv[i + 1]:
            w = (vv[i] - tgt) / max(vv[i] - vv[i + 1], 1e-12)
            return float(rr[i] + w * (rr[i + 1] - rr[i]))
    return float(rr[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--nclust", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or f"results/downstream_{args.name}.json"

    # 参照栅格 = 16 µm
    a16 = ad.read_h5ad(f"{PREP}/{args.name}_bin16.h5ad")
    px16 = float(a16.uns["px_per_um"])
    xy16 = np.asarray(a16.obsm["pxl"], np.float64) / px16
    Y16 = np.log1p(np.asarray(sparse.csr_matrix(a16.X).todense(), np.float32))
    gidx = topk_hvg(Y16, args.hvg)
    Ytrue = Y16[:, gidx]
    print(f"[{args.name}] 参照栅格 16µm: {len(xy16)} bin × {Ytrue.shape[1]} 基因", flush=True)

    from sklearn.cluster import KMeans
    def cluster(V):
        Z = PCA(n_components=min(20, V.shape[1]), random_state=0).fit_transform(
            np.nan_to_num(V))
        return KMeans(n_clusters=args.nclust, n_init=10, random_state=0).fit_predict(Z)

    lab_true = cluster(Ytrue)
    g_true = grad_mag(Ytrue, xy16)
    hi = g_true >= np.quantile(g_true, 0.9)          # 真值梯度最大的一成 = 组织界面
    len_true = moran_length(Ytrue, xy16)
    print(f"  真值：界面 bin {hi.sum()}，自相关长度 {len_true:.0f}µm", flush=True)

    res = {"name": args.name, "n_bin16": int(len(xy16)), "n_genes": int(Ytrue.shape[1]),
           "truth_corr_len_um": len_true, "scales": {}}

    for b in (8, 16, 32, 64):
        f = f"{PREP}/{args.name}_bin{b}.h5ad"
        e = f"{EMB}/emb_hibou_l_{args.name}" + ("" if b == 16 else f"_bin{b}") + ".npy"
        if not (os.path.exists(f) and os.path.exists(e)):
            print(f"  bin{b}: 缺文件，跳过", flush=True); continue
        ab = ad.read_h5ad(f)
        pxb = float(ab.uns["px_per_um"])
        xyb = np.asarray(ab.obsm["pxl"], np.float64) / pxb
        Yb = np.log1p(np.asarray(sparse.csr_matrix(ab.X).todense(), np.float32))
        Xb = np.nan_to_num(np.load(e).astype(np.float32))
        if Xb.shape[0] != Yb.shape[0]:
            print(f"  bin{b}: 嵌入 {Xb.shape[0]} vs 表达 {Yb.shape[0]} 不符，跳过", flush=True); continue
        # 该尺度上用**自己的**基因方差选 HVG 会导致基因集不同 ⇒ 强制用参照栅格的基因
        gn_b = list(gene_names(ab)); want = gene_names(a16)[gidx]
        missing = [g for g in want if g not in gn_b]
        if missing:
            print(f"  bin{b}: {len(missing)} 个参照基因不在该尺度，跳过", flush=True); continue
        gb = [gn_b.index(g) for g in want]
        Pb = block_cv_predict(Xb, Yb[:, gb], xyb)
        # 映射回 16µm 栅格：每个参照 bin 取最近的该尺度 bin
        _, nn = cKDTree(xyb).query(xy16, k=1)
        P16 = Pb[nn]
        ok = np.isfinite(P16).all(1)
        if ok.sum() < 0.5 * len(xy16):
            print(f"  bin{b}: 覆盖不足 {ok.sum()}/{len(xy16)}，跳过", flush=True); continue

        pcc = float(np.nanmean([np.corrcoef(P16[ok, j], Ytrue[ok, j])[0, 1]
                                for j in range(Ytrue.shape[1])
                                if np.std(P16[ok, j]) > 1e-8 and np.std(Ytrue[ok, j]) > 1e-8]))
        ari = float(adjusted_rand_score(lab_true[ok], cluster(P16[ok])))
        gp = grad_mag(P16[ok], xy16[ok])
        gt = grad_mag(Ytrue[ok], xy16[ok])
        hi_ok = gt >= np.quantile(gt, 0.9)
        edge = float(np.mean(gp[hi_ok]) / max(np.mean(gt[hi_ok]), 1e-12))
        clen = moran_length(P16[ok], xy16[ok])
        res["scales"][str(b)] = {"pcc": pcc, "ari": ari, "edge_ratio": edge,
                                 "corr_len_um": clen, "n_ok": int(ok.sum())}
        print(f"  bin{b:2d} → 16µm 栅格: PCC={pcc:.4f}  ARI={ari:.4f}  "
              f"边界保真={edge:.3f}  自相关长={clen:.0f}µm", flush=True)

    os.makedirs("results", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"已存 {args.out}")


if __name__ == "__main__":
    main()

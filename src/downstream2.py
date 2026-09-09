#!/usr/bin/env python
"""下游影响·第二批：更贴近实际分析流程的任务。

第一批（downstream.py）测的是 ARI / 边界保真 / 自相关长度 —— 都是统计量。
本批改测**分析者真正会做的事**，并尽量给出物理单位（µm），使结论可直接解读。

沿用同一控制：评测栅格固定 16 µm，四个分箱的预测都映射回同一批 bin，
唯一变量是预测的有效分辨率。

五个任务：
  ① SVG 检出一致性 —— Moran's I 排名，top-k Jaccard + 全局 Spearman
     （空间转录组分析的第一步几乎都是找空间可变基因）
  ② 热点定位 —— 每个基因取表达最高的 5% bin，真值与预测的 Jaccard
     （「这个基因在哪里高」是最常问的问题）
  ③ 边界定位误差（µm）—— 域边界的对称中位距，直接给出「边界偏移多少微米」
  ④ 小热点召回 vs 尺寸 —— 按连通域直径分组的召回率
     （分辨率损失应当先吃掉小结构，此项给出「多小的结构会被漏掉」）
  ⑤ 基因共定位保持 —— 基因间空间相关矩阵的保持程度
"""
import argparse, json, os
import numpy as np
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

PREP = "/path/to/systema4ST/data/prepped_xen"
EMB = "/path/to/systema4ST/results/emb_xen"


def gene_names(a):
    for c in ("gene", "gene_name", "feature_name", "symbol"):
        if c in a.var.columns:
            return np.asarray(a.var[c]).astype(str)
    return np.asarray(a.var_names).astype(str)


def smooth(V, nb):
    """轻度空间平滑：自身与 k 近邻的均值。

    16µm 分箱每 bin 仅 1–2 细胞，真值噪声极大：不平滑直接聚类会得到空间碎片化的域，
    99.5% 的 bin 都成了「边界」，边界定位与热点尺寸两项因此失效。
    做空间域识别的人本来就会先平滑 —— 这是标准流程的一步，而非放水。
    **真值与预测同等施加**，故不偏袒任何一方。
    """
    return (V + V[nb].sum(1)) / (1 + nb.shape[1])


def knn_graph(xy, k=8):
    d, idx = cKDTree(xy).query(xy, k=k + 1)
    return idx[:, 1:], d[:, 1:]


def morans_i(V, nb):
    """每个基因的 Moran's I（用 kNN 邻接，等权）。"""
    Vc = V - V.mean(0)
    num = (Vc * Vc[nb].mean(1)).sum(0)
    den = (Vc ** 2).sum(0) + 1e-12
    return num / den


def block_cv_predict(X, Y, xy, grid=16):
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


def jaccard_topq(A, B, q=0.95):
    """逐基因：表达最高的 (1-q) 比例 bin 的 Jaccard。"""
    out = []
    for j in range(A.shape[1]):
        ta, tb = np.quantile(A[:, j], q), np.quantile(B[:, j], q)
        sa, sb = A[:, j] >= ta, B[:, j] >= tb
        u = (sa | sb).sum()
        out.append((sa & sb).sum() / u if u else np.nan)
    return float(np.nanmean(out))


def boundary_bins(lab, nb):
    """标签与任一邻居不同 ⇒ 边界 bin。（已弃用：KMeans 无空间正则，域碎片化，
    平滑后边界占比仍达 75%，该定义不可用。保留仅供对照。）"""
    return (lab[:, None] != lab[nb]).any(1)


def boundary_by_gradient(V, xy, nb, q=0.90):
    """改用梯度定义边界：平滑表达的空间梯度最大的一成 bin。

    绕开聚类 —— 纯表达 KMeans 在 16µm 上得到的是空间碎片（边界占比 75-99%），
    而带空间先验的域识别方法（BayesSpace/SpaGCN 等）不在本文范围内。
    梯度定义与第一批的「边界保真度」用同一界面，但本项输出微米。
    """
    d = np.linalg.norm(xy[nb] - xy[:, None, :], axis=2)
    d = np.maximum(d, 1e-6)
    g = (np.abs(V[nb] - V[:, None, :]).mean(2) / d).mean(1)
    return g >= np.quantile(g, q)


def boundary_shift_um(bt, bp, xy):
    """预测界面与真值界面的对称中位距（µm）。

    细尺度上恒为 0 **不是饱和**：位移小于一个 bin（16µm）时最近邻距离必然取 0，
    这是栅格量化，即该度量的分辨下限就是 bin 间距。实测 5/5 片在 bin64 处给出
    17–31 µm，在 bin32 处 0–12 µm —— 度量有效，只是下限为 16µm。
    （曾据 bin8 单档为 0 误判为「结构性饱和」，见下方 boundary_f1 的动机说明。）
    """
    if bt.sum() < 10 or bp.sum() < 10:
        return float("nan")
    tt, tp = cKDTree(xy[bt]), cKDTree(xy[bp])
    d1, _ = tt.query(xy[bp], k=1)
    d2, _ = tp.query(xy[bt], k=1)
    return float((np.median(d1) + np.median(d2)) / 2)


def boundary_f1(bt, bp, xy, tols=(0.0, 16.0, 32.0, 64.0)):
    """界面检测的 F1 @ 容差（图像分割的 BF score）。

    补充而非替代对称中位距：后者的分辨下限是 bin 间距（16µm），
    F1 则能在亚 bin 尺度上区分「界面对齐得多好」，且给出精确率/召回率的分解。
    「容差 32µm 下 F1=0.6」可直接解读为「允许偏两个 bin 时，六成界面对得上」。
    """
    if bt.sum() < 10 or bp.sum() < 10:
        return {}
    tt, tp = cKDTree(xy[bt]), cKDTree(xy[bp])
    d_p, _ = tt.query(xy[bp], k=1)     # 每个预测界面点到最近真值界面
    d_t, _ = tp.query(xy[bt], k=1)     # 每个真值界面点到最近预测界面
    out = {}
    for d in tols:
        prec = float((d_p <= d + 1e-9).mean())
        rec = float((d_t <= d + 1e-9).mean())
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        out[f"{d:.0f}um"] = {"precision": prec, "recall": rec, "f1": f1}
    return out


def hotspot_recall_by_size(Yt, Yp, xy, nb, q=0.98, pitch=16.0):
    """真值中的小热点，预测是否也在同处升高。按连通域直径分组给召回率。

    分辨率损失应当先吃掉小结构 —— 本项给出「多小的结构开始被漏掉」。
    """
    n = len(xy)
    buckets = {}
    for j in range(Yt.shape[1]):
        thr = np.quantile(Yt[:, j], q)
        seed = Yt[:, j] >= thr
        if seed.sum() < 5:
            continue
        # 在 kNN 图上做连通域（并查集）
        parent = np.arange(n)
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        idx = np.where(seed)[0]
        sset = set(idx.tolist())
        for i in idx:
            for m in nb[i]:
                if m in sset:
                    ra, rb = find(i), find(m)
                    if ra != rb: parent[ra] = rb
        comp = {}
        for i in idx:
            comp.setdefault(find(i), []).append(i)
        pthr = np.quantile(Yp[:, j], q)
        for members in comp.values():
            if len(members) < 5:      # <5 bin 是散点不是区域，计入会淹没真实结构
                continue
            mem = np.array(members)
            # 直径：成员点的最大两两距离的近似（外接框对角）
            ext = xy[mem].max(0) - xy[mem].min(0)
            diam = float(np.hypot(*ext))
            hit = float(np.mean(Yp[mem, j] >= pthr) > 0.5)   # 过半成员在预测中也超阈
            b = ("<50µm" if diam < 50 else "50-100µm" if diam < 100 else
                 "100-200µm" if diam < 200 else "≥200µm")
            buckets.setdefault(b, []).append(hit)
    return {k: (float(np.mean(v)), len(v)) for k, v in sorted(buckets.items())}


def colocal_preserve(Yt, Yp):
    """基因×基因空间相关矩阵的保持：两个矩阵上三角的 Spearman。"""
    ct = np.corrcoef(Yt.T); cp = np.corrcoef(Yp.T)
    iu = np.triu_indices_from(ct, 1)
    a, b = ct[iu], cp[iu]
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() > 10 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--nclust", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or f"results/downstream2_{args.name}.json"

    a16 = ad.read_h5ad(f"{PREP}/{args.name}_bin16.h5ad")
    px16 = float(a16.uns["px_per_um"])
    xy = np.asarray(a16.obsm["pxl"], np.float64) / px16
    Y16 = np.log1p(np.asarray(sparse.csr_matrix(a16.X).todense(), np.float32))
    gidx = np.argsort(-Y16.var(0))[:args.hvg]
    Yt = Y16[:, gidx]
    nb, _ = knn_graph(xy)
    print(f"[{args.name}] {len(xy)} bin × {Yt.shape[1]} 基因", flush=True)

    def dom(V):
        Z = PCA(n_components=min(20, V.shape[1]), random_state=0).fit_transform(np.nan_to_num(V))
        return KMeans(n_clusters=args.nclust, n_init=10, random_state=0).fit_predict(Z)

    Yts = smooth(Yt, nb)
    bt = boundary_by_gradient(Yts, xy, nb)
    print(f"  真值：梯度界面 bin {bt.sum()}/{len(xy)} ({100*bt.mean():.1f}%，按构造应为 10%)", flush=True)

    res = {"name": args.name, "n_bin": int(len(xy)), "n_genes": int(Yt.shape[1]), "scales": {}}
    for b in (8, 16, 32, 64):
        f = f"{PREP}/{args.name}_bin{b}.h5ad"
        e = f"{EMB}/emb_hibou_l_{args.name}" + ("" if b == 16 else f"_bin{b}") + ".npy"
        if not (os.path.exists(f) and os.path.exists(e)):
            print(f"  bin{b}: 缺文件", flush=True); continue
        ab = ad.read_h5ad(f)
        xyb = np.asarray(ab.obsm["pxl"], np.float64) / float(ab.uns["px_per_um"])
        Yb = np.log1p(np.asarray(sparse.csr_matrix(ab.X).todense(), np.float32))
        Xb = np.nan_to_num(np.load(e).astype(np.float32))
        if Xb.shape[0] != Yb.shape[0]:
            print(f"  bin{b}: 形状不符", flush=True); continue
        gn_b = list(gene_names(ab)); want = gene_names(a16)[gidx]
        if any(g not in gn_b for g in want):
            print(f"  bin{b}: 基因缺失", flush=True); continue
        Pb = block_cv_predict(Xb, Yb[:, [gn_b.index(g) for g in want]], xyb)
        _, nn = cKDTree(xyb).query(xy, k=1)
        P = Pb[nn]
        ok = np.isfinite(P).all(1)
        if ok.sum() < 0.5 * len(xy):
            print(f"  bin{b}: 覆盖不足", flush=True); continue

        xo, Yo, Po, nbo = xy[ok], Yt[ok], P[ok], None
        nbo, _ = knn_graph(xo)
        mi_p = morans_i(Po, nbo); mi_to = morans_i(Yo, nbo)
        k = max(5, args.hvg // 5)
        top_t = set(np.argsort(-mi_to)[:k].tolist()); top_p = set(np.argsort(-mi_p)[:k].tolist())
        svg_j = len(top_t & top_p) / len(top_t | top_p)
        svg_rho = float(spearmanr(mi_to, mi_p).statistic)
        hot_j = jaccard_topq(Yo, Po)
        Yos, Pos = smooth(Yo, nbo), smooth(Po, nbo)
        b_t = boundary_by_gradient(Yos, xo, nbo)
        b_p = boundary_by_gradient(Pos, xo, nbo)
        shift = boundary_shift_um(b_t, b_p, xo)
        bf1 = boundary_f1(b_t, b_p, xo)
        frac_b = float(b_t.mean())
        rec = hotspot_recall_by_size(Yos, Pos, xo, nbo)
        coloc = colocal_preserve(Yo, Po)

        res["scales"][str(b)] = {"svg_top_jaccard": svg_j, "svg_rank_rho": svg_rho,
                                 "hotspot_jaccard": hot_j, "boundary_shift_um": shift,
                                 "boundary_frac_truth": frac_b, "boundary_f1": bf1,
                                 "hotspot_recall_by_size": rec, "coloc_preserve": coloc,
                                 "n_ok": int(ok.sum())}
        rr = "  ".join(f"{kk}:{vv[0]:.2f}(n={vv[1]})" for kk, vv in rec.items())
        print(f"  bin{b:2d}: SVG-Jac={svg_j:.3f} SVG-ρ={svg_rho:.3f} 热点Jac={hot_j:.3f} "
              f"共定位={coloc:.3f}", flush=True)
        if bf1:
            print("        界面 F1@容差 " + "  ".join(
                f"{k}:{v['f1']:.3f}" for k, v in bf1.items()), flush=True)
        print(f"        小热点召回 {rr}", flush=True)

    os.makedirs("results", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"已存 {args.out}")


if __name__ == "__main__":
    main()

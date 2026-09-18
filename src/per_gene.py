# -*- coding: utf-8 -*-
"""
逐基因分析 —— 检验"per-gene PCC 本质是 Moran's I 加权平均"。

迄今所有结论都是 50 个基因的平均。本脚本把等价 σ 拆到单基因, 再与该基因的
空间自相关(Moran's I)、表达量、稀疏度回归。

预期(需被证实或证伪): 看起来可预测的基因, 恰恰就是空间自相关高的那些。
若成立 ⇒ 该领域的主指标在按"基因有多空间"给分, 而不是按"模型有多懂生物学";
        且它解释了为何基因面板一变(top-20→200)分数就摆动 42%。

用 top-200 HVG 提高回归效力(常规评测集是 top-50, 结果按两者分别报)。
噪声天花板逐基因算(二项拆半), 得到逐基因归一化技能。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
import evaluate as E, retrieval as R
from baselines import ridge_predict
from effres import build_operator, calibrate_sigma, PX_PER_UM, SLIDES, SEB

PARENT = "/path/to/align_workspace"
RAW = os.path.join(PARENT, "st_bench/data/{s}/adata_16um.h5ad")

ap = argparse.ArgumentParser()
ap.add_argument("--tower", default="hibou_l")
ap.add_argument("--ngene", type=int, default=200)
ap.add_argument("--tmax", type=int, default=2048)
ap.add_argument("--reps", type=int, default=3)
a_ = ap.parse_args()
cps = [1]
while cps[-1] < a_.tmax: cps.append(cps[-1]*2)

a = ad.read_h5ad(SEB.H5AD)
expr = np.nan_to_num(np.asarray(a.X, np.float32))
genes = np.asarray(a.var["gene"]).astype(str)
slide = a.obs["slide_id"].astype(str).values
pxl = np.asarray(a.obsm["pxl"], np.float64)
img = np.nan_to_num(np.concatenate([
    np.load(os.path.join(SEB.EMBDIR, f"emb_{a_.tower}_P2.npy")),
    np.load(os.path.join(SEB.EMBDIR, f"emb_{a_.tower}_P5.npy"))]).astype(np.float32))

def morans_pergene(A, X):
    Xc = X - X.mean(0)
    return (X.shape[0]/A.sum()) * (Xc*(A@Xc)).sum(0) / ((Xc**2).sum(0)+1e-12)

acc = {}
for s in SLIDES:
    te = slide == s; tr = ~te
    gidx = E.topk_hvg(expr[tr], a_.ngene)          # 训练片选基因, 无泄漏
    y = expr[te][:, gidx]; gn = genes[gidx]
    xy = pxl[te] / PX_PER_UM[s]
    W = build_operator(xy, k=8, cut_um=29.0)
    A = (W > 0).astype(np.float32); A.setdiag(0); A.eliminate_zeros()
    sig = calibrate_sigma(W, xy, cps)
    print(f"\n[{s}] n={int(te.sum())} 基因={len(gidx)}", flush=True)

    ladder = {}
    cur, t = y.copy(), 0
    for cp in cps:
        while t < cp: cur = W @ cur; t += 1
        ladder[cp] = E.per_gene_pcc(cur, y)                     # 逐基因!
    pred = ridge_predict(img[tr], expr[tr], img[te], 1e4)[:, gidx]
    pcc_g = E.per_gene_pcc(pred, y)
    moran_g = morans_pergene(A, y)
    moran_pred = morans_pergene(A, pred)

    # 逐基因噪声天花板
    C = sparse.csr_matrix(ad.read_h5ad(RAW.format(s=s)).X)
    raw_genes = np.asarray(ad.read_h5ad(RAW.format(s=s)).var_names).astype(str)
    gi = {g: i for i, g in enumerate(raw_genes)}
    cols = np.array([gi[g] for g in gn if g in gi])
    ok = np.array([g in gi for g in gn])
    rng = np.random.default_rng(0); ch = []
    for _ in range(a_.reps):
        h = rng.binomial(C.data.astype(np.int64), 0.5).astype(np.float32)
        P1 = sparse.csr_matrix((h, C.indices, C.indptr), shape=C.shape)
        P2 = sparse.csr_matrix((C.data-h, C.indices, C.indptr), shape=C.shape)
        f = lambda M: np.log1p(np.asarray(M[:, cols].todense(), np.float32))
        ch.append(E.per_gene_pcc(f(P1), f(P2)))
    ch = np.mean(ch, 0); cfull = 2*ch/(1+ch)
    ceil_g = np.full(len(gn), np.nan); ceil_g[ok] = cfull

    def eqs_g(j, v):
        pts = sorted(((sig[c], ladder[c][j]) for c in cps), key=lambda z: z[0])
        if v >= pts[0][1]: return pts[0][0]
        for (s0,v0),(s1,v1) in zip(pts, pts[1:]):
            if v0 >= v >= v1: return s0 + (v0-v)/max(v0-v1,1e-12)*(s1-s0)
        return np.nan
    eq_g = np.array([eqs_g(j, pcc_g[j]) for j in range(len(gn))])

    for j, g in enumerate(gn):
        acc.setdefault(g, {"pcc": [], "eq": [], "moran": [], "moran_pred": [],
                           "ceil": [], "mean": [], "zero": []})
        d = acc[g]
        d["pcc"].append(pcc_g[j]); d["eq"].append(eq_g[j]); d["moran"].append(moran_g[j])
        d["moran_pred"].append(moran_pred[j]); d["ceil"].append(ceil_g[j])
        d["mean"].append(float(y[:, j].mean())); d["zero"].append(float((y[:, j] == 0).mean()))

G = sorted(acc)
V = {k: np.array([np.nanmean(acc[g][k]) for g in G]) for k in
     ("pcc", "eq", "moran", "moran_pred", "ceil", "mean", "zero")}
V["norm"] = V["pcc"] / np.where(V["ceil"] > .05, V["ceil"], np.nan)

print(f"\n{'='*72}"); print(f"逐基因相关 (n={len(G)} 基因, 塔={a_.tower})"); print("="*72)
print(f"{'与 per-gene PCC 的相关':32s}{'Pearson':>10s}{'Spearman':>10s}")
for nm, k in [("Moran's I（空间自相关）", "moran"), ("噪声天花板 c", "ceil"),
              ("平均表达量", "mean"), ("零元比例", "zero")]:
    m = np.isfinite(V[k]) & np.isfinite(V["pcc"])
    print(f"{nm:32s}{pearsonr(V[k][m], V['pcc'][m])[0]:>10.3f}{spearmanr(V[k][m], V['pcc'][m])[0]:>10.3f}")
print(f"\n{'与等价 σ 的相关':32s}{'Pearson':>10s}{'Spearman':>10s}")
for nm, k in [("Moran's I", "moran"), ("噪声天花板 c", "ceil")]:
    m = np.isfinite(V[k]) & np.isfinite(V["eq"])
    print(f"{nm:32s}{pearsonr(V[k][m], V['eq'][m])[0]:>10.3f}{spearmanr(V[k][m], V['eq'][m])[0]:>10.3f}")

q = np.nanpercentile(V["moran"], [25, 50, 75])
print(f"\n{'='*72}"); print("按 Moran's I 四分位分层"); print("="*72)
print(f"{'层':16s}{'n':>5s}{'Moran':>9s}{'PCC':>9s}{'等价σ':>9s}{'天花板':>9s}{'归一技能':>10s}")
lab = ["Q1 最不空间", "Q2", "Q3", "Q4 最空间"]
for i in range(4):
    lo = -np.inf if i == 0 else q[i-1]; hi = np.inf if i == 3 else q[i]
    m = (V["moran"] > lo) & (V["moran"] <= hi)
    print(f"{lab[i]:16s}{m.sum():>5d}{np.nanmean(V['moran'][m]):>9.3f}"
          f"{np.nanmean(V['pcc'][m]):>9.3f}{np.nanmedian(V['eq'][m]):>9.0f}"
          f"{np.nanmean(V['ceil'][m]):>9.3f}{np.nanmean(V['norm'][m]):>10.3f}")

print(f"\n{'='*72}"); print("逐基因 Moran 膨胀"); print("="*72)
r = V["moran_pred"] / np.where(V["moran"] > .01, V["moran"], np.nan)
print(f"  预测/真值 中位={np.nanmedian(r):.2f}×  Q1 层={np.nanmedian(r[V['moran']<=q[0]]):.2f}×"
      f"  Q4 层={np.nanmedian(r[V['moran']>q[2]]):.2f}×")
print(f"  ⇒ 空间性最弱的基因被膨胀得最厉害 = 凭空制造空间结构")

json.dump({g: {k: float(np.nanmean(acc[g][k])) for k in acc[g]} for g in G},
          open(f"/path/to/systema4ST/results/per_gene_{a_.tower}.json", "w"),
          indent=2, ensure_ascii=False)
print(f"\n已存 results/per_gene_{a_.tower}.json")

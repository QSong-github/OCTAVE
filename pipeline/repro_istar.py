"""复现 method_rank 里 iStar 的 nan —— 逐步打印中间量，定位到底哪一步崩的。"""
import numpy as np, anndata as ad
from scipy import sparse

E = "/path/to/upstream_align/exports/predictions_withinslide.h5ad"
a = ad.read_h5ad(E)
X = np.asarray(a.X.todense() if sparse.issparse(a.X) else a.X, np.float32)
sl = a.obs["slide_id"].astype(str).values
sp = a.obs["split"].astype(str).values
s = "Visium_HD_Human_Colon_Cancer_P2"
te = (sl == s) & (sp == "test"); tr = (sl == s) & (sp == "train")
gidx = np.argsort(-X[tr].var(0))[:50]
y = X[te][:, gidx]

for m in ["baseline_iStar_official", "baseline_Ridge_HEST"]:
    L = a.layers[m]
    print(f"\n=== {m} ===")
    print(f"  layer 类型 {type(L).__name__}  dtype {getattr(L,'dtype',None)}  稀疏 {sparse.issparse(L)}")
    P = np.asarray(L[te][:, gidx], np.float32)
    print(f"  P.shape {P.shape} dtype {P.dtype}  有限值占比 {100*np.isfinite(P).mean():.2f}%")
    print(f"  P 逐列 std 前5: {np.round(P.std(0)[:5], 4)}")
    p = P - P.mean(0); t = y - y.mean(0)
    pss = (p ** 2).sum(0); tss = (t ** 2).sum(0)
    print(f"  (p^2).sum 前5: {pss[:5]}")
    print(f"  (t^2).sum 前5: {tss[:5]}")
    den = np.sqrt(pss * tss)
    print(f"  den 前5: {den[:5]}   den<=1e-8 的列数: {int((den <= 1e-8).sum())}/50")
    print(f"  pss*tss 是否溢出 inf: {int(np.isinf(pss*tss).sum())} 列")
    r = np.where(den > 1e-8, (p * t).sum(0) / den, np.nan)
    print(f"  相关 nan 列数 {int(np.isnan(r).sum())}/50   均值 {np.nanmean(r) if np.isfinite(r).any() else 'ALL-NAN'}")
    # float64 对照
    p64 = P.astype(np.float64) - P.astype(np.float64).mean(0)
    t64 = y.astype(np.float64) - y.astype(np.float64).mean(0)
    d64 = np.sqrt((p64**2).sum(0) * (t64**2).sum(0))
    r64 = np.where(d64 > 1e-8, (p64*t64).sum(0)/d64, np.nan)
    print(f"  float64 重算: nan 列 {int(np.isnan(r64).sum())}/50   均值 {np.nanmean(r64):.4f}")

import numpy as np, anndata as ad
from scipy import sparse
E = "/path/to/upstream_align/exports/predictions_withinslide.h5ad"
a = ad.read_h5ad(E)
X = np.asarray(a.X.todense() if sparse.issparse(a.X) else a.X, np.float32)
sl = a.obs["slide_id"].astype(str).values
sp = a.obs["split"].astype(str).values
for s in sorted(set(sl)):
    te = (sl == s) & (sp == "test"); tr = (sl == s) & (sp == "train")
    gidx = np.argsort(-X[tr].var(0))[:50]
    P = np.asarray(a.layers["baseline_iStar_official"][te][:, gidx], np.float32)
    y = X[te][:, gidx]
    print(f"\n[{s}] test={int(te.sum())}")
    print(f"  iStar: nan占比 {100*np.isnan(P).mean():.2f}%  全nan列 {int(np.isnan(P).all(0).sum())}/50")
    with np.errstate(all="ignore"):
        sd = np.nanstd(P, 0)
    print(f"  逐列标准差: 最小 {np.nanmin(sd):.3e}  中位 {np.nanmedian(sd):.3e}  零方差列 {int((sd < 1e-8).sum())}")
    print(f"  真值 y 标准差: 最小 {y.std(0).min():.3e}  中位 {np.median(y.std(0)):.3e}")
    print(f"  iStar 值域 [{np.nanmin(P):.4f}, {np.nanmax(P):.4f}]  全零行 {int((P == 0).all(1).sum())}")
    # 与其他方法对照
    for m in ["baseline_Ridge_HEST", "baseline_imageKNN"]:
        Q = np.asarray(a.layers[m][te][:, gidx], np.float32)
        print(f"  对照 {m}: 零方差列 {int((Q.std(0) < 1e-8).sum())}  值域 [{Q.min():.3f},{Q.max():.3f}]")

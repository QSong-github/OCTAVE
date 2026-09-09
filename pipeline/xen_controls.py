# -*- coding: utf-8 -*-
"""审稿意见（第二份）#1/#4：Xenium 上的分区对照与已知退化恢复实验。
对照：坐标 K-means 分区 oracle、随机等规模分区 oracle、仅用训练块估计域均值的可学习域预测器（同 16×16 块 CV）。
退化：对实测场施加 模糊/白噪声/边界平移/热点抹除，各自把强度调到与岭回归相同的标量 PCC，再比较各带 β_i。"""
import argparse, json, os, sys, numpy as np, anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
sys.path.insert(0, "/path/to/systema4ST/src")
from per_gene_xen import build_operator, calibrate_sigma, per_gene_pcc, block_cv_predict
PREP = "/path/to/systema4ST/data/prepped_xen"; EMB = "/path/to/systema4ST/results/emb_xen"
OUTD = "/path/to/systema4ST/results/xen_controls"
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--tower", default="hibou_l")
ap.add_argument("--ngene", type=int, default=200); ap.add_argument("--tmax", type=int, default=2048); a = ap.parse_args()
cps = [1]
while cps[-1] < a.tmax: cps.append(cps[-1] * 2)
A_ = ad.read_h5ad(f"{PREP}/{a.name}_bin16.h5ad"); px = float(A_.uns["px_per_um"]); xy = np.asarray(A_.obsm["pxl"], np.float64) / px
Y_all = np.log1p(np.asarray(sparse.csr_matrix(A_.X).todense(), np.float32))
X = np.nan_to_num(np.load(f"{EMB}/emb_{a.tower}_{a.name}.npy").astype(np.float32)); assert X.shape[0] == Y_all.shape[0]
gidx = np.argsort(-Y_all.var(0))[:a.ngene]; Y = Y_all[:, gidx]; n = len(xy)
W = build_operator(xy); sig = calibrate_sigma(W, xy, cps)
P = block_cv_predict(X, Y, xy); ok = np.isfinite(P).all(1)
def group_means(Y, lab):
    out = np.zeros_like(Y); 
    for u in np.unique(lab): m = lab == u; out[m] = Y[m].mean(0)
    return out
def folds(xy, grid=16):
    qx = np.quantile(xy[:, 0], np.linspace(0, 1, grid + 1)); qx[-1] += 1
    qy = np.quantile(xy[:, 1], np.linspace(0, 1, grid + 1)); qy[-1] += 1
    return (np.searchsorted(qx, xy[:, 0], "right") - 1) * grid + (np.searchsorted(qy, xy[:, 1], "right") - 1)
fold = folds(xy)
def train_only_means(Y, lab):
    """可学习域预测器：域标签只来自图像，域均值只用训练块估计（同 block_cv_predict 的折与门槛）。"""
    Pt = np.full_like(Y, np.nan)
    for f in np.unique(fold):
        te = fold == f
        if te.sum() < 20 or (~te).sum() < 2000: continue
        mu = {u: Y[(~te) & (lab == u)].mean(0) for u in np.unique(lab[te]) if ((~te) & (lab == u)).any()}
        g = Y[~te].mean(0)
        Pt[te] = np.stack([mu.get(u, g) for u in lab[te]])
    return Pt
def score(Pm): return float(np.nanmean(per_gene_pcc(Pm[ok], Y[ok])))
def bands(M):
    out, prev, low, t = {}, M.copy(), M.copy(), 0
    for cp in cps:
        while t < cp: prev = W @ prev; t += 1
        out[cp] = low - prev; low = prev.copy()
    return out
BT = bands(Y)
def profile(Pm):
    BM = bands(Pm); return {str(cp): float(np.nanmean(per_gene_pcc(BM[cp][ok], BT[cp][ok]))) for cp in cps}
pc = PCA(n_components=50, random_state=0).fit_transform(X)
rng = np.random.default_rng(0); res = {"name": a.name, "n": int(n), "sigma_um": {str(c): float(sig[c]) for c in cps}, "ridge": {"pcc": score(P), "bands": profile(P)}}
for K in (20, 200):
    lab_img = KMeans(n_clusters=K, n_init=4, random_state=0).fit_predict(pc)
    lab_xy = KMeans(n_clusters=K, n_init=4, random_state=0).fit_predict(xy)
    lab_rand = rng.permutation(lab_img)
    O = group_means(Y, lab_img)
    res[f"K{K}"] = {"oracle_image": {"pcc": score(O), "bands": profile(O)},
                    "oracle_coord": {"pcc": score(group_means(Y, lab_xy))},
                    "oracle_random_matched": {"pcc": score(group_means(Y, lab_rand))},
                    "trainonly_image": {"pcc": score(train_only_means(Y, lab_img))},
                    "trainonly_coord": {"pcc": score(train_only_means(Y, lab_xy))}}
    print(f"[{a.name}] K={K}: ridge {res['ridge']['pcc']:.3f} | oracle img {res[f'K{K}']['oracle_image']['pcc']:.3f} coord {res[f'K{K}']['oracle_coord']['pcc']:.3f} rand {res[f'K{K}']['oracle_random_matched']['pcc']:.3f} | train-only img {res[f'K{K}']['trainonly_image']['pcc']:.3f} coord {res[f'K{K}']['trainonly_coord']['pcc']:.3f}", flush=True)
# ── 已知退化：强度匹配到岭回归的标量 PCC ──
target = res["ridge"]["pcc"]; sd = Y.std(0) + 1e-8
Wp = {}; cur = Y.copy(); Wp[0] = cur
for t in range(1, 65): cur = W @ cur; Wp[t] = cur
def blur(s):  # s ∈ [0,64] 连续扩散时间（整数步间线性插值）
    t0 = int(np.floor(s)); w = s - t0; return Wp[t0] if t0 >= 64 else (1 - w) * Wp[t0] + w * Wp[t0 + 1]
noise = rng.standard_normal(Y.shape).astype(np.float32) * sd
def white(s): return Y + s * noise
tree = cKDTree(xy)
def shifted(dx_um):
    _, j = tree.query(xy - np.array([dx_um, 0.0])); return Y[j]
def shift(s):  # s = 平移量（µm），最近邻取值；<16 µm 时与原场线性混合
    if s <= 0: return Y
    if s < 16: return (1 - s / 16) * Y + (s / 16) * shifted(16.0)
    return shifted(s)
S4 = Wp[4]
def hotspot(q):  # 抹除每个基因最高 q 比例的值，换成 4 步平滑值
    if q <= 0: return Y
    thr = np.quantile(Y, 1 - q, axis=0); M = Y > thr; return np.where(M, S4, Y)
def match(fn, lo, hi, tol=0.004, it=40):
    if score(fn(hi)) > target: return hi, score(fn(hi)), "hi-limited"
    for _ in range(it):
        mid = 0.5 * (lo + hi); v = score(fn(mid))
        if abs(v - target) < tol: return mid, v, "matched"
        if v > target: lo = mid
        else: hi = mid
    return mid, v, "approx"
res["degradations"] = {}
for nm, fn, lo, hi in [("blur", blur, 0.0, 64.0), ("white_noise", white, 0.0, 20.0), ("boundary_shift", shift, 0.0, 200.0), ("hotspot_removal", hotspot, 0.0, 0.6)]:
    s, v, flag = match(fn, lo, hi); D = fn(s)
    res["degradations"][nm] = {"strength": float(s), "pcc": v, "flag": flag, "bands": profile(D)}
    print(f"  退化 {nm}: 强度 {s:.3g} → PCC {v:.3f} ({flag}); β1 {res['degradations'][nm]['bands']['1']:.3f}", flush=True)
os.makedirs(OUTD, exist_ok=True); json.dump(res, open(f"{OUTD}/{a.name}.json", "w"), indent=1); print("→ 写出", flush=True)

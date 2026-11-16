# -*- coding: utf-8 -*-
"""同一组预测上，OCTAVE 与几种简单的多尺度替代指标并排算。

预测与 blocks_xen_bands.py / xen_controls.py 完全同协议（16 µm 栅格、top-200 HVG、16×16 块 CV 岭回归、
图像 PCA50→KMeans 域预言机、四种退化按岭回归标量 PCC 匹配强度），只是多算几种指标：
  pcc       标量逐基因 PCC 均值
  octave    图扩散二进带逐带 PCC（论文本身的指标）
  gauss_lp  高斯低通后的 PCC（同一物理宽度 σ_i）
  gauss_dog 高斯差分带的 PCC（σ_0=0，σ_i 取图扩散标定出的物理宽度）
  ssim      逐基因图像 SSIM（7×7 窗口，栅格上，仅组织内像素取均值）
MSSI 需要另一个环境，这里把预测存成 npz，交给 xen_mssi.py。
"""
import argparse, json, os, sys
import numpy as np
import anndata as ad
import h5py, pyarrow.parquet as pq
from scipy import sparse, ndimage
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity

ROOT = "/path/to/project"
sys.path.insert(0, f"{ROOT}/src")
from per_gene_xen import build_operator, calibrate_sigma, per_gene_pcc, block_cv_predict

PREP, RAW, EMB = f"{ROOT}/data/prepped_xen", f"{ROOT}/data/xenium", f"{ROOT}/results/emb_xen"
OUTD, NPZD = f"{ROOT}/results/xen_altmetrics", f"{ROOT}/results/xen_altmetrics_npz"


def grid_index(name, xy_um, bin_um=16.0):
    """复原 xen_prep.py 的 16 µm 栅格整数坐标：分箱质心落在 [m0+16b, m0+16(b+1))，m0 为保留细胞的最小坐标。"""
    d = f"{RAW}/{name}"
    C = pq.read_table(f"{d}/cells.parquet", columns=["cell_id", "x_centroid", "y_centroid"]).to_pandas()
    with h5py.File(f"{d}/cell_feature_matrix.h5", "r") as f:
        bc = set(x.decode() for x in f["matrix/barcodes"][:])
    keep = C["cell_id"].astype(str).isin(bc).to_numpy()
    m0 = np.array([C["x_centroid"].to_numpy()[keep].min(), C["y_centroid"].to_numpy()[keep].min()])
    g = np.floor((xy_um - m0) / bin_um).astype(np.int64)
    assert (g >= 0).all() and len(np.unique(g[:, 0] * (g[:, 1].max() + 2) + g[:, 1])) == len(g), "栅格坐标冲突"
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tower", default="hibou_l")
    ap.add_argument("--ngene", type=int, default=200)
    ap.add_argument("--tmax", type=int, default=2048)
    a = ap.parse_args()
    cps = [1]
    while cps[-1] < a.tmax:
        cps.append(cps[-1] * 2)

    A_ = ad.read_h5ad(f"{PREP}/{a.name}_bin16.h5ad")
    px = float(A_.uns["px_per_um"])
    xy = np.asarray(A_.obsm["pxl"], np.float64) / px
    Y_all = np.log1p(np.asarray(sparse.csr_matrix(A_.X).todense(), np.float32))
    X = np.nan_to_num(np.load(f"{EMB}/emb_{a.tower}_{a.name}.npy").astype(np.float32))
    assert X.shape[0] == Y_all.shape[0]
    gidx = np.argsort(-Y_all.var(0))[:a.ngene]
    Y = Y_all[:, gidx]; n = len(xy)
    gnames = np.asarray(A_.var_names)[gidx].astype(str)
    print(f"[{a.name}] n={n} bin", flush=True)

    W = build_operator(xy)
    sig = calibrate_sigma(W, xy, cps)
    P = block_cv_predict(X, Y, xy); ok = np.isfinite(P).all(1)

    # 域预言机（K=20 与 200），随机数序列与 xen_controls.py 一致，使白噪声退化可逐位复现
    pc = PCA(n_components=50, random_state=0).fit_transform(X)
    rng = np.random.default_rng(0)
    DOM = {}
    for K in (20, 200):
        lab = KMeans(n_clusters=K, n_init=4, random_state=0).fit_predict(pc)
        out = np.zeros_like(Y)
        for u in np.unique(lab):
            m = lab == u; out[m] = Y[m].mean(0)
        DOM[K] = out
        rng.permutation(lab)

    def score(M): return float(np.nanmean(per_gene_pcc(M[ok], Y[ok])))
    target = score(P); sd = Y.std(0) + 1e-8
    Wp = {0: Y.copy()}; cur = Y.copy()
    for t in range(1, 65):
        cur = W @ cur; Wp[t] = cur
    def blur(s):
        t0 = int(np.floor(s)); w = s - t0
        return Wp[t0] if t0 >= 64 else (1 - w) * Wp[t0] + w * Wp[t0 + 1]
    noise = rng.standard_normal(Y.shape).astype(np.float32) * sd
    def white(s): return Y + s * noise
    tree = cKDTree(xy)
    def shifted(dx):
        _, j = tree.query(xy - np.array([dx, 0.0])); return Y[j]
    def shift(s):
        if s <= 0: return Y
        if s < 16: return (1 - s / 16) * Y + (s / 16) * shifted(16.0)
        return shifted(s)
    S4 = Wp[4]
    def hotspot(q):
        if q <= 0: return Y
        thr = np.quantile(Y, 1 - q, axis=0); return np.where(Y > thr, S4, Y)
    def match(fn, lo, hi, tol=0.004, it=40):
        if score(fn(hi)) > target: return hi, "hi-limited"
        for _ in range(it):
            mid = 0.5 * (lo + hi); v = score(fn(mid))
            if abs(v - target) < tol: return mid, "matched"
            if v > target: lo = mid
            else: hi = mid
        return mid, "approx"
    preds = {"ridge": P, "dom20": DOM[20]}   # K=200 只为与 xen_controls.py 保持随机数序列，不进比较
    strengths = {}
    for nm, fn, lo, hi in [("blur", blur, 0.0, 64.0), ("white_noise", white, 0.0, 20.0),
                           ("boundary_shift", shift, 0.0, 200.0), ("hotspot_removal", hotspot, 0.0, 0.6)]:
        s, flag = match(fn, lo, hi); preds[nm] = np.asarray(fn(s), np.float32); strengths[nm] = [float(s), flag]
    del Wp

    # ── OCTAVE 二进带 ──
    def octave(M):
        out, prev, low, t = {}, M.copy(), M.copy(), 0
        for cp in cps:
            while t < cp:
                prev = W @ prev; t += 1
            out[cp] = low - prev; low = prev.copy()
        return out
    OT = octave(Y)

    # ── 高斯：栅格上归一化卷积（组织外与块 CV 未覆盖的 bin 不参与）──
    g = grid_index(a.name, np.asarray(A_.obsm["xy_um"], np.float64))
    H, Wd = g[:, 0].max() + 1, g[:, 1].max() + 1
    mask = np.zeros((H, Wd), np.float32); mask[g[ok, 0], g[ok, 1]] = 1.0
    sig_px = [sig[c] / 16.0 for c in cps]
    def raster(M):
        R = np.zeros((H, Wd, M.shape[1]), np.float32); R[g[ok, 0], g[ok, 1]] = M[ok]; return R
    def gsmooth(R, s):
        num = ndimage.gaussian_filter(R, sigma=(s, s, 0), mode="constant", truncate=3.0)
        den = ndimage.gaussian_filter(mask, sigma=s, mode="constant", truncate=3.0)[..., None]
        return (num / np.maximum(den, 1e-6))[g[ok, 0], g[ok, 1]]
    def gauss_all(M):
        R = raster(M); low = {0: M[ok]}
        for i, s in enumerate(sig_px, 1):
            low[i] = gsmooth(R, s)
        return low
    GT = gauss_all(Y)

    def ssim_mean(M):
        RT, RM = raster(Y), raster(M); v = []
        for j in range(Y.shape[1]):
            rng_j = float(RT[..., j][mask > 0].max() - RT[..., j][mask > 0].min()) or 1.0
            _, S = structural_similarity(RT[..., j], RM[..., j], data_range=rng_j, full=True)
            v.append(float(S[mask > 0].mean()))
        return float(np.mean(v))

    def pcc_ok(A, B): return float(np.nanmean(per_gene_pcc(A, B)))
    res = {"name": a.name, "tower": a.tower, "n_bins": int(n), "n_ok": int(ok.sum()), "genes": gnames.tolist(),
           "sigma_um": {str(c): float(sig[c]) for c in cps}, "degradation_strengths": strengths, "pred": {}}
    for nm, M in preds.items():
        Mf = np.nan_to_num(M); OM = octave(Mf); GM = gauss_all(Mf)
        r = {"pcc": pcc_ok(Mf[ok], Y[ok]),
             "octave": {str(c): pcc_ok(OM[c][ok], OT[c][ok]) for c in cps},
             "gauss_lp": {str(c): pcc_ok(GM[i], GT[i]) for i, c in enumerate(cps, 1)},
             "gauss_dog": {str(c): pcc_ok(GM[i - 1] - GM[i], GT[i - 1] - GT[i]) for i, c in enumerate(cps, 1)},
             "ssim": ssim_mean(Mf)}
        res["pred"][nm] = r
        print(f"  {nm:<16s} PCC {r['pcc']:.4f}  OCTAVE β1 {r['octave']['1']:.4f}  DoG β1 {r['gauss_dog']['1']:.4f}  "
              f"SSIM {r['ssim']:.4f}", flush=True)

    # 与已存结果核对（同协议应一致；KMeans 受线程数影响，预言机可有微小差异）
    old = f"{ROOT}/results/xen_controls/{a.name}.json"
    if os.path.exists(old):
        o = json.load(open(old))
        res["check_vs_xen_controls"] = {"ridge_pcc": [o["ridge"]["pcc"], res["pred"]["ridge"]["pcc"]],
                                        "dom20_pcc": [o["K20"]["oracle_image"]["pcc"], res["pred"]["dom20"]["pcc"]]}
        print("  核对 xen_controls:", res["check_vs_xen_controls"], flush=True)
    os.makedirs(OUTD, exist_ok=True); os.makedirs(NPZD, exist_ok=True)
    json.dump(res, open(f"{OUTD}/{a.name}.json", "w"), indent=1)
    np.savez(f"{NPZD}/{a.name}.npz", xy=xy[ok].astype(np.float32), grid=g[ok], truth=Y[ok].astype(np.float16),
             genes=gnames, **{k: np.nan_to_num(v)[ok].astype(np.float16) for k, v in preds.items()})
    print(f"→ {OUTD}/{a.name}.json", flush=True)


if __name__ == "__main__":
    main()

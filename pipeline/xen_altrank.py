# -*- coding: utf-8 -*-
"""57 编码器 × 22 区域，同一组岭回归预测上算 标量 PCC / OCTAVE 最细带 / 高斯差分最细带 / SSIM / MSSI，
比较各指标给出的编码器排序。岭回归与 blocks_xen_bands.py 同协议；每个任务与已存结果核对 PCC 与最细带。
MSSI 在 mssi 环境里用子进程算，粗化算子复用 xen_mssi.py 存下的按区域算子。"""
import argparse, json, os, sys, subprocess, tempfile
import numpy as np
import anndata as ad
from scipy import sparse, ndimage
from skimage.metrics import structural_similarity

ROOT = "/path/to/project"
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, ROOT)
from per_gene_xen import build_operator, calibrate_sigma, per_gene_pcc, block_cv_predict
from xen_altmetrics import grid_index

PREP, EMB = f"{ROOT}/data/prepped_xen", f"{ROOT}/results/emb_xen"
OUTD, PREDD = f"{ROOT}/results/xen_altrank", f"{ROOT}/results/xen_altrank_pred"
MSSI_PY = "/path/to/miniconda3/envs/mssi/bin/python"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--tower", required=True)
    ap.add_argument("--save_pred", action="store_true", help="存下预测（定性图用）")
    a = ap.parse_args()
    out_f = f"{OUTD}/{a.tower}/{a.name}.json"
    A_ = ad.read_h5ad(f"{PREP}/{a.name}_bin16.h5ad")
    px = float(A_.uns["px_per_um"]); xy = np.asarray(A_.obsm["pxl"], np.float64) / px
    Y_all = np.log1p(np.asarray(sparse.csr_matrix(A_.X).todense(), np.float32))
    X = np.nan_to_num(np.load(f"{EMB}/emb_{a.tower}_{a.name}.npy").astype(np.float32))
    assert X.shape[0] == Y_all.shape[0]
    gidx = np.argsort(-Y_all.var(0))[:200]; Y = Y_all[:, gidx]
    W = build_operator(xy); sig1 = calibrate_sigma(W, xy, [1])[1]
    P = block_cv_predict(X, Y, xy); ok = np.isfinite(P).all(1); Pf = np.nan_to_num(P)

    def pcc(A, B): return float(np.nanmean(per_gene_pcc(A, B)))
    r = {"name": a.name, "tower": a.tower, "n_ok": int(ok.sum()), "sigma1_um": float(sig1)}
    r["pcc"] = pcc(Pf[ok], Y[ok])
    r["octave_b1"] = pcc((Pf - W @ Pf)[ok], (Y - W @ Y)[ok])

    g = grid_index(a.name, np.asarray(A_.obsm["xy_um"], np.float64))
    H, Wd = g[:, 0].max() + 1, g[:, 1].max() + 1
    mask = np.zeros((H, Wd), np.float32); mask[g[ok, 0], g[ok, 1]] = 1.0
    def raster(M):
        R = np.zeros((H, Wd, M.shape[1]), np.float32); R[g[ok, 0], g[ok, 1]] = M[ok]; return R
    den = np.maximum(ndimage.gaussian_filter(mask, sigma=sig1 / 16.0, mode="constant", truncate=3.0), 1e-6)[..., None]
    def dog1(M):
        R = raster(M)
        return M[ok] - (ndimage.gaussian_filter(R, sigma=(sig1 / 16.0,) * 2 + (0,), mode="constant", truncate=3.0) / den)[g[ok, 0], g[ok, 1]]
    r["dog_b1"] = pcc(dog1(Pf), dog1(Y))
    RT, RM = raster(Y), raster(Pf); v = []
    for j in range(Y.shape[1]):
        rg = float(RT[..., j][mask > 0].max() - RT[..., j][mask > 0].min()) or 1.0
        _, S = structural_similarity(RT[..., j], RM[..., j], data_range=rg, full=True); v.append(float(S[mask > 0].mean()))
    r["ssim"] = float(np.mean(v))

    ops = f"{ROOT}/results/xen_mssi_ops/{a.name}.pkl"
    try:   # MSSI 失败不丢岭回归结果
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as td:
            np.savez(f"{td}/p.npz", truth=Y[ok].astype(np.float16), pred=Pf[ok].astype(np.float16))
            subprocess.run([MSSI_PY, f"{ROOT}/xen_mssi_one.py", "--npz", f"{td}/p.npz", "--ops", ops, "--out", f"{td}/m.json"], check=True)
            r["mssi"] = json.load(open(f"{td}/m.json"))["mssi"]
    except Exception as e:
        r["mssi"] = None; r["mssi_error"] = str(e)[:300]
        if not os.path.exists(ops):   # 该区域暂无 MSSI 算子：存下预测，日后补算 MSSI，不必重跑岭回归
            a.save_pred = True
    if r["mssi"] is None and os.environ.get("REQUIRE_MSSI"):   # 只在试跑任务里设，MSSI 不通就不放行全量
        raise SystemExit("MSSI failed: " + r.get("mssi_error", ""))

    for old in [f"{ROOT}/results/blocks_xen_bands_{a.tower}/{a.name}.json"] + \
               ([f"{ROOT}/results/blocks_xen_bands/{a.name}.json"] if a.tower == "hibou_l" else []):
        if os.path.exists(old):
            o = json.load(open(old))["pred"]["ridge"]; r["check_stored"] = {"pcc": o["pcc"], "band_pcc": o["band_pcc"]}; break
    os.makedirs(os.path.dirname(out_f), exist_ok=True); json.dump(r, open(out_f, "w"), indent=1)
    if a.save_pred:
        os.makedirs(PREDD, exist_ok=True)
        np.savez(f"{PREDD}/{a.tower}_{a.name}.npz", grid=g[ok], pred=Pf[ok].astype(np.float16),
                 genes=np.asarray(A_.var_names)[gidx].astype(str))
    print(json.dumps({k: r[k] for k in ("pcc", "octave_b1", "dog_b1", "ssim", "mssi")}), r.get("check_stored"), flush=True)


if __name__ == "__main__":
    main()

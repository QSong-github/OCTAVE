# -*- coding: utf-8 -*-
"""把 Xenium 区域转成官方 iStar 输入（与 src/istar_prep.py 的 Visium HD 版同构）。
image：每区域一次 → he-raw.jpg（金字塔层 ~1 µm/px）、pixel-size-raw.txt、pixel-size.txt、level-downsample.txt、heraw-dims.txt。
fold：train/test 用 within_bench.make_split(xy_um, "half", 320 µm 块, 64 µm 隔离带)，与 Visium HD 折同协议；
     cnts.tsv 喂训练 bin 的 log1p 表达（200 个最高方差基因，与 blocks_xen_bands 同选法），locs-raw.tsv 为 he-raw 像素坐标，
     test.npz 存测试 bin 的像素坐标、µm 坐标、真值与基因名。"""
import os, sys, argparse, numpy as np, anndata as ad, pandas as pd
from scipy import sparse
sys.path.insert(0, "/path/to/systema4ST/src")
from within_bench import make_split
from per_gene_xen import gene_names
PREP = "/path/to/systema4ST/data/prepped_xen"; XEN = "/path/to/systema4ST/data/xenium"
def tif_path(N):
    for f in (f"{XEN}/{N}/{N}_PYRAMIDAL.tif", f"{XEN}/{N}/{N}_he_image.ome.tif"):
        if os.path.exists(f): return f
    raise SystemExit(f"无 H&E: {N}")
def write_image(N, outdir, target_um=1.0):
    import openslide
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    A = ad.read_h5ad(f"{PREP}/{N}_bin16.h5ad", backed="r"); ppu = float(A.uns["px_per_um"]); umpx0 = 1.0 / ppu
    so = openslide.OpenSlide(tif_path(N))
    # 选下采样最接近 target_um/umpx0 的金字塔层（Visium HD 版取 level 2 ≈ 1.1 µm/px）
    want = target_um / umpx0; lvl = int(np.argmin([abs(d - want) for d in so.level_downsamples])); ds = float(so.level_downsamples[lvl]); W, H = so.level_dimensions[lvl]
    img = so.read_region((0, 0), lvl, (W, H)).convert("RGB"); os.makedirs(outdir, exist_ok=True); img.save(os.path.join(outdir, "he-raw.jpg"), quality=92)
    open(os.path.join(outdir, "pixel-size-raw.txt"), "w").write(f"{umpx0*ds:.6f}\n"); open(os.path.join(outdir, "pixel-size.txt"), "w").write(f"{target_um}\n")
    open(os.path.join(outdir, "level-downsample.txt"), "w").write(f"{ds}\n"); open(os.path.join(outdir, "heraw-dims.txt"), "w").write(f"{W} {H}\n")
    print(f"[{N}] 全分辨率 {umpx0:.4f} µm/px; 层{lvl} 下采样 {ds:.2f} → {W}×{H}, he-raw {umpx0*ds:.4f} µm/px", flush=True)
def write_fold(N, shared, outdir, split="half", ngene=200, block_um=320.0, margin_um=64.0):
    A = ad.read_h5ad(f"{PREP}/{N}_bin16.h5ad"); ppu = float(A.uns["px_per_um"]); pxl = np.asarray(A.obsm["pxl"], np.float64); xy = pxl / ppu
    Y_all = np.log1p(np.asarray(sparse.csr_matrix(A.X).todense(), np.float32)); genes = np.asarray(gene_names(A), dtype=str)
    gidx = np.argsort(-Y_all.var(0))[:ngene]; C = Y_all[:, gidx]; g = genes[gidx]
    ds = float(open(os.path.join(shared, "level-downsample.txt")).read()); pxl_raw = pxl / ds
    trm, tem = make_split(xy, split, block_um, margin_um)
    W, H = [int(x) for x in open(os.path.join(shared, "heraw-dims.txt")).read().split()]; mg = 48
    inbox = (pxl_raw[:, 0] >= mg) & (pxl_raw[:, 0] < W - mg) & (pxl_raw[:, 1] >= mg) & (pxl_raw[:, 1] < H - mg); ndrop = int((trm & ~inbox).sum()); trm = trm & inbox
    os.makedirs(outdir, exist_ok=True); ids = [f"s{i}" for i in np.where(trm)[0]]
    pd.DataFrame(C[trm], index=ids, columns=g).to_csv(os.path.join(outdir, "cnts.tsv"), sep="\t")
    pd.DataFrame({"x": pxl_raw[trm, 0].round().astype(int), "y": pxl_raw[trm, 1].round().astype(int)}, index=ids).to_csv(os.path.join(outdir, "locs-raw.tsv"), sep="\t")
    open(os.path.join(outdir, "gene-names.txt"), "w").write("\n".join(g) + "\n")
    umpx_raw = float(open(os.path.join(shared, "pixel-size-raw.txt")).read()); open(os.path.join(outdir, "radius-raw.txt"), "w").write(f"{8.0/umpx_raw:.4f}\n")
    for fn in ("pixel-size-raw.txt", "pixel-size.txt", "level-downsample.txt", "heraw-dims.txt"): os.system(f"cp {shared}/{fn} {outdir}/")
    np.savez(os.path.join(outdir, "test.npz"), pxl_raw=pxl_raw[tem].astype(np.float32), xy_um=xy[tem].astype(np.float32), truth=C[tem].astype(np.float32), eval_genes=g, test_idx=np.where(tem)[0], train_idx=np.where(trm)[0])
    print(f"[{N}/{split}] train={trm.sum()}(丢越界 {ndrop}) test={tem.sum()} genes={len(g)} radius {8.0/umpx_raw:.1f}px", flush=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["image", "fold"]); ap.add_argument("--name", required=True); ap.add_argument("--shared", default=None); ap.add_argument("--out", required=True); ap.add_argument("--split", default="half"); a = ap.parse_args()
    write_image(a.name, a.out) if a.mode == "image" else write_fold(a.name, a.shared, a.out, split=a.split)

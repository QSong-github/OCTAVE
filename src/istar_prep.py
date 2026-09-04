# -*- coding: utf-8 -*-
"""
把我们的数据转成【官方 iStar】输入格式,并为片内 train/test 划分做准备。

iStar 是单切片超分方法: 给它某切片的 spot(counts+locs)+ 整张 H&E, 它训练并超分整片。
片内评测: 只喂【训练 fold】的 spot, 训练+超分后, 在【测试 fold】的 bin 位置采样 → per-gene PCC。

两种模式:
  image : 每切片一次 —— 写 he-raw.jpg + pixel-size-raw.txt + pixel-size.txt(图像与划分无关, 特征只提一次)
  fold  : 每 (切片×划分) 一次 —— 写 cnts.tsv/locs-raw.tsv(仅训练spot)+radius+gene-names,
          并存 test.npz(测试 bin 的 he-raw 像素坐标 + 真值 counts + 评测基因), 供 istar_eval 用

坐标: locs 用 he-raw.jpg 像素空间(= 全分辨率 tiff 的某金字塔层)。

【systema4ST 的两处修改, 相对母项目版本】
1. train/test 划分的坐标改用 obsm['pxl']/px_per_um, 不再用 obsm['spatial']*2。
   母项目版本用 spatial, 而 spatial 被 prep_bin.py 的 NaN→0 质心 bug 污染(位移中位
   P2 149µm / P5 286µm)。棋盘块 320µm、隔离带 64µm 都小于或可比于该位移 ⇒ 空间分块
   实质失效、隔离带完全失效 ⇒ 训练 spot 与测试 bin 相邻 ⇒ iStar 分数被泄漏抬高。
   注意 iStar 拿到的 spot 位置(pxl_raw)本来就是干净的, 坏的只是 train/test 归属。
2. 基因名改用 var['gene'] 的真实符号; 母项目版本取 var_names, 那是 '0'..'199' 占位索引。
"""
import os, sys, argparse, numpy as np, anndata as ad, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E
from within_bench import make_split
from st_encoder_bench import H5AD, SLIDES

STBENCH = "st_bench"
PX_PER_UM = {"Visium_HD_Human_Colon_Cancer_P2": 3.6499,
             "Visium_HD_Human_Colon_Cancer_P5": 3.6526}      # scalecheck.py 标定


def native_umperpx(slide_dir):
    """从源 adata 的 array_col/row(2µm bin 索引)与 pxl_*_in_fullres 线性拟合出 µm/px。"""
    import h5py, glob
    with h5py.File(os.path.join(slide_dir, "adata.h5ad"), "r") as h:
        ac = h["obs"]["array_col"][:].astype(np.float64)
        ar = h["obs"]["array_row"][:].astype(np.float64)
        pc = h["obs"]["pxl_col_in_fullres"][:].astype(np.float64)
        pr = h["obs"]["pxl_row_in_fullres"][:].astype(np.float64)
    sx = np.polyfit(ac, pc, 1)[0]      # px per 1 array-col step (=2µm)
    sy = np.polyfit(ar, pr, 1)[0]
    px_per_um = (abs(sx) + abs(sy)) / 2 / 2.0    # /2 因 array step = 2µm
    return 1.0 / px_per_um


def write_image(slide, outdir, level=2, target_px=1.0):
    import glob, openslide
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    sd = f"data/virtualST/{slide}"
    tiff = glob.glob(os.path.join(sd, "*PYRAMIDAL*.tif*"))[0]
    slide_o = openslide.OpenSlide(tiff)
    umpx0 = native_umperpx(sd)
    ds = slide_o.level_downsamples[level]
    W, Hh = slide_o.level_dimensions[level]
    print(f"[{slide}] 全分辨率 {umpx0:.4f} µm/px; 用金字塔层{level}(下采样{ds:.0f}) → {W}×{Hh}, "
          f"he-raw {umpx0*ds:.4f} µm/px", flush=True)
    img = slide_o.read_region((0, 0), level, (W, Hh)).convert("RGB")
    os.makedirs(outdir, exist_ok=True)
    img.save(os.path.join(outdir, "he-raw.jpg"), quality=92)
    with open(os.path.join(outdir, "pixel-size-raw.txt"), "w") as f:
        f.write(f"{umpx0*ds:.6f}\n")
    with open(os.path.join(outdir, "pixel-size.txt"), "w") as f:
        f.write(f"{target_px}\n")
    # 记录该层的下采样 + he-raw 尺寸, 供 fold 换算 locs / 过滤越界 spot
    with open(os.path.join(outdir, "level-downsample.txt"), "w") as f:
        f.write(f"{ds}\n")
    with open(os.path.join(outdir, "heraw-dims.txt"), "w") as f:
        f.write(f"{W} {Hh}\n")           # W H (he-raw 像素)
    print(f"[{slide}] 已写 {outdir}/he-raw.jpg + pixel-size (W={W} H={Hh})", flush=True)


def write_fold(slide, split, shared, outdir, hvg=200, block_um=320.0, margin_um=64.0):
    a = ad.read_h5ad(H5AD)
    sl = a.obs["slide_id"].astype(str).values
    m = np.where(sl == slide)[0]
    pxl = np.asarray(a.obsm["pxl"], np.float64)[m]          # 全分辨率像素中心(干净)
    coords_um = pxl / PX_PER_UM[slide]                      # 划分用的物理坐标, 见文件头说明
    ds = float(open(os.path.join(shared, "level-downsample.txt")).read())
    pxl_raw = pxl / ds                                       # → he-raw.jpg 像素空间

    # 直接用我们管线的 log-expr(top-200 HVG)当"counts"喂 iStar(内部只 min-max, 不假设整数)
    # → iStar 输出也在 log-expr 空间, 与我们的 per_gene_pcc 真值同空间, 可直接比。
    C = np.nan_to_num(np.asarray(a.X, np.float32))[m]
    genes = np.asarray(a.var["gene"] if "gene" in a.var else a.var_names, dtype=str)

    trm, tem = make_split(coords_um, split, block_um, margin_um)
    # 过滤越界【训练】spot: pxl_raw 须在 he-raw 图像框内留 margin(否则 iStar 网格坐标≤0/越界会崩)
    W, Hh = [int(x) for x in open(os.path.join(shared, "heraw-dims.txt")).read().split()]
    mg = 48
    inbox = (pxl_raw[:, 0] >= mg) & (pxl_raw[:, 0] < W - mg) & (pxl_raw[:, 1] >= mg) & (pxl_raw[:, 1] < Hh - mg)
    ndrop = int((trm & ~inbox).sum())
    trm = trm & inbox
    gidx = E.topk_hvg(C[trm], hvg)                           # 评测基因(与我们协议同法)
    eval_genes = genes[gidx]
    print(f"[{slide}/{split}] train={trm.sum()}(丢越界{ndrop}) test={tem.sum()}, 评测{len(eval_genes)}基因", flush=True)

    os.makedirs(outdir, exist_ok=True)
    ids = [f"s{i}" for i in np.where(trm)[0]]
    pd.DataFrame(C[trm], index=ids, columns=genes).to_csv(os.path.join(outdir, "cnts.tsv"), sep="\t")
    pd.DataFrame({"x": pxl_raw[trm, 0].round().astype(int),
                  "y": pxl_raw[trm, 1].round().astype(int)}, index=ids).to_csv(
        os.path.join(outdir, "locs-raw.tsv"), sep="\t")
    with open(os.path.join(outdir, "gene-names.txt"), "w") as f:
        f.write("\n".join(eval_genes) + "\n")
    umpx_raw = float(open(os.path.join(shared, "pixel-size-raw.txt")).read())
    with open(os.path.join(outdir, "radius-raw.txt"), "w") as f:
        f.write(f"{8.0/umpx_raw:.4f}\n")
    for fn in ("pixel-size-raw.txt", "pixel-size.txt", "level-downsample.txt"):
        os.system(f"cp {shared}/{fn} {outdir}/")
    np.savez(os.path.join(outdir, "test.npz"),
             pxl_raw=pxl_raw[tem].astype(np.float32),
             truth=C[tem][:, gidx].astype(np.float32),
             eval_genes=eval_genes)
    print(f"[{slide}/{split}] 已写 {outdir}: cnts {trm.sum()}×{len(genes)}, radius {8.0/umpx_raw:.1f}px", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["image", "fold"])
    ap.add_argument("--slide", required=True)
    ap.add_argument("--split", default="checker")
    ap.add_argument("--shared", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--hvg", type=int, default=200)
    args = ap.parse_args()
    if args.mode == "image":
        write_image(args.slide, args.out, level=args.level)
    else:
        write_fold(args.slide, args.split, args.shared, args.out, hvg=args.hvg)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
评测官方 iStar 的超分输出: 在【测试 bin】的网格位置采样 cnts-super/{gene}.pickle, 算 per-gene PCC。
坐标对齐复刻 iStar 的 get_locs: he-raw → he.jpg(×scale) → 特征网格(//rescale_factor)。
"""
import os, sys, glob, pickle, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", required=True)
    ap.add_argument("--win", type=int, default=1, help="采样窗口半径(网格单元); 0=点采样")
    ap.add_argument("--topk", type=int, default=0, help=">0 时只评方差最高的 K 个基因(top-K HVG 子集)")
    args = ap.parse_args()
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    d = args.fold_dir.rstrip("/") + "/"
    t = np.load(os.path.join(d, "test.npz"), allow_pickle=True)
    pxl_raw = t["pxl_raw"].astype(np.float64)       # (Nte,2) x,y in he-raw
    truth = t["truth"].astype(np.float32)
    eval_genes = [str(g) for g in t["eval_genes"]]
    # eval_genes/truth 按训练方差升序(topk_hvg 用 argsort[-k:]) → 末尾 K 个 = top-K HVG
    if args.topk > 0 and args.topk < len(eval_genes):
        truth = truth[:, -args.topk:]
        eval_genes = eval_genes[-args.topk:]
        print(f"只评 top-{args.topk} HVG(方差最高子集)", flush=True)

    scale = float(open(d + "pixel-size-raw.txt").read()) / float(open(d + "pixel-size.txt").read())
    W_he, H_he = Image.open(d + "he.jpg").size                # PIL: (W,H)
    # 任取一个超分图拿网格尺寸
    sup_files = {os.path.basename(p)[:-7]: p for p in glob.glob(os.path.join(d, "cnts-super", "*.pickle"))}
    any_arr = load_pkl(next(iter(sup_files.values())))
    Hg, Wg = any_arr.shape[:2]
    rf_i = (H_he) // Hg; rf_j = (W_he) // Wg
    print(f"he.jpg {W_he}×{H_he}, 超分网格 {Wg}×{Hg}, rescale_factor=({rf_i},{rf_j}), scale={scale:.4f}", flush=True)

    # test 像素 → he.jpg → 网格 ij
    hej_x = pxl_raw[:, 0] * scale; hej_y = pxl_raw[:, 1] * scale
    gi = np.floor(hej_y / rf_i).astype(int)
    gj = np.floor(hej_x / rf_j).astype(int)
    inb = (gi >= 0) & (gi < Hg) & (gj >= 0) & (gj < Wg)
    print(f"测试 bin {len(gi)}, 落在网格内 {inb.sum()}", flush=True)

    def sample(arr):
        v = np.full(len(gi), np.nan, np.float32)
        w = args.win
        for idx in np.where(inb)[0]:
            i0, i1 = max(gi[idx]-w, 0), min(gi[idx]+w+1, Hg)
            j0, j1 = max(gj[idx]-w, 0), min(gj[idx]+w+1, Wg)
            patch = arr[i0:i1, j0:j1]
            v[idx] = np.nanmean(patch) if np.isfinite(patch).any() else np.nan
        return v

    pccs, miss = [], []
    for k, g in enumerate(eval_genes):
        if g not in sup_files:
            miss.append(g); continue
        pred = sample(np.asarray(load_pkl(sup_files[g]), np.float32))
        y = truth[:, k]
        ok = np.isfinite(pred) & np.isfinite(y)
        if ok.sum() < 10 or np.std(pred[ok]) < 1e-8 or np.std(y[ok]) < 1e-8:
            pccs.append(0.0); continue
        pccs.append(float(np.corrcoef(pred[ok], y[ok])[0, 1]))
    pccs = np.array(pccs)
    print(f"\n=== iStar(官方) 评测: {os.path.basename(d.rstrip('/'))} ===")
    print(f"评测基因 {len(eval_genes)}, 缺失超分输出 {len(miss)}", flush=True)
    print(f">>> per-gene PCC = {np.nanmean(pccs):.4f} (中位 {np.nanmedian(pccs):.4f}, n={len(pccs)})")
    import json
    suf = f"_top{args.topk}" if args.topk > 0 else ""
    json.dump({"fold": os.path.basename(d.rstrip("/")), "pcc_mean": float(np.nanmean(pccs)),
               "pcc_median": float(np.nanmedian(pccs)), "n_genes": len(pccs), "n_missing": len(miss)},
              open(d + f"eval_result{suf}.json", "w"), indent=2)
    print(f"已存 {d}eval_result{suf}.json")


if __name__ == "__main__":
    main()

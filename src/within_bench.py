# -*- coding: utf-8 -*-
"""
【片内】benchmark —— 与跨片协议并列的第二套评测(片内插补/超分场景)。

片内评测最大的坑是**空间自相关泄漏**: 相邻 16µm bin 的表达几乎相同,
随机切分会让"训练点"紧挨着"测试点",分数被极大虚高。因此并排跑三种切分:

  random  : 片内随机 50/50 —— 多数论文的做法, 作为【被泄漏污染】的参照
  checker : 空间棋盘格分块(默认 320µm 块)+ 隔离带 —— 推荐口径, 训练/测试在空间上分离
  half    : 按中位 x 切成左右两半 + 隔离带 —— 最难的片内口径(接近跨区域外推)

隔离带: 用 KD 树丢掉距任一测试点 < margin µm 的训练点, 从根上断掉边界泄漏。
方法集与跨片表一致: ST 编码器基线 / 各 ST 编码器 / BLEEP-style / Ridge(HEST协议) / 纯图像kNN。
每张切片独立评测后再平均。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E, retrieval as R
from align import build_aligner
from baselines import ridge_predict, bleep
from st_encoder_bench import load_model_emb, H5AD, EXT_ST, EMBDIR, SLIDES

UM = 2.0   # obsm['spatial'] 存的是 2µm bin 索引 → ×2 得 µm


def make_split(coords_um, mode, block_um, margin_um, seed=0):
    """返回 (train_mask, test_mask)。coords_um: (n,2) µm 坐标。"""
    n = coords_um.shape[0]
    rng = np.random.default_rng(seed)
    if mode == "random":
        te = np.zeros(n, bool); te[rng.choice(n, n // 2, replace=False)] = True
    elif mode == "checker":
        bx = np.floor(coords_um[:, 0] / block_um).astype(int)
        by = np.floor(coords_um[:, 1] / block_um).astype(int)
        te = ((bx + by) % 2 == 1)
    elif mode == "half":
        te = coords_um[:, 0] > np.median(coords_um[:, 0])
    else:
        raise ValueError(mode)
    tr = ~te
    if margin_um > 0 and te.any() and tr.any():
        from scipy.spatial import cKDTree
        d, _ = cKDTree(coords_um[te]).query(coords_um[tr], k=1)
        keep = d >= margin_um                      # 丢掉贴着测试点的训练点
        idx_tr = np.where(tr)[0]
        tr = np.zeros(n, bool); tr[idx_tr[keep]] = True
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", default="hibou_l")
    ap.add_argument("--mode", default="checker", choices=["random", "checker", "half"])
    ap.add_argument("--block_um", type=float, default=320.0)
    ap.add_argument("--margin_um", type=float, default=64.0)
    ap.add_argument("--k", type=int, default=800)
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--jepa", type=float, default=4.0)
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    a = ad.read_h5ad(H5AD)
    expr = np.nan_to_num(np.asarray(a.X, np.float32))
    slide = a.obs["slide_id"].astype(str).values
    coords = np.asarray(a.obsm["spatial"], np.float32) * UM
    n = a.n_obs
    nps = [int((slide == s).sum()) for s in SLIDES]
    img = np.nan_to_num(np.concatenate([
        np.load(os.path.join(EMBDIR, f"emb_{args.tower}_P2.npy")),
        np.load(os.path.join(EMBDIR, f"emb_{args.tower}_P5.npy"))]).astype(np.float32))
    ST = {"collab": np.nan_to_num(np.asarray(ad.read_h5ad(EXT_ST).obsm["img_emb"], np.float32)),
          "st_pca": np.asarray(a.obsm["st_pca"], np.float32)}
    for m in ("nicheformer", "scgpt_spatial", "scgpt", "novae"):
        e = load_model_emb(m, n, nps)
        if e is not None:
            ST[m] = e
    print(f"片内评测 | 塔={args.tower} 切分={args.mode} 块={args.block_um}µm "
          f"隔离带={args.margin_um}µm k={args.k}", flush=True)

    acc = {}
    def add(nm, v): acc.setdefault(nm, []).append(float(v))

    for s in SLIDES:
        sm = np.where(slide == s)[0]
        c = coords[sm]
        tr_m, te_m = make_split(c, args.mode, args.block_um, args.margin_um)
        tr, te = sm[tr_m], sm[te_m]
        gidx = E.topk_hvg(expr[tr], args.hvg)
        Ytr = expr[tr]
        print(f"[{s}] train={len(tr)} test={len(te)} (原 {len(sm)}, 隔离带丢弃 "
              f"{len(sm)-len(tr)-len(te)})", flush=True)

        floor = R.image_floor(img[te], img[tr], Ytr, k=args.k)
        add("kNN图像检索(下界)", E.per_gene_pcc(floor, expr[te], gidx).mean())
        add("Ridge(HEST协议,α=1e4)",
            E.per_gene_pcc(ridge_predict(img[tr], Ytr, img[te], 1e4), expr[te], gidx).mean())
        add("BLEEP-style(冻结塔)",
            E.per_gene_pcc(bleep(img[tr], Ytr, img[te], Ytr, k=args.k), expr[te], gidx).mean())
        for enc, st in ST.items():
            al = build_aligner("mlp", hidden=0, jepa_weight=args.jepa,
                               temp=args.temp, epochs=40).fit(img[tr], st[tr])
            p = R.retrieve_cross_modal(al.project_img(img[te]), al.project_st(st[tr]), Ytr, k=args.k)
            nm = "ST=external" if enc == "collab" else f"ST={enc}"
            add(nm, E.per_gene_pcc(p, expr[te], gidx).mean())

    rows = sorted(((k, float(np.mean(v)), float(np.std(v))) for k, v in acc.items()),
                  key=lambda r: -r[1])
    print(f"\n=== 片内 benchmark (塔={args.tower}, 切分={args.mode}, top-{args.hvg} HVG) ===")
    print(f"{'方法':26s} {'per-gene PCC':>13s} {'片间std':>9s}")
    for k, m, sd in rows:
        print(f"{k:26s} {m:>13.4f} {sd:>9.4f}")
    os.makedirs(args.out, exist_ok=True)
    blk = f"_b{int(args.block_um)}" if args.mode == "checker" else ""   # 防止不同块尺寸互相覆盖
    op = os.path.join(args.out, f"within_{args.mode}{blk}_{args.tower}_k{args.k}_hvg{args.hvg}.json")
    json.dump([{"method": k, "pcc": m, "std": sd} for k, m, sd in rows],
              open(op, "w"), indent=2, ensure_ascii=False)
    print(f"已存 {op}")


if __name__ == "__main__":
    main()

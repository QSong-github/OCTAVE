# -*- coding: utf-8 -*-
"""
ST 编码器对比主实验(本方向的核心图)。
同一管线、同一图像塔、同一划分,只换 ST 编码器 → 比 per-gene PCC。

对每个 ST 编码器 E 给两个数,把"编码器好不好"和"对齐好不好"分开:
  ceiling(E) = 用 E 自己当 query 检索(oracle) —— E 索引表达的内在能力
  clip(E)    = 图像 query → 对齐 → 检索 E 的参考 —— 实际可用性能(无泄漏)
  gap        = ceiling - clip —— 图像侧对齐的损失
另有 image-floor(完全不用 ST)作为下界。

用法:
  python st_encoder_bench.py --tower phikon        # 某图像塔 × 全部 ST 编码器
  python st_encoder_bench.py --ceiling             # 只算各 ST 编码器的 oracle 天花板
"""
import os, sys, glob, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import run
import evaluate as E, retrieval as R

H5AD = "data/binned_16um.h5ad"
EXT_ST = "data/binned_16um_dino1024.h5ad"      # obsm['img_emb'] 为外部 ST 编码器的嵌入
EMBDIR = "results"
STBENCH = "st_bench"
SLIDES = ["Visium_HD_Human_Colon_Cancer_P2", "Visium_HD_Human_Colon_Cancer_P5"]
MODELS = ["scgpt", "scgpt_spatial", "nicheformer", "novae"]


def load_model_emb(model, n_expect, n_per_slide):
    """拼 P2+P5(顺序须与 binned h5ad 一致)。返回 None 表示该模型还没跑完。"""
    parts = []
    for i, sl in enumerate(SLIDES):
        p = os.path.join(STBENCH, f"{model}_results", f"{sl}__16um_embedding.h5ad")
        if not os.path.exists(p):
            return None
        a = ad.read_h5ad(p)
        keys = [k for k in a.obsm if k != "spatial"]
        if not keys:
            return None
        e = np.asarray(a.obsm[keys[0]], np.float32)
        if e.shape[0] != n_per_slide[i]:
            print(f"  [warn] {model} {sl} 行数 {e.shape[0]} != {n_per_slide[i]}, 跳过")
            return None
        parts.append(e)
    emb = np.nan_to_num(np.concatenate(parts).astype(np.float32))
    assert emb.shape[0] == n_expect
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", default=None, help="图像塔 tag(results/emb_<tag>_P{2,5}.npy)")
    ap.add_argument("--ceiling", action="store_true", help="只算各 ST 编码器 oracle 天花板")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--aligner", default="cca")
    ap.add_argument("--hidden", type=int, default=0, help="mlp 隐层宽度; 0=线性头(推荐)")
    ap.add_argument("--jepa", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    akw = {} if args.aligner == "cca" else dict(hidden=args.hidden, jepa_weight=args.jepa,
                                                epochs=args.epochs, temp=args.temp)

    a = ad.read_h5ad(H5AD)
    expr = np.nan_to_num(np.asarray(a.X, np.float32))
    coords = np.asarray(a.obsm["spatial"], np.float32)
    slide = a.obs["slide_id"].astype(str).values
    genes = np.asarray(a.var_names)
    n = a.n_obs
    n_per_slide = [int((slide == s).sum()) for s in SLIDES]

    ST = {"ext_st": np.nan_to_num(np.asarray(ad.read_h5ad(EXT_ST).obsm["img_emb"], np.float32)),
          "st_pca": np.asarray(a.obsm["st_pca"], np.float32)}
    for m in MODELS:
        e = load_model_emb(m, n, n_per_slide)
        if e is None:
            print(f"  [skip] {m}: 输出还没就绪")
        else:
            ST[m] = e
    print(f"N={n} | ST 编码器: " + ", ".join(f"{k}({v.shape[1]}d)" for k, v in ST.items()), flush=True)

    def mk(img, st):
        return {"img": img, "st": st, "expr": expr, "coords": coords,
                "slide": slide, "patient": slide, "genes": genes}

    def floor_cv(rep, reduce_dim=64):
        """留一片: 直接用 rep 做检索(不经对齐)。rep=图像→地板; rep=ST→oracle 天花板。"""
        p = []
        for g in sorted(set(slide.tolist())):
            te = (slide == g); tr = ~te
            gidx = E.topk_hvg(expr[tr], args.hvg)
            pf = R.image_floor(rep[te], rep[tr], expr[tr], k=args.k, reduce_dim=reduce_dim)
            p.append(float(E.per_gene_pcc(pf, expr[te], gidx).mean()))
        return float(np.mean(p)), float(np.std(p))

    res = {}
    if args.ceiling:
        for k, v in ST.items():
            m, sd = floor_cv(v)
            res[k] = {"ceiling": m, "ceiling_std": sd}
            print(f"  ceiling[{k:14s}] = {m:.4f} ± {sd:.4f}", flush=True)
        tag = "ceiling"
    else:
        mats = []
        for t in args.tower.split(","):
            e = np.nan_to_num(np.concatenate([
                np.load(os.path.join(EMBDIR, f"emb_{t}_P2.npy")),
                np.load(os.path.join(EMBDIR, f"emb_{t}_P5.npy"))]).astype(np.float32))
            assert e.shape[0] == n, f"{t} 行数 {e.shape[0]} != {n}"
            mats.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8))  # 多塔拼接前归一化
        img = mats[0] * (1.0 if len(mats) == 1 else 1.0) if len(mats) == 1 else np.concatenate(mats, axis=1)
        if len(mats) == 1:
            img = np.nan_to_num(np.concatenate([
                np.load(os.path.join(EMBDIR, f"emb_{args.tower}_P2.npy")),
                np.load(os.path.join(EMBDIR, f"emb_{args.tower}_P5.npy"))]).astype(np.float32))
        print(f"图像塔 {args.tower} dim={img.shape[1]}", flush=True)
        s = run(mk(img, ST), args.aligner, "cross_slide", k=args.k, hvg=args.hvg, aligner_kw=akw)
        res["__image_floor__"] = s["__image_floor__"]
        for k in ST:
            res[k] = {"clip": s[k]["mean"], "clip_std": s[k]["std"]}
            print(f"  clip[{k:14s}] = {s[k]['mean']:.4f}  (floor {s['__image_floor__']['mean']:.4f})", flush=True)
        tag = args.tower.replace(",", "+")

    os.makedirs(args.out, exist_ok=True)
    suf = args.aligner if args.aligner == "cca" else f"mlp-h{args.hidden}-j{args.jepa}-t{args.temp}-e{args.epochs}"
    op = os.path.join(args.out, f"stbench_{tag}_{suf}_k{args.k}_hvg{args.hvg}.json")
    json.dump(res, open(op, "w"), indent=2, ensure_ascii=False)
    print(f"\n已存 {op}", flush=True)


if __name__ == "__main__":
    main()

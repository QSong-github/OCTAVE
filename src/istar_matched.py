#!/usr/bin/env python
"""骨干匹配的 iStar 对照：在 iStar 自己的 HIPT 特征上跑 Ridge / kNN。

内部记录 记录 iStar 表暂不可进论文，因两处不对称：
  ① 喂的是 log-expr 而非原始 counts；
  ② 其余方法用 Hibou-L 特征，而 iStar 用自带 HIPT。

①经查是**有正当理由的权衡**（iStar 的 predict_single_out 会用 y_range 反归一化，
输出落回输入单位；喂 log-expr 使输出与真值同空间，省去反演与聚合两步各自的误差）。
②才是真不对称：iStar 的成绩里混着「HIPT vs Hibou-L」这一项。

本脚本消除②：读 iStar 自己抽好的 embeddings-hist.pickle，在**同一网格、同一采样窗口、
同一批测试 bin、同一批基因**上跑 Ridge 与 kNN。三者之差便只剩方法本身。
"""
import argparse, glob, json, os, pickle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

Image.MAX_IMAGE_PIXELS = None


def load_pkl(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def grid_ij(pxl_raw, scale, rf_i, rf_j, Hg, Wg):
    """复刻 istar_eval.py 的坐标链：he-raw → he.jpg(×scale) → 特征网格(//rescale_factor)。"""
    gi = np.floor(pxl_raw[:, 1] * scale / rf_i).astype(int)
    gj = np.floor(pxl_raw[:, 0] * scale / rf_j).astype(int)
    inb = (gi >= 0) & (gi < Hg) & (gj >= 0) & (gj < Wg)
    return gi, gj, inb


def sample_stack(arr, gi, gj, inb, win):
    """arr: (C,Hg,Wg) → 每个点取 (2win+1)² 窗口均值，与 istar_eval 的采样一致。"""
    C = arr.shape[0]
    out = np.full((len(gi), C), np.nan, np.float32)
    Hg, Wg = arr.shape[1], arr.shape[2]
    for idx in np.where(inb)[0]:
        i0, i1 = max(gi[idx] - win, 0), min(gi[idx] + win + 1, Hg)
        j0, j1 = max(gj[idx] - win, 0), min(gj[idx] + win + 1, Wg)
        patch = arr[:, i0:i1, j0:j1].reshape(C, -1)
        with np.errstate(all="ignore"):
            out[idx] = np.nanmean(patch, 1)
    return out


def per_gene_pcc(pred, true):
    p = pred - pred.mean(0); t = true - true.mean(0)
    den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    return np.where(den > 1e-8, (p * t).sum(0) / den, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", required=True)
    ap.add_argument("--win", type=int, default=1)
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--knn", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.fold_dir.rstrip("/") + "/"
    fold = os.path.basename(d.rstrip("/"))
    args.out = args.out or f"results/istar_matched_{fold}.json"

    scale = float(open(d + "pixel-size-raw.txt").read()) / float(open(d + "pixel-size.txt").read())
    W_he, H_he = Image.open(d + "he.jpg").size
    sup = sorted(glob.glob(os.path.join(d, "cnts-super", "*.pickle")))
    Hg, Wg = np.asarray(load_pkl(sup[0])).shape[:2]
    rf_i, rf_j = H_he // Hg, W_he // Wg
    print(f"[{fold}] he.jpg {W_he}×{H_he}  网格 {Wg}×{Hg}  rf=({rf_i},{rf_j})  scale={scale:.4f}", flush=True)

    embs = load_pkl(d + "embeddings-hist.pickle")
    E = np.concatenate([embs["cls"], embs["sub"], embs["rgb"]]).astype(np.float32)
    print(f"  HIPT 特征 {E.shape}（cls+sub+rgb）", flush=True)

    # 训练 spot：cnts.tsv 的行 + locs-raw.tsv 的像素坐标
    cnts = pd.read_csv(d + "cnts.tsv", sep="\t", index_col=0)
    locs = pd.read_csv(d + "locs-raw.tsv", sep="\t", index_col=0)
    ids = [i for i in cnts.index if i in locs.index]
    cnts = cnts.loc[ids]; locs = locs.loc[ids]
    Ytr = cnts.to_numpy(np.float32)
    gi, gj, inb = grid_ij(locs[["x", "y"]].to_numpy(np.float64), scale, rf_i, rf_j, Hg, Wg)
    Xtr = sample_stack(E, gi, gj, inb, args.win)
    ok_tr = np.isfinite(Xtr).all(1)
    Xtr, Ytr = Xtr[ok_tr], Ytr[ok_tr]
    print(f"  训练 spot {len(ids)} → 网格内且特征有限 {ok_tr.sum()}", flush=True)

    t = np.load(d + "test.npz", allow_pickle=True)
    pxl_te = t["pxl_raw"].astype(np.float64)
    truth = t["truth"].astype(np.float32)
    eval_genes = [str(g) for g in t["eval_genes"]]
    gi2, gj2, inb2 = grid_ij(pxl_te, scale, rf_i, rf_j, Hg, Wg)
    Xte = sample_stack(E, gi2, gj2, inb2, args.win)
    ok_te = np.isfinite(Xte).all(1)
    print(f"  测试 bin {len(pxl_te)} → 可用 {ok_te.sum()}", flush=True)

    # 训练基因列与评测基因列对齐
    col = {g: i for i, g in enumerate(cnts.columns.astype(str))}
    keep = [(k, col[g]) for k, g in enumerate(eval_genes) if g in col]
    if not keep:
        raise SystemExit("训练与评测基因无交集")
    ke = np.array([k for k, _ in keep]); kc = np.array([c for _, c in keep])
    Ytr = Ytr[:, kc]; Yte = truth[ok_te][:, ke]
    print(f"  共同基因 {len(keep)}/{len(eval_genes)}", flush=True)

    pipe = Pipeline([("sc", StandardScaler()),
                     ("pca", PCA(n_components=min(args.pca, Xtr.shape[1], Xtr.shape[0] - 1),
                                 random_state=0))]).fit(Xtr)
    Ztr, Zte = pipe.transform(Xtr), pipe.transform(Xte[ok_te])

    res = {}
    reg = Ridge(solver="lsqr", alpha=100.0 / (Ztr.shape[1] * Ytr.shape[1]),
                random_state=0, fit_intercept=False, max_iter=1000).fit(Ztr, Ytr)
    res["Ridge_HIPT"] = float(np.nanmean(per_gene_pcc(reg.predict(Zte).astype(np.float32), Yte)))

    from scipy.spatial import cKDTree
    tree = cKDTree(Ztr)
    _, nb = tree.query(Zte, k=min(args.knn, len(Ztr)))
    res["kNN_HIPT"] = float(np.nanmean(per_gene_pcc(Ytr[nb].mean(1).astype(np.float32), Yte)))

    ev = os.path.join(d, "eval_result.json")
    istar = json.load(open(ev))["pcc_mean"] if os.path.exists(ev) else None

    print(f"\n=== 骨干匹配对照（全部用 iStar 自己的 HIPT 特征）===")
    print(f"  iStar(官方)   {istar if istar is None else round(istar, 4)}")
    for k, v in res.items():
        print(f"  {k:<12}  {v:.4f}")
    os.makedirs("results", exist_ok=True)
    json.dump({"fold": fold, "istar_official": istar, **res,
               "n_train": int(ok_tr.sum()), "n_test": int(ok_te.sum()),
               "n_genes": len(keep), "win": args.win},
              open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"已存 {args.out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
子集版标尺 —— 让 iStar 能被正确投影。

主标尺(ruler.py)是【跨片留一 + 测试片全部 bin + 用另一片选的 top-50 HVG】;
iStar 是【片内划分 + 测试 bin 子集 + 用本片训练 fold 选的基因】。
bin 集合与基因集合都不同, 直接把 iStar 的分数往主标尺上读会得出假结论。

本脚本在【与 iStar 完全相同的 bin 与基因】上重建整个合成预测器族:
  - 划分: 复刻 istar_prep.write_fold —— make_split(pxl/px_per_um, split, 320µm, 64µm)
  - 基因: topk_hvg(X[train], 200) 取末 50 个(方差最高), 与 istar_eval --topk 50 一致
  - 低通 S_T(y) 在【整片】上算再restrict到测试 bin —— A_coarse 按定义就是
    "知道真实表达场被模糊到 σ" 的 oracle, 它本就可以看全片; 这是参照系, 不是竞争者。

输出每个 fold 的 A_coarse 阶梯, 供把 iStar 的 PCC 插值读成"等价 σ"。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import evaluate as E, retrieval as R
from baselines import ridge_predict
from within_bench import make_split
from effres import build_operator, calibrate_sigma, PX_PER_UM, SLIDES, SEB
from ruler import group_means, corr_cols

BLOCK_UM, MARGIN_UM, HVG_POOL, HVG_EVAL = 320.0, 64.0, 200, 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", default="hibou_l")
    ap.add_argument("--tmax", type=int, default=2048)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    cps = [1]
    while cps[-1] < args.tmax:
        cps.append(cps[-1] * 2)

    a = ad.read_h5ad(SEB.H5AD)
    X = np.nan_to_num(np.asarray(a.X, np.float32))
    slide = a.obs["slide_id"].astype(str).values
    pxl = np.asarray(a.obsm["pxl"], np.float64)
    img = np.nan_to_num(np.concatenate([
        np.load(os.path.join(SEB.EMBDIR, f"emb_{args.tower}_P2.npy")),
        np.load(os.path.join(SEB.EMBDIR, f"emb_{args.tower}_P5.npy"))]).astype(np.float32))

    out = {}
    for s in SLIDES:
        tag = s.rsplit("_", 1)[-1]
        m = np.where(slide == s)[0]
        xy = pxl[m] / PX_PER_UM[s]
        W = build_operator(xy, k=8, cut_um=29.0)          # 整片建图
        sig = calibrate_sigma(W, xy, cps)
        Xs = X[m]

        for split in ("checker", "half"):
            trm, tem = make_split(xy, split, BLOCK_UM, MARGIN_UM)
            # 基因: 与 istar_prep + istar_eval --topk 50 完全一致
            gpool = E.topk_hvg(Xs[trm], HVG_POOL)          # 方差升序
            gidx = gpool[-HVG_EVAL:]
            y_full = Xs[:, gidx]
            print(f"\n[{tag}/{split}] train={int(trm.sum())} test={int(tem.sum())} "
                  f"基因={len(gidx)}", flush=True)

            S, cur, t = {}, y_full.copy(), 0
            for cp in cps:
                while t < cp:
                    cur = W @ cur; t += 1
                S[cp] = cur.copy()

            yte = y_full[tem]
            rows = {}
            for cp in cps:
                rows[f"A_coarse_t{cp}"] = float(E.per_gene_pcc(S[cp][tem], yte).mean())
            # 真实锚点: 同一划分下训练的 Ridge / imageKNN(HEST 协议与纯形态学下界)
            Ytr = Xs[trm]
            gi = np.arange(len(gidx))
            imk = R.image_floor(img[m][tem], img[m][trm], Ytr[:, gidx], k=800)
            rdg = ridge_predict(img[m][trm], Ytr[:, gidx], img[m][tem], 1e4)
            rows["R_imageKNN"] = float(E.per_gene_pcc(imk, yte).mean())
            rows["R_ridgeHEST"] = float(E.per_gene_pcc(rdg, yte).mean())
            # 域均值 oracle
            pc = PCA(n_components=50, random_state=0).fit_transform(img[m])
            for k in (20, 200):
                lab = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(pc)
                rows[f"D_domImg_k{k}"] = float(E.per_gene_pcc(
                    group_means(y_full, lab)[tem], yte).mean())

            out[f"{tag}_{split}"] = {"sigma_um": {str(c): sig[c] for c in cps},
                                     "n_test": int(tem.sum()), "scores": rows}
            print(f"  A_coarse 阶梯: " + " ".join(
                f"{sig[c]:.0f}µm={rows[f'A_coarse_t{c}']:.4f}" for c in cps), flush=True)
            print(f"  锚点: imageKNN={rows['R_imageKNN']:.4f} Ridge={rows['R_ridgeHEST']:.4f} "
                  f"domImg20={rows['D_domImg_k20']:.4f} domImg200={rows['D_domImg_k200']:.4f}",
                  flush=True)

    # ---- 把 iStar 的实测 PCC 插值读成等价 σ
    print(f"\n=== iStar 投影(同 bin 同基因) ===")
    print(f"{'fold':14s}{'iStar PCC':>11s}{'等价 σ':>10s}{'imageKNN→σ':>12s}{'Ridge→σ':>10s}")
    for fold, d in out.items():
        p = f"/path/to/systema4ST/istar_run/{fold}/eval_result_top50.json"
        if not os.path.exists(p):
            print(f"{fold:14s}{'(缺 eval)':>11s}"); continue
        istar = json.load(open(p))
        v = istar.get("pcc_mean", istar.get("pcc", None))
        sig = {int(k): val for k, val in d["sigma_um"].items()}
        xs = sorted(sig, key=lambda c: sig[c])
        curve = [(sig[c], d["scores"][f"A_coarse_t{c}"]) for c in xs]

        def to_sigma(val):
            if val is None: return None
            for i in range(len(curve) - 1):
                (s0, v0), (s1, v1) = curve[i], curve[i + 1]
                if (v0 >= val >= v1) or (v0 <= val <= v1):
                    if abs(v0 - v1) < 1e-12: return s0
                    return s0 + (v0 - val) / (v0 - v1) * (s1 - s0)
            return float("nan") if val < curve[-1][1] else 0.0

        d["istar_pcc"] = v
        d["istar_sigma_um"] = to_sigma(v)
        print(f"{fold:14s}{(v if v is not None else float('nan')):>11.4f}"
              f"{(d['istar_sigma_um'] or float('nan')):>10.0f}"
              f"{(to_sigma(d['scores']['R_imageKNN']) or float('nan')):>12.0f}"
              f"{(to_sigma(d['scores']['R_ridgeHEST']) or float('nan')):>10.0f}")

    os.makedirs(args.out, exist_ok=True)
    op = os.path.join(args.out, "ruler_fold.json")
    json.dump(out, open(op, "w"), indent=2, ensure_ascii=False)
    print(f"\n已存 {op}\n注: 等价 σ 为 nan 表示该方法的分数低于 σ={cps[-1]} 档的 oracle, 即"
          "\n    它比'只知道真实表达场被模糊到该尺度'还差 —— 需在正文明确说明。")


if __name__ == "__main__":
    main()

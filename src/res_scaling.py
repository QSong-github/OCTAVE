# -*- coding: utf-8 -*-
"""
分辨率标度实验 —— 检验"有效分辨率 ≈ 平台间距的一个数量级倍"这个猜想。

两个观测点提示了这条规律, 但编码器/基因集/协议都不同, 连不成线:
    Visium HD  bin 16µm  → Ridge 等价 σ ≈ 120µm   (7.5×)
    HEST      spot 100µm → phikon 等价 σ ≈ 1100µm (11×)
本脚本做严格对照: 同一编码器(hibou_l)、同一协议(跨片留一)、同一基因选法,
只变 bin 尺寸(2/4/8/16/32µm), 看等价 σ 是否跟着按比例走。

若等价 σ ∝ bin 尺寸 ⇒ 换更精细的平台买不到细尺度预测能力, 直接冲击"超分辨率"叙事。
若等价 σ 趋于常数   ⇒ 存在一个绝对的形态学-表达耦合尺度, 结论完全不同。
两种结果都是可发表的。

⚠️ 前置陷阱: res_bench.py 说各分辨率的 bin 数相同(各片约 13 万), 这意味着 2µm 那档
   可能是从 870 万个 2µm bin 里【抽样】出来的, 实际点间距未必是 2µm。所以本脚本
   先实测每档的最近邻间距, 图的 cut 半径按实测值自适应, 并把实测间距一并报告 ——
   如果各档实测间距都差不多, 这个实验本身就不成立, 必须立刻发现而不是算出假结论。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.spatial import cKDTree
import evaluate as E, retrieval as R
from baselines import ridge_predict
from effres import build_operator, calibrate_sigma, PX_PER_UM, SLIDES

PARENT = "/path/to/upstream_align"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ums", default="2,4,8,16,32")
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--k", type=int, default=800)
    ap.add_argument("--tmax", type=int, default=1024)
    ap.add_argument("--out", default="results/res_scaling.json")
    a_ = ap.parse_args()
    cps = [1]
    while cps[-1] < a_.tmax:
        cps.append(cps[-1] * 2)

    out = {}
    for um in [int(x) for x in a_.ums.split(",")]:
        f = os.path.join(PARENT, f"data/res{um}um.h5ad")
        eb = [os.path.join(PARENT, f"results/emb_res{um}um_hibou_l_P{p}.npy") for p in (2, 5)]
        if not os.path.exists(f) or not all(os.path.exists(e) for e in eb):
            print(f"[skip] {um}µm 数据或嵌入缺失", flush=True); continue
        a = ad.read_h5ad(f)
        X = np.nan_to_num(np.asarray(a.X, np.float32))
        slide = a.obs["slide_id"].astype(str).values
        pxl = np.asarray(a.obsm["pxl"], np.float64)
        img = np.nan_to_num(np.concatenate([np.load(e) for e in eb]).astype(np.float32))
        assert img.shape[0] == a.n_obs, f"{um}µm: 图像 {img.shape[0]} vs 数据 {a.n_obs}"
        print(f"\n########## {um}µm | {a.n_obs} bin | 基因 {a.n_vars} ##########", flush=True)

        acc = {}
        for s in SLIDES:
            te = slide == s; tr = ~te
            gidx = E.topk_hvg(X[tr], a_.hvg)
            Ytr, y = X[tr], X[te][:, gidx]
            xy = pxl[te] / PX_PER_UM[s]
            d2, _ = cKDTree(xy).query(xy, k=2)
            nn = float(np.median(d2[:, 1]))
            cut = 1.8 * nn
            print(f"  [{s[-2:]}] n={int(te.sum())} 实测最近邻={nn:.2f}µm "
                  f"(标称 {um}µm, 比值 {nn/um:.2f}) → cut={cut:.1f}µm", flush=True)
            W = build_operator(xy, k=8, cut_um=cut)
            sig = calibrate_sigma(W, xy, cps)
            ladder, cur, t = {}, y.copy(), 0
            for cp in cps:
                while t < cp:
                    cur = W @ cur; t += 1
                ladder[cp] = float(E.per_gene_pcc(cur, y).mean())
            preds = {"imageKNN": R.image_floor(img[te], img[tr], Ytr, k=a_.k)[:, gidx],
                     "Ridge_HEST": ridge_predict(img[tr], Ytr, img[te], 1e4)[:, gidx]}

            def eqs(v):
                pts = sorted(((sig[c], ladder[c]) for c in cps), key=lambda t: t[0])
                if v >= pts[0][1]:
                    return pts[0][0]
                for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
                    if v0 >= v >= v1:
                        return s0 + (v0 - v) / max(v0 - v1, 1e-12) * (s1 - s0)
                return np.nan

            rec = dict(nn_um=nn, sigma_um={str(c): sig[c] for c in cps},
                       ladder={str(c): ladder[c] for c in cps})
            for nm, p in preds.items():
                v = float(E.per_gene_pcc(p, y).mean())
                rec[nm] = dict(pcc=v, eq_sigma=eqs(v))
                print(f"    {nm:12s} pcc={v:.4f} 等价σ={rec[nm]['eq_sigma']:.0f}µm "
                      f"(= {rec[nm]['eq_sigma']/nn:.1f}× 实测间距)", flush=True)
            acc[s] = rec
        out[str(um)] = acc

    os.makedirs("results", exist_ok=True)
    json.dump(out, open(a_.out, "w"), indent=2, ensure_ascii=False)

    print(f"\n=== 分辨率标度总表 (hibou_l, 跨片留一, top-{a_.hvg} HVG) ===")
    print(f"{'标称bin':>8s}{'实测间距':>10s}" + "".join(
        f"{nm+'.pcc':>13s}{nm+'.σ':>11s}{nm+'.×':>8s}" for nm in ("Ridge_HEST", "imageKNN")))
    for um, d in sorted(out.items(), key=lambda kv: int(kv[0])):
        nn = np.mean([d[s]["nn_um"] for s in d])
        row = f"{um+'µm':>8s}{nn:>10.2f}"
        for nm in ("Ridge_HEST", "imageKNN"):
            p = np.mean([d[s][nm]["pcc"] for s in d])
            q = np.nanmean([d[s][nm]["eq_sigma"] for s in d])
            row += f"{p:>13.4f}{q:>11.0f}{q/nn:>8.1f}"
        print(row)
    print("\n判读: 看最后一列(等价σ / 实测间距)。若各档都在 7–11× 附近 ⇒ 标度律成立;"
          "\n      若等价σ 绝对值不随 bin 变化 ⇒ 存在绝对耦合尺度, 结论相反。"
          "\n      若各档【实测间距】本身都差不多 ⇒ 本实验无效(数据是等数量抽样而非等分辨率)。")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
图像塔扫描 —— 检验"等价分辨率 ≈110µm"是否依赖编码器。

上游项目已为 P2/P5 提取了 49 份嵌入, 其中包含 HF gated 的最强几个
(uni_v1/uni_v2/virchow/virchow2/gigapath/hoptimus0/conch/gpfm/keep/openmidnight)。
用嵌入本身不需要权重, 所以这条线上可以直接覆盖到领域最强档 —— 这正是之前
标记为"需 HF 授权"的那一项。

对每个塔跑同一套: 跨片留一 → A_coarse 阶梯 → Ridge / imageKNN 的等价 σ(四个指标)。
若 25 个塔给出的等价 σ 都在同一量级 ⇒ 结论与编码器无关, 杀伤力远大于单塔结果;
若强塔明显更细 ⇒ 说明是编码器能力问题而非指标问题, 结论要改写。两种都必须如实报告。

SLURM 数组作业: 每个任务处理一个塔。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import evaluate as E, retrieval as R
from baselines import ridge_predict
from effres import build_operator, calibrate_sigma, PX_PER_UM, SLIDES, SEB
from vhd_downstream import zs, ret_at_k, dom_at1, morans_I

# 排除: 分辨率阶梯专用(res*um)、上下文/网格变体(ctx/grid)、测试残留 —— 只留独立基座模型
EXCLUDE_BASE = ("res2um", "res4um", "res8um", "res16um", "res32um", "test_")
EXCLUDE_VAR = ("_ctx", "_grid")


def towers(mode="base"):
    """mode=base: 25 个独立基座模型; mode=ctx: 同一编码器的上下文/网格变体。"""
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(SEB.EMBDIR, "emb_*_P2.npy"))):
        t = os.path.basename(p)[4:-7]
        if any(x in t for x in EXCLUDE_BASE):
            continue
        is_var = any(x in t for x in EXCLUDE_VAR)
        if (mode == "base") == is_var:
            continue
        if os.path.exists(os.path.join(SEB.EMBDIR, f"emb_{t}_P5.npy")):
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", default=None, help="不给则按 --idx 从清单里取")
    ap.add_argument("--idx", type=int, default=-1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--k", type=int, default=800)
    ap.add_argument("--tmax", type=int, default=2048)
    ap.add_argument("--k_dom", type=int, default=10)
    ap.add_argument("--mode", default="base", choices=["base", "ctx"])
    ap.add_argument("--outdir", default="results/tower_sweep")
    a_ = ap.parse_args()
    T = towers(a_.mode)
    if a_.list:
        print(f"{len(T)} 个塔:", " ".join(T)); return
    tw = a_.tower or T[a_.idx]
    print(f"塔 = {tw}  (清单共 {len(T)} 个)", flush=True)

    cps = [1]
    while cps[-1] < a_.tmax:
        cps.append(cps[-1] * 2)
    a = ad.read_h5ad(SEB.H5AD)
    expr = np.nan_to_num(np.asarray(a.X, np.float32))
    slide = a.obs["slide_id"].astype(str).values
    pxl = np.asarray(a.obsm["pxl"], np.float64)
    img = np.nan_to_num(np.concatenate([
        np.load(os.path.join(SEB.EMBDIR, f"emb_{tw}_P2.npy")),
        np.load(os.path.join(SEB.EMBDIR, f"emb_{tw}_P5.npy"))]).astype(np.float32))
    assert img.shape[0] == a.n_obs, f"{tw}: 嵌入 {img.shape[0]} vs 数据 {a.n_obs}"
    print(f"  嵌入维度 {img.shape[1]}", flush=True)

    out = {}
    for s in SLIDES:
        te = slide == s; tr = ~te
        gidx = E.topk_hvg(expr[tr], a_.hvg)
        Ytr, y = expr[tr], expr[te][:, gidx]
        xy = pxl[te] / PX_PER_UM[s]
        W = build_operator(xy, k=8, cut_um=29.0)
        A = (W > 0).astype(np.float32); A.setdiag(0); A.eliminate_zeros()
        sig = calibrate_sigma(W, xy, cps)
        lab = KMeans(n_clusters=a_.k_dom, n_init=4, random_state=0).fit_predict(zs(y))
        I_true = morans_I(A, y)

        def metrics(P):
            r, n = ret_at_k(P, y)
            lp = KMeans(n_clusters=a_.k_dom, n_init=4, random_state=0).fit_predict(zs(P))
            return dict(pcc=float(E.per_gene_pcc(P, y).mean()), **r, dom1=dom_at1(P, y, lab),
                        ari=float(adjusted_rand_score(lab, lp)), moran=morans_I(A, P), n_ret=n)

        M = {"imageKNN": metrics(R.image_floor(img[te], img[tr], Ytr, k=a_.k)[:, gidx]),
             "Ridge_HEST": metrics(ridge_predict(img[tr], Ytr, img[te], 1e4)[:, gidx])}
        ladder, cur, t = {}, y.copy(), 0
        for cp in cps:
            while t < cp:
                cur = W @ cur; t += 1
            ladder[str(cp)] = metrics(cur)
        out[s] = dict(sigma_um={str(c): sig[c] for c in cps}, moran_true=I_true,
                      methods=M, ladder=ladder)
        print(f"  [{s[-2:]}] Ridge pcc={M['Ridge_HEST']['pcc']:.4f} "
              f"imageKNN pcc={M['imageKNN']['pcc']:.4f} Moran真={I_true:.3f}", flush=True)

    def eqs(d, mname, key):
        v = d["methods"][mname][key]
        pts = sorted(((d["sigma_um"][c], d["ladder"][c][key]) for c in d["ladder"]), key=lambda t: t[0])
        if v >= pts[0][1]:
            return pts[0][0]
        for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
            if v0 >= v >= v1:
                return s0 + (v0 - v) / max(v0 - v1, 1e-12) * (s1 - s0)
        return float("nan")

    summ = {nm: {k: float(np.nanmean([eqs(out[s], nm, k) for s in SLIDES]))
                 for k in ("pcc", "ret@1", "dom1", "ari")} for nm in ("Ridge_HEST", "imageKNN")}
    out["_summary"] = dict(tower=tw, dim=int(img.shape[1]), eq_sigma=summ,
                           pcc={nm: float(np.mean([out[s]["methods"][nm]["pcc"] for s in SLIDES]))
                                for nm in ("Ridge_HEST", "imageKNN")},
                           moran_ratio={nm: float(np.mean([out[s]["methods"][nm]["moran"] /
                                                           out[s]["moran_true"] for s in SLIDES]))
                                        for nm in ("Ridge_HEST", "imageKNN")})
    os.makedirs(a_.outdir, exist_ok=True)
    json.dump(out, open(os.path.join(a_.outdir, f"{tw}.json"), "w"), indent=2, ensure_ascii=False)
    print(f"\n=== {tw} ===")
    for nm in ("Ridge_HEST", "imageKNN"):
        print(f"  {nm:12s} pcc={out['_summary']['pcc'][nm]:.4f}  等价σ(µm) " +
              " ".join(f"{k}={summ[nm][k]:.0f}" for k in ("pcc", "ret@1", "dom1", "ari")) +
              f"  Moran比={out['_summary']['moran_ratio'][nm]:.2f}×")
    print(f"已存 {a_.outdir}/{tw}.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
标尺 (ruler) —— 本项目的骨架实验, 与任何方法实现无关。

思路: 不靠观察几个真实方法来归纳"指标在测什么", 而是**造一批真实技能已知的预测器**,
直接测出指标如何随三个成分变化:
    (1) 解剖结构保真度   (2) 区域内细结构保真度   (3) 噪声

预测器族(全部只用测试片自己的真值构造, 真实技能按构造已知):
  A coarse(T)      = S_T(y)                  完美粗结构、零细结构   —— 系统性变异的天花板
  B blend(β)       = S_T*(y) + β·(y - S_T*(y))  完美粗 + 比例 β 的细
  C noisy(ν)       = y + ν·σ_g·ε              完美一切 + 噪声
  D dom_img(k)     = 图像 KMeans 域内真值均值   H&E 原则上可达的"纯域"预测器
  E dom_expr(k)    = 表达 KMeans 域内真值均值   任何基于域的预测器的上界
  F global_mean                                  常数, PCC≡0 的合法性检查
真实锚点(按公开协议, 不塞进任何人的检索管线):
  imageKNN  纯形态学下界      Ridge  HEST-benchmark 协议

对每个预测器同时报告: 常规 per-gene PCC / 池化残差版簇内 PCC / 分解恒等式的四个分量,
并数值验证
    r_global = √(η·η̂)·r_B + √((1-η)(1-η̂))·r_W
这条恒等式成立与否, 决定论文的理论部分能不能写。

注意: 池化残差版与母项目"按簇分别算再加权平均"的实现不同 —— 只有池化版才与恒等式精确对应。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import evaluate as E, retrieval as R
from baselines import ridge_predict
from effres import build_operator, calibrate_sigma, PX_PER_UM, SLIDES, SEB


def corr_cols(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    na = np.sqrt((A ** 2).sum(0)); nb = np.sqrt((B ** 2).sum(0))
    ok = (na > 1e-9) & (nb > 1e-9)
    out = np.zeros(A.shape[1], np.float64)
    out[ok] = (A[:, ok] * B[:, ok]).sum(0) / (na[ok] * nb[ok])
    return out


def group_means(X, labels):
    """把每点替换成其所属簇的均值。返回同形状数组。"""
    out = np.empty_like(X)
    for c in np.unique(labels):
        m = labels == c
        out[m] = X[m].mean(0)
    return out


def decompose(pred, truth, labels):
    """分解恒等式的四个分量 + 两边数值。逐基因算后取均值。

    全程 float64: 首轮在 float32 下最大误差 ~1e-3, 需排除是精度而非理论缺口。
    常数预测器(Var=0)下 η̂ 无定义, 恒等式本就不适用, 单独标记而不是算出个假值。
    """
    pred = np.asarray(pred, np.float64); truth = np.asarray(truth, np.float64)
    if pred.var(0).max() < 1e-12:
        return dict(eta=float("nan"), eta_hat=float("nan"), r_between=float("nan"),
                    r_within=float("nan"), r_global=0.0, identity_rhs=float("nan"),
                    identity_max_abs_err=float("nan"), degenerate=1.0)
    mu_t, mu_p = group_means(truth, labels), group_means(pred, labels)
    ep_t, ep_p = truth - mu_t, pred - mu_p
    vt, vp = truth.var(0), pred.var(0)
    eta = np.divide(mu_t.var(0), vt, out=np.zeros_like(vt), where=vt > 1e-12)
    eta_h = np.divide(mu_p.var(0), vp, out=np.zeros_like(vp), where=vp > 1e-12)
    rB, rW = corr_cols(mu_p, mu_t), corr_cols(ep_p, ep_t)
    lhs = corr_cols(pred, truth)
    rhs = np.sqrt(np.clip(eta * eta_h, 0, 1)) * rB + \
          np.sqrt(np.clip((1 - eta) * (1 - eta_h), 0, 1)) * rW
    return dict(eta=float(eta.mean()), eta_hat=float(eta_h.mean()),
                r_between=float(rB.mean()), r_within=float(rW.mean()),
                r_global=float(lhs.mean()), identity_rhs=float(rhs.mean()),
                identity_max_abs_err=float(np.abs(lhs - rhs).max()), degenerate=0.0)


def pcc_within_pooled(pred, truth, labels, min_n=30):
    """池化残差版簇内 PCC —— 与恒等式的 r_W 一致。"""
    keep = np.zeros(len(labels), bool)
    P, T = pred.copy(), truth.copy()
    for c in np.unique(labels):
        m = labels == c
        if m.sum() < min_n:
            continue
        P[m] -= P[m].mean(0); T[m] -= T[m].mean(0); keep |= m
    return corr_cols(P[keep], T[keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", default="hibou_l")
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--k", type=int, default=800)
    ap.add_argument("--tmax", type=int, default=2048)
    ap.add_argument("--tstar", type=int, default=128, help="blend 族用的粗结构尺度(步数)")
    ap.add_argument("--nclust", type=int, default=20, help="报告簇内 PCC 用的簇数")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cps = [1]
    while cps[-1] < args.tmax:
        cps.append(cps[-1] * 2)

    a = ad.read_h5ad(SEB.H5AD)
    expr = np.nan_to_num(np.asarray(a.X, np.float32))
    slide = a.obs["slide_id"].astype(str).values
    pxl = np.asarray(a.obsm["pxl"], np.float64)
    img = np.nan_to_num(np.concatenate([
        np.load(os.path.join(SEB.EMBDIR, f"emb_{args.tower}_P2.npy")),
        np.load(os.path.join(SEB.EMBDIR, f"emb_{args.tower}_P5.npy"))]).astype(np.float32))

    acc = {}
    def add(name, key, v):
        acc.setdefault(name, {}).setdefault(key, []).append(v)

    for s in SLIDES:
        te = slide == s; tr = ~te
        gidx = E.topk_hvg(expr[tr], args.hvg)
        Ytr, y = expr[tr], expr[te][:, gidx]
        xy = pxl[te] / PX_PER_UM[s]
        n = y.shape[0]
        print(f"\n[fold {s}] test={n}", flush=True)

        W = build_operator(xy, k=8, cut_um=29.0)
        sig = calibrate_sigma(W, xy, cps)

        # 各尺度的低通 S_T(y)
        S, cur, t = {}, y.copy(), 0
        for cp in cps:
            while t < cp:
                cur = W @ cur; t += 1
            S[cp] = cur.copy()
        print(f"  低通已备齐 {len(S)} 个尺度", flush=True)

        # ---- 报告簇内 PCC 用的分层(图像特征, 训练片拟合, 无泄漏)
        pca = PCA(n_components=min(50, img.shape[1]), random_state=0).fit(img[tr])
        km = KMeans(n_clusters=args.nclust, n_init=4, random_state=0).fit(pca.transform(img[tr]))
        labels = km.predict(pca.transform(img[te]))

        # ---- 构造预测器
        rng = np.random.default_rng(0)
        P = {}
        # 按扩散步数 t 命名, 不按实测 σ —— 首轮按 σ 命名时两折的 σ 略有差异(如 154 vs 155),
        # 导致粗端四档各自只含单折数据。σ 另存, 跨折平均后再用于标注。
        for cp in cps:                                              # A
            P[f"A_coarse_t{cp}"] = S[cp]
            add("__sigma__", str(cp), float(sig[cp]))
        Tst = args.tstar
        for b in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0):        # B
            P[f"B_blend_b{b:g}"] = S[Tst] + b * (y - S[Tst])
        sg = y.std(0, keepdims=True)
        for v in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):                   # C
            P[f"C_noisy_v{v:g}"] = y + v * sg * rng.standard_normal(y.shape).astype(np.float32)
        pc_img = pca.transform(img[te])
        pc_exp = PCA(n_components=50, random_state=0).fit_transform(y)
        for k in (5, 10, 20, 50, 100, 200):                         # D, E
            li = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(pc_img)
            le = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(pc_exp)
            P[f"D_domImg_k{k}"] = group_means(y, li)
            P[f"E_domExpr_k{k}"] = group_means(y, le)
        P["F_globalmean"] = np.tile(y.mean(0), (n, 1))              # F
        P["R_imageKNN"] = R.image_floor(img[te], img[tr], Ytr, k=args.k)[:, gidx]
        P["R_ridgeHEST"] = ridge_predict(img[tr], Ytr, img[te], 1e4)[:, gidx]
        print(f"  预测器 {len(P)} 个", flush=True)

        for nm, p in P.items():
            p = np.asarray(p, np.float32)
            add(nm, "pcc", float(E.per_gene_pcc(p, y).mean()))
            add(nm, "pcc_within_pooled", float(pcc_within_pooled(p, y, labels).mean()))
            d = decompose(p, y, labels)
            for kk, vv in d.items():
                add(nm, kk, vv)

    sigma = {k: float(np.mean(v)) for k, v in acc.pop("__sigma__").items()}
    res = {nm: {k: float(np.mean(v)) for k, v in d.items()} for nm, d in acc.items()}
    lab = {f"A_coarse_t{cp}": f"A_coarse σ≈{sigma[str(cp)]:.0f}µm" for cp in cps}

    print(f"\n=== 标尺 (塔={args.tower}, top-{args.hvg} HVG, {args.nclust} 图像簇, 跨片留一) ===")
    print(f"{'预测器':26s}{'常规PCC':>9s}{'簇内PCC':>9s}{'η':>7s}{'η̂':>7s}"
          f"{'r_B':>7s}{'r_W':>7s}{'恒等式右边':>10s}{'最大误差':>10s}")
    for nm in sorted(res, key=lambda x: (x[0], -res[x]["pcc"])):
        r = res[nm]
        if r.get("degenerate", 0) > 0:
            print(f"{lab.get(nm, nm):26s}{r['pcc']:>9.4f}{r['pcc_within_pooled']:>9.4f}"
                  f"{'—':>7s}{'—':>7s}{'—':>7s}{'—':>7s}{'退化(Var=0)':>10s}{'n/a':>10s}")
            continue
        print(f"{lab.get(nm, nm):26s}{r['pcc']:>9.4f}{r['pcc_within_pooled']:>9.4f}{r['eta']:>7.3f}"
              f"{r['eta_hat']:>7.3f}{r['r_between']:>7.3f}{r['r_within']:>7.3f}"
              f"{r['identity_rhs']:>10.4f}{r['identity_max_abs_err']:>10.2e}")
    ok = [r["identity_max_abs_err"] for r in res.values() if r.get("degenerate", 0) == 0]
    print(f"\n恒等式 float64 复核: 非退化预测器 {len(ok)} 个, 最大逐基因误差 = {max(ok):.3e}")

    os.makedirs(args.out, exist_ok=True)
    op = os.path.join(args.out, f"ruler_{args.tower}_hvg{args.hvg}_k{args.nclust}.json")
    json.dump({"config": vars(args), "results": res}, open(op, "w"), indent=2, ensure_ascii=False)
    print(f"\n已存 {op}")
    print("\n判读要点:"
          "\n  1. A_coarse 各尺度的常规 PCC = 纯解剖结构预测器能拿到的分数 —— 真实方法若低于它,"
          "\n     说明连解剖都没吃透, README 的立意需要重写。"
          "\n  2. B_blend 给出 '报告 PCC ↔ 真实细结构保真度 β' 的换算表。"
          "\n  3. 恒等式最大误差应 ~1e-6; 若不成立, 论文的理论部分作废。")


if __name__ == "__main__":
    main()

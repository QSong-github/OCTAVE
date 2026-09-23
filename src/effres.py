# -*- coding: utf-8 -*-
"""
有效分辨率 (Effective Resolution) —— 本项目的核心新指标。

动机: H&E→ST 领域在 16µm 上评测, 并以"超分辨"为卖点, 但整体 per-gene PCC 有约一半
来自粗尺度组织结构。把预测与真值都做多尺度分解, 逐尺度算相关, 就能直接读出
"这个模型实际在哪个空间尺度上还有预测力" —— 单位是微米, 领域外的人也看得懂。

实现: 图上热核扩散 (不需要规则格点)
  - 在测试片的 obsm['pxl'](换算成 µm)上建 kNN 图, 只连 < CUT µm 的边(不跨组织空洞)
  - 惰性随机游走算子 W = (I + D^-1 A)/2, 低通 S_t = W^t
  - 带通 B_t = S_t - S_{2t};  最细带 B_0 = X - S_{t0}
  - 每条带上算 per-gene Pearson(带通后的 pred vs 带通后的 truth)
  - σ(t) 用【实测】方式标定: 对随机 delta 种子扩散, 量其空间标准差 ⇒ σ 轴是真微米

有效分辨率 ER(τ) = 仍满足 r(σ) ≥ τ 的最小 σ。

方法集与 evaluate_beyond.py 完全一致, 保证与常规 PCC 表可比。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.spatial import cKDTree
from scipy import sparse

import evaluate as E, retrieval as R
from align import build_aligner
from baselines import ridge_predict, bleep
import st_encoder_bench as SEB

PARENT = "/path/to/upstream_align"
SEB.H5AD = os.path.join(PARENT, "data/binned_16um.h5ad")
SEB.EXT_ST = os.path.join(PARENT, "data/binned_16um_dino1024.h5ad")
SEB.EMBDIR = os.path.join(PARENT, "results")
SEB.STBENCH = os.path.join(PARENT, "st_bench")
SLIDES = SEB.SLIDES
PX_PER_UM = {SLIDES[0]: 3.6499, SLIDES[1]: 3.6526}      # scalecheck.py 标定
PITCH_UM = 16.0


# ---------------------------------------------------------------- 图与扩散
def build_operator(xy_um, k=8, cut_um=29.0):
    """惰性随机游走算子 W = (I + D^-1 A)/2。cut 只保留 4 正交(16µm)+4 对角(22.6µm)邻居。"""
    n = xy_um.shape[0]
    d, idx = cKDTree(xy_um).query(xy_um, k=k + 1)
    d, idx = d[:, 1:], idx[:, 1:]                      # 去掉自身
    ok = d <= cut_um
    rows = np.repeat(np.arange(n), ok.sum(1))
    cols = idx[ok]
    A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)                                  # 对称化
    deg = np.asarray(A.sum(1)).ravel()
    iso = deg == 0
    deg[iso] = 1.0
    P = sparse.diags(1.0 / deg).tocsr() @ A
    W = (sparse.identity(n, format="csr", dtype=np.float32) + P) * 0.5
    print(f"  图: {n} 点, 平均度 {(deg[~iso]).mean():.2f}, 孤立点 {int(iso.sum())}", flush=True)
    return W.astype(np.float32)


def calibrate_sigma(W, xy_um, checkpoints, n_seed=192, seed=0):
    """实测 σ(t): 对随机 delta 种子扩散, 用二阶矩量其空间标准差(µm)。"""
    rng = np.random.default_rng(seed)
    s = rng.choice(xy_um.shape[0], size=min(n_seed, xy_um.shape[0]), replace=False)
    V = np.zeros((xy_um.shape[0], len(s)), np.float32)
    V[s, np.arange(len(s))] = 1.0
    out, t = {}, 0
    for cp in checkpoints:
        while t < cp:
            V = W @ V; t += 1
        w = np.maximum(V, 0).astype(np.float64)
        tot = w.sum(0) + 1e-12
        mx = (w * xy_um[:, [0]]).sum(0) / tot
        my = (w * xy_um[:, [1]]).sum(0) / tot
        m2 = ((w * (xy_um[:, [0]] ** 2)).sum(0) + (w * (xy_um[:, [1]] ** 2)).sum(0)) / tot
        var = m2 - mx ** 2 - my ** 2                    # 二维总方差
        out[cp] = float(np.sqrt(max(np.median(var), 0.0) / 2.0))   # 每轴 σ
        print(f"    σ(t={cp}) = {out[cp]:.1f}µm", flush=True)
    return out


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tower", default="hibou_l")
    ap.add_argument("--k", type=int, default=800)
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--jepa", type=float, default=4.0)
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--tmax", type=int, default=2048)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--cut_um", type=float, default=29.0)
    ap.add_argument("--tau", default="0.1,0.2,0.3")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cps = [1]
    while cps[-1] < args.tmax:
        cps.append(cps[-1] * 2)
    print(f"扩散检查点 t = {cps}", flush=True)

    a = ad.read_h5ad(SEB.H5AD)
    expr = np.nan_to_num(np.asarray(a.X, np.float32))
    genes = np.asarray(a.var["gene"]).astype(str)
    slide = a.obs["slide_id"].astype(str).values
    pxl = np.asarray(a.obsm["pxl"], np.float64)
    n = a.n_obs
    nps = [int((slide == s).sum()) for s in SLIDES]

    img = np.nan_to_num(np.concatenate([
        np.load(os.path.join(SEB.EMBDIR, f"emb_{args.tower}_P2.npy")),
        np.load(os.path.join(SEB.EMBDIR, f"emb_{args.tower}_P5.npy"))]).astype(np.float32))
    ST = {"ext_st": np.nan_to_num(np.asarray(ad.read_h5ad(SEB.EXT_ST).obsm["img_emb"], np.float32)),
          "st_pca": np.asarray(a.obsm["st_pca"], np.float32)}
    for m in ("nicheformer", "scgpt", "novae", "scgpt_spatial"):
        e = SEB.load_model_emb(m, n, nps)
        if e is not None:
            ST[m] = e
    print(f"塔={args.tower} N={n} ST={list(ST)}", flush=True)

    acc, sig_acc, pcc_acc = {}, {}, {}
    for s in SLIDES:
        te = (slide == s); tr = ~te
        gidx = E.topk_hvg(expr[tr], args.hvg)
        Ytr = expr[tr]
        xy = pxl[te] / PX_PER_UM[s]                      # → µm
        print(f"\n[fold {s}] test={int(te.sum())}", flush=True)

        W = build_operator(xy, k=args.knn, cut_um=args.cut_um)
        sig = calibrate_sigma(W, xy, cps)
        for cp, v in sig.items():
            sig_acc.setdefault(cp, []).append(v)

        floor = R.image_floor(img[te], img[tr], Ytr, k=args.k)
        preds = {"imageKNN": floor,
                 "Ridge_HEST": ridge_predict(img[tr], Ytr, img[te], 1e4),
                 "BLEEP": bleep(img[tr], Ytr, img[te], Ytr, k=args.k)}
        for enc, st in ST.items():
            al = build_aligner("mlp", hidden=0, jepa_weight=args.jepa,
                               temp=args.temp, epochs=40).fit(img[tr], st[tr])
            preds[enc] = R.retrieve_cross_modal(
                al.project_img(img[te]), al.project_st(st[tr]), Ytr, k=args.k)
        print(f"  预测完成: {list(preds)}", flush=True)

        # 方法级聚合 PCC —— 论文的核心检验需要它：
        # 「PCC 排序」与「σ 排序」是否一致？一致 ⇒ 指标只是压缩幅度；
        # 不一致 ⇒ 按 PCC 选方法会选错。此前该量未保存，无法回答这个问题。
        for _m in preds:
            _r = E.per_gene_pcc(preds[_m][:, gidx], expr[te][:, gidx])
            pcc_acc.setdefault(_m, []).append(float(np.nanmean(_r)))

        # 把 truth 与所有 pred 叠成一个矩阵一起扩散 —— 只扩散评测用的 HVG 列
        names = ["__truth__"] + list(preds)
        M = np.concatenate([expr[te][:, gidx]] + [preds[m][:, gidx] for m in preds], 1).astype(np.float32)
        G = len(gidx)
        prev = M.copy(); t = 0
        low_prev = M.copy()                              # S_0 = 原始
        for bi, cp in enumerate(cps):
            while t < cp:
                prev = W @ prev; t += 1
            band = low_prev - prev                       # B = S_{t_{i-1}} - S_{t_i}
            for j, nm in enumerate(names):
                if nm == "__truth__":
                    continue
                jt, jp = slice(0, G), slice(j * G, (j + 1) * G)
                r = E.per_gene_pcc(band[:, jp], band[:, jt])
                acc.setdefault(nm, {}).setdefault(cp, []).append(float(np.mean(r)))
            # 真值在该带上的功率占比
            pw = float((band[:, :G] ** 2).sum() / (M[:, :G] ** 2).sum())
            acc.setdefault("__truthpower__", {}).setdefault(cp, []).append(pw)
            low_prev = prev.copy()
            print(f"  带 t≤{cp} (σ≈{sig[cp]:.0f}µm) 完成, 真值功率占比 {pw*100:.1f}%", flush=True)

    sigma = {cp: float(np.mean(v)) for cp, v in sig_acc.items()}
    method_pcc = {k: float(np.mean(v)) for k, v in pcc_acc.items()}
    res = {nm: {str(cp): float(np.mean(v)) for cp, v in d.items()} for nm, d in acc.items()}
    taus = [float(x) for x in args.tau.split(",")]

    def eff_res(curve):
        """ER(τ): 仍满足 r ≥ τ 的最小 σ，在 log σ 上插值。

        检查点按 t 的 2 次幂走 ⇒ 相邻 σ 差 √2≈1.41×。直接取「首个达标的检查点」会把
        小于该步长的方法间差异全部量化掉 —— 实测 ours / st_pca / scgpt_spatial /
        nicheformer 四条曲线明显不同（0.1775/0.1722/0.1672/0.1718），ER 却被压成
        同一组值 (9.8/13.86/19.6)。同时保留离散值以便对账。
        """
        out, disc = {}, {}
        xs = sorted(sigma, key=lambda c: sigma[c])
        for tau in taus:
            val = None
            for i, c in enumerate(xs):
                if curve.get(str(c), -1) >= tau:
                    if i == 0:
                        val = float(sigma[c])          # 首档已达标 ⇒ 右删失于测量地板
                    else:
                        p0 = curve[str(xs[i - 1])]; p1 = curve[str(c)]
                        s0 = np.log(sigma[xs[i - 1]]); s1 = np.log(sigma[c])
                        w = 0.0 if p1 == p0 else (tau - p0) / (p1 - p0)
                        val = float(np.exp(s0 + w * (s1 - s0)))
                    break
            out[str(tau)] = val
            hit = [sigma[c] for c in xs if curve.get(str(c), -1) >= tau]
            disc[str(tau)] = float(min(hit)) if hit else None
        out["_discrete"] = disc
        return out
    def _eff_res_old(curve):
        out = {}
        xs = sorted(sigma, key=lambda c: sigma[c])
        for tau in taus:
            hit = [sigma[c] for c in xs if curve.get(str(c), -1) >= tau]
            out[str(tau)] = float(min(hit)) if hit else None
        return out

    print(f"\n=== 有效分辨率 (塔={args.tower}, top-{args.hvg} HVG, 跨片留一) ===")
    hdr = "".join(f"{('σ='+str(int(sigma[c]))):>9s}" for c in cps)
    print(f"{'方法':12s}{hdr}" + "".join(f"{('ER@'+str(t)):>9s}" for t in taus) + f"{'PCC':>9s}")
    print(f"{'真值功率%':12s}" + "".join(f"{res['__truthpower__'][str(c)]*100:>9.1f}" for c in cps))
    rows = []
    for nm in sorted([k for k in res if not k.startswith("__")],
                     key=lambda x: -np.mean(list(res[x].values()))):
        er = eff_res(res[nm])
        rows.append({"method": nm, "curve": res[nm], "eff_res_um": er,
                     "pcc": method_pcc.get(nm)})   # 聚合 PCC：用于「PCC 排序 vs σ 排序」检验
        _p = method_pcc.get(nm)
        print(f"{nm:12s}" + "".join(f"{res[nm][str(c)]:>9.3f}" for c in cps)
              + "".join(f"{(er[str(t)] if er[str(t)] else float('nan')):>9.1f}" for t in taus)
              + (f"{_p:>9.4f}" if isinstance(_p, float) else f"{'—':>9s}"))

    os.makedirs(args.out, exist_ok=True)
    op = os.path.join(args.out, f"effres_{args.tower}_hvg{args.hvg}_t{args.tmax}.json")
    json.dump({"sigma_um": {str(k): v for k, v in sigma.items()},
               "truth_power": res["__truthpower__"], "methods": rows,
               "config": vars(args)}, open(op, "w"), indent=2, ensure_ascii=False)
    print(f"\n已存 {op}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
噪声天花板 c(σ) —— 让"有效分辨率"变得可解读。

effres.py 给出各方法的分尺度相关 r(σ), 但 16µm bin 有 57.7% 零元, 散粒噪声本身就压着
最细带的可达上限。r(σ=10µm)=0.177 是可达的 20% 还是 90%? 不知道就不能下结论。

做法: 对原始 counts 做二项拆半(每个 count c 抽 h~Binom(c,0.5), 两半为 h 与 c-h),
两半各自按 prep_bin 的口径归一化(CP10k+log1p), 走同一套图扩散带通分解, 再互相求相关
⇒ c_half(σ)。因每半只有一半深度, 用 Spearman-Brown 校正到全深度:

    c(σ) = 2·c_half(σ) / (1 + c_half(σ))

然后 归一化技能 ρ(σ) = r(σ)/c(σ), 有效分辨率 ER = min{σ : ρ(σ) ≥ 0.5}。

顺带修正 effres.py 的一个错误: 真值功率占比必须【逐基因中心化】后再算, 否则 log1p 的
DC 分量(均值≈1.15)把分母撑爆, 所有频带都显得只占零点几个百分点。

只读源数据, 结果写 systema4ST/results。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import sparse
import evaluate as E
from effres import build_operator, calibrate_sigma, PX_PER_UM

PARENT = "/path/to/align_workspace"
BINNED = os.path.join(PARENT, "data/binned_16um.h5ad")
RAW = os.path.join(PARENT, "st_bench/data/{slide}/adata_16um.h5ad")
SLIDES = ["Visium_HD_Human_Colon_Cancer_P2", "Visium_HD_Human_Colon_Cancer_P5"]


def norm_log(counts_csr):
    """prep_bin 口径: 按全基因总 counts 归到 CP10k, 再 log1p。"""
    tot = np.asarray(counts_csr.sum(1)).ravel()
    tot[tot == 0] = 1.0
    Y = counts_csr.multiply((1e4 / tot)[:, None]).tocsr()
    Y.data = np.log1p(Y.data)
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hvg", type=int, default=50)
    ap.add_argument("--tmax", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--cut_um", type=float, default=29.0)
    ap.add_argument("--effres", default="results/effres_hibou_l_hvg50_t2048.json")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cps = [1]
    while cps[-1] < args.tmax:
        cps.append(cps[-1] * 2)

    b = ad.read_h5ad(BINNED)
    expr = np.nan_to_num(np.asarray(b.X, np.float32))
    bgene = np.asarray(b.var["gene"]).astype(str)
    bslide = b.obs["slide_id"].astype(str).values
    bpxl = np.asarray(b.obsm["pxl"], np.float64)

    ceil_acc, power_acc, sig_acc = {}, {}, {}
    for s in SLIDES:
        te = bslide == s
        tr = ~te
        gidx = E.topk_hvg(expr[tr], args.hvg)          # 与 effres.py 完全一致
        want = bgene[gidx]
        print(f"\n[fold {s}] test={int(te.sum())} 评测基因 {len(want)} 个", flush=True)

        r = ad.read_h5ad(RAW.format(slide=s))
        assert r.n_obs == int(te.sum()), f"行数不一致 {r.n_obs} vs {int(te.sum())}"

        # ---- 行序核对 + x_um 污染诊断
        # 首轮发现 st_bench 的 x_um/y_um 与 pxl 差一个中位 166/242µm 的非仿射位移,
        # 且尺度因子 <1 (P2 0.971, P5 0.916) —— 点云被向原点收缩, 正是 coord.npy 的
        # NaN→0 污染的指纹, 且与 NaN 比例(5.9%/7.7%)呈剂量-反应。故 x_um 不可用于建图。
        # 行序本身没问题: 若行序错乱, 仿射残差应是组织尺度(~2000µm)而非百微米量级。
        xy_raw = np.stack([r.obs["x_um"].to_numpy(np.float64),
                           r.obs["y_um"].to_numpy(np.float64)], 1)
        xy_pxl = bpxl[te] / PX_PER_UM[s]
        D = np.hstack([xy_pxl, np.ones((xy_pxl.shape[0], 1))])
        T, *_ = np.linalg.lstsq(D, xy_raw, rcond=None)
        res = np.linalg.norm(xy_raw - D @ T, axis=1)
        scale = np.sqrt(abs(np.linalg.det(T[:2])))
        # 判据: 与"行序被随机打乱"的零分布比。固定阈值(如 残差<500µm 占比>90%)不可用 ——
        # 污染幅度本身随 NaN 比例变化, P5 会被误杀。零分布才是正确的参照。
        perm = np.random.default_rng(0).permutation(len(res))
        Tn, *_ = np.linalg.lstsq(D, xy_raw[perm], rcond=None)
        res_null = np.linalg.norm(xy_raw[perm] - D @ Tn, axis=1)
        ratio = np.median(res) / np.median(res_null)
        print(f"  x_um 诊断: 仿射残差 中位={np.median(res):.1f}µm p99={np.percentile(res,99):.1f}µm "
              f"尺度因子={scale:.4f} | 打乱行序的零分布中位={np.median(res_null):.1f}µm "
              f"⇒ 比值={ratio:.3f} "
              f"{'✅ 行序一致(x_um 另有污染, 不用它)' if ratio < 0.5 else '⚠️ 行序可能错乱'}",
              flush=True)
        assert ratio < 0.5, f"行序核对未通过(比值 {ratio:.3f}), 中止"
        xy = xy_pxl                                     # 与 effres.py 用同一套坐标, 保证 r 与 c 可比

        gi = {g: i for i, g in enumerate(np.asarray(r.var_names).astype(str))}
        cols = np.array([gi[g] for g in want])
        C = sparse.csr_matrix(r.X)                      # 全基因(归一化要用全基因总数)
        del r
        print(f"  原始 counts: {C.shape} nnz={C.nnz}", flush=True)

        W = build_operator(xy, k=args.knn, cut_um=args.cut_um)
        sig = calibrate_sigma(W, xy, cps)
        for cp, v in sig.items():
            sig_acc.setdefault(cp, []).append(v)

        # ---- 二项拆半 × reps, 堆成一个矩阵一次扩散
        rng = np.random.default_rng(0)
        G = len(cols)
        blocks = []
        for rep in range(args.reps):
            h = rng.binomial(C.data.astype(np.int64), 0.5).astype(np.float32)
            A = sparse.csr_matrix((h, C.indices, C.indptr), shape=C.shape)
            B = sparse.csr_matrix((C.data - h, C.indices, C.indptr), shape=C.shape)
            blocks += [np.asarray(norm_log(A)[:, cols].todense(), np.float32),
                       np.asarray(norm_log(B)[:, cols].todense(), np.float32)]
            del h, A, B
            print(f"  rep{rep} 拆半完成", flush=True)
        truth = expr[te][:, gidx]
        M = np.concatenate(blocks + [truth], 1).astype(np.float32)
        nb = len(blocks)

        prev = M.copy(); low_prev = M.copy(); t = 0
        for cp in cps:
            while t < cp:
                prev = W @ prev; t += 1
            band = low_prev - prev
            rs = []
            for rep in range(args.reps):
                a_ = band[:, (2 * rep) * G:(2 * rep + 1) * G]
                b_ = band[:, (2 * rep + 1) * G:(2 * rep + 2) * G]
                rs.append(float(np.mean(E.per_gene_pcc(a_, b_))))
            ceil_acc.setdefault(cp, []).append(float(np.mean(rs)))
            # 修正版功率占比: 逐基因中心化后再算
            tb = band[:, nb * G:]
            tf = M[:, nb * G:]
            power_acc.setdefault(cp, []).append(
                float(((tb - tb.mean(0)) ** 2).sum() / ((tf - tf.mean(0)) ** 2).sum()))
            low_prev = prev.copy()
            print(f"  带 σ≈{sig[cp]:.0f}µm: c_half={ceil_acc[cp][-1]:.3f} "
                  f"功率(中心化)={power_acc[cp][-1]*100:.1f}%", flush=True)

    sigma = {cp: float(np.mean(v)) for cp, v in sig_acc.items()}
    c_half = {cp: float(np.mean(v)) for cp, v in ceil_acc.items()}
    c_full = {cp: 2 * v / (1 + v) for cp, v in c_half.items()}     # Spearman-Brown
    power = {cp: float(np.mean(v)) for cp, v in power_acc.items()}

    print(f"\n=== 噪声天花板 (top-{args.hvg} HVG, 跨片留一, {args.reps} 次拆半) ===")
    print(f"{'σ(µm)':>8s}" + "".join(f"{int(sigma[c]):>9d}" for c in cps))
    print(f"{'功率%':>8s}" + "".join(f"{power[c]*100:>9.1f}" for c in cps))
    print(f"{'c_half':>8s}" + "".join(f"{c_half[c]:>9.3f}" for c in cps))
    print(f"{'c(全深度)':>8s}" + "".join(f"{c_full[c]:>9.3f}" for c in cps))

    out = {"sigma_um": {str(k): v for k, v in sigma.items()},
           "c_half": {str(k): v for k, v in c_half.items()},
           "c_full": {str(k): v for k, v in c_full.items()},
           "truth_power_centered": {str(k): v for k, v in power.items()}}

    # ---- 若 effres 结果在, 直接算归一化技能与有效分辨率
    if os.path.exists(args.effres):
        er = json.load(open(args.effres))
        print(f"\n=== 归一化技能 ρ(σ)=r/c 与有效分辨率 (ρ≥0.5) ===")
        print(f"{'方法':14s}" + "".join(f"{('σ='+str(int(sigma[c]))):>9s}" for c in cps) + f"{'ER(µm)':>9s}")
        rows = []
        for m in er["methods"]:
            rho = {c: m["curve"][str(c)] / c_full[c] for c in cps if c_full[c] > 1e-6}
            hit = [sigma[c] for c in sorted(cps, key=lambda x: sigma[x]) if rho.get(c, 0) >= 0.5]
            ER = float(min(hit)) if hit else None
            rows.append({"method": m["method"], "rho": {str(c): rho[c] for c in rho}, "eff_res_um": ER})
            print(f"{m['method']:14s}" + "".join(f"{rho[c]:>9.3f}" for c in cps)
                  + (f"{ER:>9.0f}" if ER else f"{'>max':>9s}"))
        out["normalized"] = rows

    os.makedirs(args.out, exist_ok=True)
    op = os.path.join(args.out, f"ceiling_hvg{args.hvg}_t{args.tmax}.json")
    json.dump(out, open(op, "w"), indent=2, ensure_ascii=False)
    print(f"\n已存 {op}")


if __name__ == "__main__":
    main()

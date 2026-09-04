#!/usr/bin/env python
"""HEST 广度线的等价分辨率 σ —— 与深度线 effres.py 同一套带通扩散口径。

深度线（Visium HD 16µm）此前只在单塔 hibou_l 上算过 σ；HEST 的 16 塔一直只有报告 PCC。
本脚本复用已缓存的 16×72 嵌入（results/hest_emb/*.npz），只做 CPU 侧的图扩散，
因此可与深度线的 σ 直接并列讨论「容量买不到分辨率」这一主张。

口径与 effres.py 完全一致：
  · 惰性随机游走 W = (I + D⁻¹A)/2
  · σ(t) 由 δ 种子扩散的二阶矩实测，不用解析近似
  · 带 B_i = S_{t_{i-1}} − S_{t_i}，逐带算 truth 与 pred 的 per-gene PCC
  · ER(τ) = 仍满足 r ≥ τ 的最小 σ
唯一的平台差异：HEST 是 Visium 六角栅格（~100µm 间距），不是 HD 的 16µm 方栅格，
故邻接半径由每片自身的最近邻中位距推出，而非硬编码 29µm。
"""
import os, sys, glob, json, argparse, warnings
import numpy as np, anndata as ad
from scipy.spatial import cKDTree
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

B = "/blue/qsong1/wang.qing/he2st/HEST/eval/bench_data"
EMB = "/blue/qsong1/wang.qing/systema4ST/results/hest_emb"
LADDER = "/blue/qsong1/wang.qing/systema4ST/results/hest_ladder.json"


def build_operator(xy_um, k=6):
    """惰性随机游走。半径取该片最近邻中位距的 1.15 倍 —— Visium 六角栅格只保留一环。"""
    n = xy_um.shape[0]
    d, idx = cKDTree(xy_um).query(xy_um, k=min(k, n - 1) + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    pitch = float(np.median(d[:, 0]))
    cut = 1.15 * pitch
    ok = d <= cut
    rows = np.repeat(np.arange(n), ok.sum(1))
    A = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, idx[ok])), shape=(n, n))
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(1)).ravel()
    iso = deg == 0
    deg[iso] = 1.0
    W = (sparse.identity(n, format="csr", dtype=np.float32)
         + sparse.diags(1.0 / deg).tocsr() @ A) * 0.5
    return W.astype(np.float32), pitch, float(deg[~iso].mean()) if (~iso).any() else 0.0


def calibrate_sigma(W, xy_um, cps, n_seed=128, seed=0):
    """σ(t) 实测：δ 种子扩散后的空间二阶矩（每轴）。"""
    rng = np.random.default_rng(seed)
    s = rng.choice(xy_um.shape[0], size=min(n_seed, xy_um.shape[0]), replace=False)
    V = np.zeros((xy_um.shape[0], len(s)), np.float32)
    V[s, np.arange(len(s))] = 1.0
    out, t = {}, 0
    for cp in cps:
        while t < cp:
            V = W @ V; t += 1
        w = np.maximum(V, 0).astype(np.float64)
        tot = w.sum(0) + 1e-12
        mx = (w * xy_um[:, [0]]).sum(0) / tot
        my = (w * xy_um[:, [1]]).sum(0) / tot
        m2 = ((w * xy_um[:, [0]] ** 2).sum(0) + (w * xy_um[:, [1]] ** 2).sum(0)) / tot
        out[cp] = float(np.sqrt(max(np.median(m2 - mx ** 2 - my ** 2), 0.0) / 2.0))
    return out


def per_gene_pcc(A, Bm):
    """逐列 Pearson，忽略零方差列（扩散到高 t 时常见）。"""
    A = A - A.mean(0); Bm = Bm - Bm.mean(0)
    na = np.sqrt((A ** 2).sum(0)); nb = np.sqrt((Bm ** 2).sum(0))
    ok = (na > 1e-12) & (nb > 1e-12)
    if not ok.any():
        return np.array([np.nan])
    return ((A[:, ok] * Bm[:, ok]).sum(0) / (na[ok] * nb[ok]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--latent_dim", type=int, default=256)
    ap.add_argument("--tmax", type=int, default=1024)
    ap.add_argument("--tau", default="0.1,0.2,0.3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target", default="official", choices=["official", "cp10k"],
                    help="official=仅 log1p（官方代码实际做法）；cp10k=先按总计数归一化再 log1p")
    ap.add_argument("--skip_sigma", action="store_true", help="只算 PCC，跳过图扩散（用于排序对照）")
    args = ap.parse_args()
    args.out = args.out or f"results/hest_effres_{args.encoder}.json"

    cps = [1]
    while cps[-1] < args.tmax:
        cps.append(cps[-1] * 2)
    umpx = {k: v["umpx"] for k, v in json.load(open(LADDER)).items()}

    curve, sig_all, meta, own_pcc, eq_ladder = {}, {}, {}, {}, []
    for c in sorted(os.listdir(B)):
        if not os.path.isdir(os.path.join(B, c, "adata")):
            continue
        genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]
        cache = {}

        def get(sid):
            if sid in cache:
                return cache[sid]
            z = np.load(os.path.join(EMB, f"{sid}_{args.encoder}.npz"), allow_pickle=True)
            X, bc = z["X"], z["bc"].tolist()
            a = ad.read_h5ad(os.path.join(B, c, "adata", f"{sid}.h5ad"))
            pos = {b: i for i, b in enumerate(a.obs_names.astype(str))}
            sel = np.array([pos[b] for b in bc])           # 按嵌入的 barcode 顺序对齐
            sub = a[sel]
            gi = [list(sub.var_names.astype(str)).index(g) for g in genes]
            Y = np.asarray(sub.X.todense() if sparse.issparse(sub.X) else sub.X, np.float32)[:, gi]
            # 官方 hest.bench.st_dataset.normalize_adata 只做 sc.pp.log1p —— 没有 normalize_total，
            # 尽管它的 docstring 写着「按总计数归一化」。文档与代码不一致，以代码为准。
            # 多做一步 CP10K 会让 PCC 从 0.271 掉到 0.183（自检抓到）。
            Y = Y.astype(np.float64)
            if args.target == "cp10k":
                # 对照臂：官方目标未做深度归一化，故所有塔的分数里都掺着文库大小的空间梯度。
                # 此臂检验「塔的排序有多少是深度效应」——排序不变则为共模，变了则榜单名次部分来自深度。
                Y = Y / np.maximum(Y.sum(1, keepdims=True), 1.0) * 1e4
            Y = np.log1p(Y).astype(np.float32)
            xy = np.asarray(sub.obsm["spatial"], np.float64) * umpx.get(sid, 1.0)
            cache[sid] = (X.astype(np.float32), Y, xy)
            return cache[sid]

        for k in range(len(glob.glob(os.path.join(B, c, "splits", "test_*.csv")))):
            rd = lambda f: [l.split(",")[0] for l in
                            open(os.path.join(B, c, "splits", f)).read().splitlines()[1:] if l.strip()]
            tr, te = rd(f"train_{k}.csv"), rd(f"test_{k}.csv")
            Xtr = np.concatenate([get(s)[0] for s in tr])
            Ytr = np.concatenate([get(s)[1] for s in tr])
            pipe = Pipeline([("sc", StandardScaler()),
                             ("pca", PCA(n_components=min(args.latent_dim, Xtr.shape[1],
                                                          Xtr.shape[0] - 1), random_state=0))]).fit(Xtr)
            # 与官方 hest.bench.trainer.train_test_reg 逐参数一致：
            # alpha = 100/(D*G)、lsqr 求解、无截距。用 alpha=100 会欠拟合一万倍，
            # 那样测出的 σ 是回归器的，不是编码器的。
            Ztr = pipe.transform(Xtr)
            reg = Ridge(solver="lsqr", alpha=100.0 / (Ztr.shape[1] * Ytr.shape[1]),
                        random_state=0, fit_intercept=False, max_iter=1000).fit(Ztr, Ytr)

            for s in te:
                Xs, Ys, xy = get(s)
                P = reg.predict(pipe.transform(Xs)).astype(np.float32)
                # 自检：本脚本的预测须对得上官方报告 PCC，否则 σ 是在错的预测上测的
                own_pcc[s] = float(np.nanmean(per_gene_pcc(P, Ys)))
                if args.skip_sigma:
                    meta[s] = {"cohort": c, "n": int(Ys.shape[0])}
                    continue
                W, pitch, deg = build_operator(xy)
                sig = calibrate_sigma(W, xy, cps)
                for cp, v in sig.items():
                    sig_all.setdefault(cp, []).append(v)
                M = np.concatenate([Ys, P], 1).astype(np.float32)
                G = Ys.shape[1]
                prev, low_prev, t = M.copy(), M.copy(), 0
                lad_s = {}
                for cp in cps:
                    while t < cp:
                        prev = W @ prev; t += 1
                    band = low_prev - prev
                    r = per_gene_pcc(band[:, G:], band[:, :G])
                    curve.setdefault(cp, []).append(float(np.nanmean(r)))
                    # 阶梯（估计量 A，与 nine.py / 25 塔扫描同口径）：
                    # 真值扩散到 t 后与原真值的相关。eq_sigma = 该曲线上等于方法 PCC 的 σ。
                    # 与带通 ER(τ)（估计量 B）是两个不同的量，不可混比 —— 本节两个都出。
                    lad_s[cp] = float(np.nanmean(per_gene_pcc(prev[:, :G], Ys)))
                    low_prev = prev.copy()
                pts = sorted((sig[c], lad_s[c]) for c in cps)
                mp = own_pcc[s]
                eqs, fl = float("nan"), "右删失"
                if mp >= pts[0][1]:
                    eqs, fl = pts[0][0], "低于最细档"
                else:
                    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
                        if v0 >= mp >= v1:
                            eqs = s0 + (v0 - mp) / max(v0 - v1, 1e-12) * (s1 - s0); fl = "ok"; break
                eq_ladder.append((eqs, fl))
                meta[s] = {"cohort": c, "n": int(Ys.shape[0]), "pitch_um": round(pitch, 1),
                           "avg_deg": round(deg, 2)}
                print(f"  [{c} fold{k}] {s}: n={Ys.shape[0]} 间距={pitch:.0f}µm "
                      f"σ(1)={sig[cps[0]]:.0f} σ({cps[-1]})={sig[cps[-1]]:.0f}µm", flush=True)

    if args.skip_sigma:
        json.dump({"encoder": args.encoder, "target": args.target,
                   "per_sample_pcc": own_pcc, "mean_pcc": float(np.mean(list(own_pcc.values())))},
                  open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"[{args.encoder}/{args.target}] {len(own_pcc)} 样本 PCC 均值 "
              f"{np.mean(list(own_pcc.values())):.4f} → {args.out}")
        return
    sigma = {cp: float(np.mean(v)) for cp, v in sig_all.items()}
    cur = {cp: float(np.nanmean(v)) for cp, v in curve.items()}
    taus = [float(x) for x in args.tau.split(",")]

    # 检查点按 t 的 2 次幂走，σ ∝ √t ⇒ 相邻 σ 差 √2≈1.41×。若直接取「首个达到 τ 的检查点」，
    # 任何小于 1.41× 的塔间差异都会被量化掉（实测顶部三塔与其余恰好差一格 = 最小步长）。
    # 故在 log σ 上对带相关曲线插值求穿越点，并同时保留离散值以便对账。
    xs = sorted(sigma, key=lambda c: sigma[c])
    er, er_disc = {}, {}
    for tau in taus:
        hit = [sigma[c] for c in xs if cur.get(c, -1) >= tau]
        er_disc[str(tau)] = float(min(hit)) if hit else None
        val = None
        for i, c in enumerate(xs):
            if cur.get(c, -1) >= tau:
                if i == 0:
                    val = float(sigma[c])          # 首个检查点已达标 ⇒ 右删失于测量地板
                else:
                    p0, p1 = cur[xs[i - 1]], cur[c]
                    s0, s1 = np.log(sigma[xs[i - 1]]), np.log(sigma[c])
                    w = 0.0 if p1 == p0 else (tau - p0) / (p1 - p0)
                    val = float(np.exp(s0 + w * (s1 - s0)))
                break
        er[str(tau)] = val
    # 地板标记：τ 在最细带就已达标 ⇒ 该 τ 下的 ER 是「优于此值」，不是估计值
    floor_sigma = sigma[xs[0]]
    censored = {str(t): (er[str(t)] is not None and abs(er[str(t)] - floor_sigma) < 1e-6) for t in taus}

    ref_f = f"results/hest_reported_pcc_{args.encoder}.json"
    check = None
    if os.path.exists(ref_f):
        ref = json.load(open(ref_f))
        ks = [k for k in own_pcc if k in ref]
        if ks:
            a1 = np.array([own_pcc[k] for k in ks])
            b1 = np.array([ref[k]["pcc"] for k in ks])
            check = {"n": len(ks), "mine": float(a1.mean()), "official": float(b1.mean()),
                     "corr": float(np.corrcoef(a1, b1)[0, 1]),
                     "max_abs_diff": float(np.abs(a1 - b1).max())}
            print("\n自检 vs 官方报告 PCC: 本脚本 %.4f / 官方 %.4f  相关 %.3f  最大差 %.4f"
                  % (check["mine"], check["official"], check["corr"], check["max_abs_diff"]),
                  flush=True)

    ok_eq = [e for e, f in eq_ladder if f == "ok"]
    eq_stats = {"n_ok": len(ok_eq), "n_total": len(eq_ladder),
                "median_um": float(np.median(ok_eq)) if ok_eq else None,
                "mean_um": float(np.mean(ok_eq)) if ok_eq else None,
                "flags": {f: sum(1 for _, g in eq_ladder if g == f)
                          for f in set(g for _, g in eq_ladder)}}
    print(f"阶梯匹配 eq_sigma（与深度线同口径）: 中位 "
          f"{eq_stats['median_um'] if eq_stats['median_um'] is None else round(eq_stats['median_um'],1)}µm "
          f"({eq_stats['n_ok']}/{eq_stats['n_total']} ok)  标记 {eq_stats['flags']}", flush=True)

    res = {"encoder": args.encoder, "pcc_check": check, "eq_sigma_ladder": eq_stats,
           "eff_res_um_discrete": er_disc, "censored_at_floor": censored,
           "sigma_floor_um": float(floor_sigma), "sigma_um": {str(k): v for k, v in sigma.items()},
           "band_pcc": {str(k): v for k, v in cur.items()}, "eff_res_um": er,
           "n_samples": len(meta), "meta": meta}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n塔={args.encoder}  {len(meta)} 样本")
    for cp in cps:
        print(f"  σ={sigma[cp]:7.1f}µm   带 PCC={cur[cp]:.4f}")
    print(f"等价分辨率 ER(τ) 插值: { {k: (round(v,1) if v else None) for k,v in er.items()} }")
    print(f"            离散检查点: { {k: (round(v,1) if v else None) for k,v in er_disc.items()} }")
    print(f"            测量地板 {floor_sigma:.1f}µm  删失: {censored}")
    print(f"已存 {args.out}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
跨 72 样本的模型无关半 —— 逐样本的 A_coarse 阶梯 + 噪声天花板 + 稀疏度。

这是 Systema Fig 3c 对应图的左半边: 先把"该样本的信号有多少是粗尺度结构"量出来,
之后再与该样本上按官方协议报告的 PCC 求相关。本脚本完全不涉及任何预测模型。

每样本产出:
  sigma_um[t]     图扩散实测尺度(µm)
  ladder[t]       A_coarse(σ) = PCC(S_t(y), y) —— 纯解剖 oracle 在该样本能拿多少分
  c_raw           原始信号的拆半可重复性(Spearman-Brown 校正到全深度)
  c_band[t]       各频带的可重复性
  zero_frac / median_counts / platform / cohort

坐标: obsm['spatial'] × pixel_size_um_estimated (hest_coords.py 已验证 72/72 精确等于
      pxl_(col,row)_in_fullres, 且换算后最近邻间距全部为 100.0µm 的统一栅格)。
归一化: 全基因总数 CP10k + log1p, 再取官方 50 个评测基因 —— 与 HEST-benchmark 口径一致。
只读源数据。
"""
import os, sys, glob, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import sparse
from scipy.spatial import cKDTree
import evaluate as E

B = "/blue/qsong1/wang.qing/he2st/HEST/eval/bench_data"
SAMPLES = "/blue/qsong1/wang.qing/systema4ST/results/hest_samples.json"
PITCH_UM = 100.0                      # hest_coords.py: 全部样本统一 100µm 栅格


def build_operator(xy, k=8, cut_um=180.0):
    """惰性随机游走 W=(I+D^-1 A)/2。cut 取 1.8×pitch: 保留 4 正交(100µm)+4 对角(141µm)。"""
    n = xy.shape[0]
    d, idx = cKDTree(xy).query(xy, k=min(k + 1, n))
    d, idx = d[:, 1:], idx[:, 1:]
    ok = d <= cut_um
    rows = np.repeat(np.arange(n), ok.sum(1))
    A = sparse.csr_matrix((np.ones(int(ok.sum()), np.float32), (rows, idx[ok])), shape=(n, n))
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(1)).ravel()
    iso = deg == 0
    deg[iso] = 1.0
    W = (sparse.identity(n, format="csr", dtype=np.float32)
         + sparse.diags(1.0 / deg).tocsr() @ A) * 0.5
    return W.astype(np.float32), float(deg[~iso].mean() if (~iso).any() else 0), int(iso.sum())


def calibrate_sigma(W, xy, cps, n_seed=128, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.choice(xy.shape[0], size=min(n_seed, xy.shape[0]), replace=False)
    V = np.zeros((xy.shape[0], len(s)), np.float32)
    V[s, np.arange(len(s))] = 1.0
    out, t = {}, 0
    for cp in cps:
        while t < cp:
            V = W @ V; t += 1
        w = np.maximum(V, 0).astype(np.float64)
        tot = w.sum(0) + 1e-12
        mx = (w * xy[:, [0]]).sum(0) / tot
        my = (w * xy[:, [1]]).sum(0) / tot
        m2 = ((w * xy[:, [0]] ** 2).sum(0) + (w * xy[:, [1]] ** 2).sum(0)) / tot
        out[cp] = float(np.sqrt(max(np.median(m2 - mx ** 2 - my ** 2), 0.0) / 2.0))
    return out


def norm_log(C, mode="log1p"):
    """与 HEST-benchmark 对齐: 官方 normalize_adata 只做 sc.pp.log1p, 不做总数归一化。
    两边必须用同一种归一化, 否则 ladder 与 reported PCC 描述的不是同一个信号。"""
    if mode == "log1p":
        Y = C.copy().tocsr(); Y.data = np.log1p(Y.data); return Y
    tot = np.asarray(C.sum(1)).ravel(); tot[tot == 0] = 1.0
    Y = C.multiply((1e4 / tot)[:, None]).tocsr(); Y.data = np.log1p(Y.data)
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--norm", default="log1p", choices=["log1p","cp10k"])
    ap.add_argument("--out", default="results/hest_ladder.json")
    args = ap.parse_args()
    cps = [1]
    while cps[-1] < args.tmax:
        cps.append(cps[-1] * 2)

    info = {r["sid"]: r for r in json.load(open(SAMPLES))}
    res = {}
    for c in sorted(os.listdir(B)):
        d = os.path.join(B, c, "adata")
        if not os.path.isdir(d):
            continue
        g50 = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]
        for p in sorted(glob.glob(os.path.join(d, "*.h5ad"))):
            sid = os.path.basename(p)[:-5]
            a = ad.read_h5ad(p)
            umpx = info[sid]["umpx"]
            xy = np.asarray(a.obsm["spatial"], np.float64) * umpx
            C = sparse.csr_matrix(a.X)
            cols = np.array([list(a.var_names).index(g) for g in g50])
            y = np.asarray(norm_log(C, args.norm)[:, cols].todense(), np.float32)

            W, deg, iso = build_operator(xy)
            sig = calibrate_sigma(W, xy, cps)

            ladder, cur, t = {}, y.copy(), 0
            S = {}
            for cp in cps:
                while t < cp:
                    cur = W @ cur; t += 1
                S[cp] = cur.copy()
                ladder[str(cp)] = float(E.per_gene_pcc(S[cp], y).mean())

            # 噪声天花板: 二项拆半 ×reps, 原始信号 + 各频带
            rng = np.random.default_rng(0)
            craw, cband = [], {str(cp): [] for cp in cps}
            for _ in range(args.reps):
                h = rng.binomial(C.data.astype(np.int64), 0.5).astype(np.float32)
                A_ = sparse.csr_matrix((h, C.indices, C.indptr), shape=C.shape)
                B_ = sparse.csr_matrix((C.data - h, C.indices, C.indptr), shape=C.shape)
                ya = np.asarray(norm_log(A_, args.norm)[:, cols].todense(), np.float32)
                yb = np.asarray(norm_log(B_, args.norm)[:, cols].todense(), np.float32)
                craw.append(float(E.per_gene_pcc(ya, yb).mean()))
                pa, pb, la, lb, t2 = ya.copy(), yb.copy(), ya.copy(), yb.copy(), 0
                for cp in cps:
                    while t2 < cp:
                        pa = W @ pa; pb = W @ pb; t2 += 1
                    cband[str(cp)].append(float(E.per_gene_pcc(la - pa, lb - pb).mean()))
                    la, lb = pa.copy(), pb.copy()
            sb = lambda v: 2 * v / (1 + v) if v > -1 else float("nan")
            res[sid] = dict(cohort=c, platform=info[sid]["tech"], n=int(a.n_obs),
                            n_genes_total=int(a.n_vars), umpx=umpx,
                            zero_frac=float((C[:, cols].todense() == 0).mean()),
                            median_counts=float(np.median(np.asarray(C.sum(1)).ravel())),
                            avg_degree=deg, isolated=iso,
                            sigma_um={str(k): v for k, v in sig.items()}, ladder=ladder,
                            c_raw_half=float(np.mean(craw)), c_raw=sb(float(np.mean(craw))),
                            c_band={k: sb(float(np.mean(v))) for k, v in cband.items()})
            print(f"{c:10s}{sid:10s} n={a.n_obs:6d} 零元={res[sid]['zero_frac']*100:5.1f}% "
                  f"σ范围=[{sig[cps[0]]:.0f},{sig[cps[-1]]:.0f}]µm "
                  f"A_coarse[{sig[cps[0]]:.0f}µm]={ladder[str(cps[0])]:.3f} "
                  f"[{sig[cps[-1]]:.0f}µm]={ladder[str(cps[-1])]:.3f} c_raw={res[sid]['c_raw']:.3f}",
                  flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n=== 汇总 ({len(res)} 样本) ===")
    for plat in sorted(set(r["platform"] for r in res.values())):
        sub = [r for r in res.values() if r["platform"] == plat]
        zf = np.mean([r["zero_frac"] for r in sub]) * 100
        cr = np.mean([r["c_raw"] for r in sub])
        print(f"  {plat:10s} n={len(sub):3d} 平均零元={zf:5.1f}% 平均 c_raw={cr:.3f}")
    print(f"已存 {args.out}")


if __name__ == "__main__":
    main()

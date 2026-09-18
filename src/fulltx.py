# -*- coding: utf-8 -*-
"""
全转录组重做 —— 检验所有结论是否只成立于那 200 个最变化的基因。

至今全部实验跑在 binned_16um.h5ad 的 200 个 HVG 上；原始 adata_16um.h5ad 有 18,085 个。
那 200 个是空间性最强、天花板最高的一批，所以等价 σ≈110µm、per-gene PCC~Moran=0.838
都可能是这个选择的产物。本脚本在全谱上重做，并把项目原来那 200 个作为显式对照行。

本版按一轮 4 视角对抗审查（28 agent）重写，修掉的实质缺陷：
  · ridge 改回 baselines.ridge_predict —— 手写版只中心化不标准化，固定 alpha=1e4
    等价于给每个特征施加 alpha_j=1e4/sd_j²，与项目既有结果不可比。
  · 面板 "200" 行改为项目真实的 200 基因面板（prior200），不是全谱方差前 200。
    否则本实验的核心对比根本没做。
  · 基因过滤只用训练片（原版读了测试片），且放宽到 0.5% —— SVG 的真阴性正在低表达端。
  · 断言两张片 var_names 完全一致（原版只假设）。
  · tmax 恢复 2048（原版砍到 1024，实测 σ(1024)=302µm，全谱 PCC 可能掉出阶梯）。
  · Moran 膨胀改报差值 I_p−I_t 并按真实 Moran 四分位分层 —— 原版比值的 I_t>0.01
    守卫恰好删掉膨胀最大的基因。
  · SVG 饱和是结论本身：显式报出 BH-0.05 对应的 Moran 阈值；主指标改为排序重合
    与「预测 top-K 里落在真实最低四分位的比例」（原版 low_in_true 恒为 0）。
  · moran_z 分块 + float64 归约；负方差记 NaN 而非 1e-30 地板（后者= 自动显著）。
  · npz 存盘移到面板扫描之前；阶梯按基因块并行。
  · 报逐片数值，不只两片均值；补噪声天花板作为 PCC~Moran 的混杂控制。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import sparse
from scipy.stats import pearsonr, spearmanr, norm
from joblib import Parallel, delayed
import evaluate as E
from baselines import ridge_predict
from effres import build_operator, calibrate_sigma, PX_PER_UM, SLIDES, SEB

RAW = "/path/to/align_workspace/st_bench/data/{s}/adata_16um.h5ad"
ap = argparse.ArgumentParser()
ap.add_argument("--tower", default="hibou_l");    ap.add_argument("--tmax", type=int, default=2048)
ap.add_argument("--chunk", type=int, default=1200); ap.add_argument("--minfrac", type=float, default=0.005)
ap.add_argument("--njobs", type=int, default=6);  ap.add_argument("--ceil_reps", type=int, default=2)
ap.add_argument("--panels", default="20,50,100,200,500,1000,2000,5000,10000,0")
a_ = ap.parse_args()
cps = [1]
while cps[-1] < a_.tmax: cps.append(cps[-1]*2)
RES = "/path/to/systema4ST/results"

def moran_moments(A):
    S0 = float(A.sum()); S1 = 0.5*float(((A + A.T).power(2)).sum())
    rs = np.asarray(A.sum(1)).ravel(); cs = np.asarray(A.sum(0)).ravel()
    return S0, S1, float(((rs + cs)**2).sum())

def moran_z_chunk(A, X, S0, S1, S2):
    """逐基因 Moran's I 与随机化 z。float64 归约（float32 在 n=132k 上丢约 3 位有效数字）。"""
    n = float(X.shape[0])
    Xc = (X - X.mean(0, dtype=np.float64)).astype(np.float64)
    s2 = (Xc**2).sum(0) + 1e-12
    I = (n/S0) * (Xc*(A@Xc)).sum(0) / s2
    b2 = n * (Xc**4).sum(0) / (s2**2)
    EI = -1.0/(n-1)
    num = n*((n*n-3*n+3)*S1 - n*S2 + 3*S0*S0) - b2*((n*n-n)*S1 - 2*n*S2 + 6*S0*S0)
    VI = num/((n-1)*(n-2)*(n-3)*S0*S0) - 1.0/((n-1)**2)
    z = np.where(VI > 0, (I - EI)/np.sqrt(np.where(VI > 0, VI, 1.0)), np.nan)  # 负方差→NaN，不是自动显著
    return I, z

def moran_all(A, X, S0, S1, S2, chunk):
    Is, zs = [], []
    for c0 in range(0, X.shape[1], chunk):
        i, z = moran_z_chunk(A, np.ascontiguousarray(X[:, c0:c0+chunk]), S0, S1, S2)
        Is.append(i); zs.append(z)
    return np.concatenate(Is), np.concatenate(zs)

def bh(p, q=0.05):
    p = np.where(np.isfinite(p), p, 1.0)
    o = np.argsort(p); m = len(p)
    k = np.where(p[o] <= q*np.arange(1, m+1)/m)[0]
    out = np.zeros(m, bool)
    if len(k): out[o[:k[-1]+1]] = True
    return out

def ladder_chunk(W, ych, cps):
    cur = ych.copy(); t = 0; out = np.zeros((len(cps), ych.shape[1]), np.float32)
    for ci, cp in enumerate(cps):
        while t < cp: cur = W @ cur; t += 1
        out[ci] = E.per_gene_pcc(cur, ych)
    return out

# ── 载入 ───────────────────────────────────────────────────────────────
b = ad.read_h5ad(SEB.H5AD)
bexpr = np.nan_to_num(np.asarray(b.X, np.float32))
bgene = np.asarray(b.var["gene"]).astype(str)
bslide = b.obs["slide_id"].astype(str).values
bpxl = np.asarray(b.obsm["pxl"], np.float64)

Yf, XY, EMB, CNT, GN = {}, {}, {}, {}, {}
for s in SLIDES:
    r = ad.read_h5ad(RAW.format(s=s))
    C = sparse.csr_matrix(r.X)
    GN[s] = np.asarray(r.var_names).astype(str)
    Y = np.log1p(np.asarray(C.todense(), np.float32))
    m = bslide == s
    assert Y.shape[0] == m.sum(), f"{s}: 原始 {Y.shape[0]} vs 预处理 {m.sum()}"
    emb = np.nan_to_num(np.load(os.path.join(SEB.EMBDIR, f"emb_{a_.tower}_{s[-2:]}.npy")).astype(np.float32))
    assert emb.shape[0] == Y.shape[0], f"{s}: 嵌入 {emb.shape[0]} vs 表达 {Y.shape[0]}"
    gi = {g: i for i, g in enumerate(GN[s])}
    com = [(j, gi[g]) for j, g in enumerate(bgene) if g in gi][:40]
    bs = bexpr[m]
    # 行序核对用置换零分布, 不用绝对阈值: 两文件归一化口径不同(预处理片做过文库归一化),
    # 对齐时 r≈0.90 而非 1.0 —— 首轮我用 0.95 硬阈值把正确对齐误判为错位。
    rr = np.array([pearsonr(bs[:, j], Y[:, k])[0] for j, k in com])
    pm = np.random.default_rng(0).permutation(Y.shape[0])
    rs = np.array([pearsonr(bs[:, j], Y[pm, k])[0] for j, k in com])
    print(f"[{s}] n={Y.shape[0]} 基因={Y.shape[1]}  行序核对 {len(com)} 基因: "
          f"对齐 r 中位={np.median(rr):.4f}(最小 {np.min(rr):.4f})  "
          f"打乱 r 中位={np.median(rs):.4f}", flush=True)
    assert np.median(rr) > 0.5 and abs(np.median(rs)) < 0.1, \
        f"{s}: 行序不一致（对齐 {np.median(rr):.3f} vs 打乱 {np.median(rs):.3f}）"
    Yf[s], CNT[s], XY[s], EMB[s] = Y, C, bpxl[m] / PX_PER_UM[s], emb
    del r
assert np.array_equal(GN[SLIDES[0]], GN[SLIDES[1]]), "两张片的 var_names 顺序不一致"
G = GN[SLIDES[0]]
prior200 = np.array([i for i, g in enumerate(G) if g in set(bgene)])
print(f"两片基因顺序一致；项目原 200 基因面板在全谱中命中 {len(prior200)}/200", flush=True)

OUT = {}
for s in SLIDES:
    tr_s = [x for x in SLIDES if x != s][0]
    Ytr, Yte = Yf[tr_s], Yf[s]
    gk = np.where((Ytr > 0).mean(0) >= a_.minfrac)[0]          # 只用训练片，无泄漏
    y = np.ascontiguousarray(Yte[:, gk]); ytr = np.ascontiguousarray(Ytr[:, gk])
    n, ng = y.shape
    print(f"\n[{s}] 训练片={tr_s[-2:]}  训练片表达率≥{a_.minfrac:.1%} 的基因 {ng}/{len(G)} "
          f"（丢弃 {len(G)-ng}）", flush=True)

    W = build_operator(XY[s], k=8, cut_um=29.0)
    A = (W > 0).astype(np.float32); A = ((A + A.T) > 0).astype(np.float32)
    A.setdiag(0); A.eliminate_zeros()
    S0, S1, S2 = moran_moments(A)
    sig = calibrate_sigma(W, XY[s], cps)

    pred = ridge_predict(EMB[tr_s], ytr, EMB[s], 1e4)          # 项目正典实现
    pcc_g = E.per_gene_pcc(pred, y)
    print(f"  Ridge 完成  全谱平均 PCC={np.nanmean(pcc_g):.4f}", flush=True)

    lad = np.concatenate(Parallel(n_jobs=a_.njobs, verbose=1)(
        delayed(ladder_chunk)(W, np.ascontiguousarray(y[:, c0:c0+a_.chunk]), cps)
        for c0 in range(0, ng, a_.chunk)), axis=1)
    print(f"  阶梯完成 {lad.shape}", flush=True)

    I_t, z_t = moran_all(A, y, S0, S1, S2, a_.chunk)
    I_p, z_p = moran_all(A, pred, S0, S1, S2, a_.chunk)

    # 噪声天花板（二项拆半 + Spearman-Brown），作为 PCC~Moran 的混杂控制
    Cs = CNT[s][:, gk].tocsr(); rng = np.random.default_rng(0); ch = []
    for _ in range(a_.ceil_reps):
        h = rng.binomial(Cs.data.astype(np.int64), 0.5).astype(np.float32)
        f = lambda dat: np.log1p(np.asarray(
            sparse.csr_matrix((dat, Cs.indices, Cs.indptr), shape=Cs.shape).todense(), np.float32))
        ch.append(E.per_gene_pcc(f(h), f(Cs.data - h)))
    ch = np.mean(ch, 0); ceil_g = 2*ch/(1+ch)

    np.savez_compressed(f"{RES}/fulltx_{s[-2:]}.npz", gene=G[gk], pcc=pcc_g, ceil=ceil_g,
                        I_true=I_t, I_pred=I_p, z_true=z_t, z_pred=z_p, ladder=lad,
                        sigma=np.array([sig[c] for c in cps]), gk=gk)   # 先存盘再分析
    print(f"  已存 fulltx_{s[-2:]}.npz", flush=True)

    def eq_sigma(v, curve):
        pts = sorted(zip([sig[c] for c in cps], curve))
        if v >= pts[0][1]: return pts[0][0], "低于最细档"
        for (a0, v0), (a1, v1) in zip(pts, pts[1:]):
            if v0 >= v >= v1: return a0 + (v0-v)/max(v0-v1, 1e-12)*(a1-a0), "ok"
        return np.nan, f">{pts[-1][0]:.0f}µm 右删失"

    order = np.argsort(-ytr.var(0))                            # HVG 排名用训练片
    pos = {g: i for i, g in enumerate(gk)}
    p200 = np.array([pos[i] for i in prior200 if i in pos])
    pan = {}
    for k in [int(x) for x in a_.panels.split(",")] + [-1]:
        idx = p200 if k == -1 else (order if k == 0 else order[:min(k, ng)])
        kk = "prior200" if k == -1 else ("all" if k == 0 else str(k))
        if not len(idx): continue
        v = float(np.nanmean(pcc_g[idx])); e, flag = eq_sigma(v, lad[:, idx].mean(1))
        d = I_p[idx] - I_t[idx]
        pan[kk] = dict(n=len(idx), pcc=v, eq=float(e), eq_flag=flag,
                       moran_true=float(np.nanmean(I_t[idx])),
                       moran_diff=float(np.nanmedian(d)),
                       ceil=float(np.nanmean(ceil_g[idx])))
    # SVG: 饱和度本身是结论 —— 报 BH-0.05 对应的 Moran 阈值
    S_t, S_p = bh(norm.sf(z_t)), bh(norm.sf(z_p))
    thr = float(np.min(I_t[S_t])) if S_t.any() else np.nan
    q = np.nanpercentile(I_t, [25, 50, 75])
    infl = [float(np.nanmedian((I_p - I_t)[(I_t > lo) & (I_t <= hi)]))
            for lo, hi in [(-np.inf, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], np.inf)]]
    OUT[s] = dict(n_gene=ng, n_drop=int(len(G)-ng), panels=pan,
                  sigma_um={str(c): sig[c] for c in cps},
                  pcc_moran=[float(pearsonr(I_t, pcc_g)[0]), float(spearmanr(I_t, pcc_g)[0])],
                  pcc_ceil=[float(pearsonr(ceil_g, pcc_g)[0]), float(spearmanr(ceil_g, pcc_g)[0])],
                  svg_true=int(S_t.sum()), svg_pred=int(S_p.sum()),
                  svg_thr_moran=thr, moran_q=[float(x) for x in q], moran_diff_by_q=infl)
    lowq = I_t <= q[0]
    for K in (100, 500, 1000, 2000):
        tp_, tt = np.argsort(-I_p)[:K], np.argsort(-I_t)[:K]
        OUT[s][f"topK{K}"] = dict(overlap=len(set(tp_) & set(tt))/K,
                                  lowq_in_pred=float(lowq[tp_].mean()),
                                  lowq_rate_overall=float(lowq.mean()))
    del pred, y, ytr, lad, Cs

print(f"\n{'='*100}"); print("基因面板扫描（全转录组；逐片 + 均值）"); print("="*100)
hdr = f"{'面板':>10s}{'基因':>7s}" + "".join(f"{'PCC('+s[-2:]+')':>11s}" for s in SLIDES) + \
      f"{'PCC均值':>10s}{'相对20':>9s}{'等价σ':>9s}{'天花板':>9s}{'Moran差':>10s}"
print(hdr)
ks = ["20", "50", "100", "prior200", "200", "500", "1000", "2000", "5000", "10000", "all"]
ok = lambda k: all(k in OUT[s]["panels"] for s in SLIDES)
g = lambda k, z: np.nanmean([OUT[s]["panels"][k][z] for s in SLIDES])
p20 = g("20", "pcc")
for k in ks:
    if not ok(k): continue
    row = f"{k:>10s}{g(k,'n'):>7.0f}" + "".join(f"{OUT[s]['panels'][k]['pcc']:>11.4f}" for s in SLIDES)
    e = g(k, "eq"); fl = OUT[SLIDES[0]]["panels"][k]["eq_flag"]
    print(row + f"{g(k,'pcc'):>10.4f}{g(k,'pcc')/p20:>9.3f}" +
          (f"{e:>9.0f}" if np.isfinite(e) else f"{'右删失':>9s}") +
          f"{g(k,'ceil'):>9.3f}{g(k,'moran_diff'):>10.3f}")

print(f"\n{'='*100}"); print("SVG 检验：n=132k 下的饱和"); print("="*100)
for s in SLIDES:
    o = OUT[s]
    print(f"  [{s[-2:]}] 基因={o['n_gene']}  BH-0.05 判为 SVG：真实 {o['svg_true']} "
          f"({o['svg_true']/o['n_gene']:.1%})  预测 {o['svg_pred']} ({o['svg_pred']/o['n_gene']:.1%})")
    print(f"        显著性阈值 = Moran's I ≥ {o['svg_thr_moran']:.4f}"
          f"   （真实 Moran 四分位 {o['moran_q'][0]:.3f}/{o['moran_q'][1]:.3f}/{o['moran_q'][2]:.3f}）")
    for K in (100, 500, 1000, 2000):
        d = o[f"topK{K}"]
        print(f"        top-{K:<5d} 重合={d['overlap']:.3f}  预测 top-K 中真实最低四分位占比="
              f"{d['lowq_in_pred']:.3f}（全谱基线 {d['lowq_rate_overall']:.3f}）")
print(f"\n  Moran 差值 I_pred−I_true，按真实 Moran 四分位：")
for s in SLIDES:
    print(f"    [{s[-2:]}] Q1={OUT[s]['moran_diff_by_q'][0]:+.3f}  Q2={OUT[s]['moran_diff_by_q'][1]:+.3f}"
          f"  Q3={OUT[s]['moran_diff_by_q'][2]:+.3f}  Q4={OUT[s]['moran_diff_by_q'][3]:+.3f}")
print(f"\n  per-gene PCC 的相关（全谱）：")
for s in SLIDES:
    print(f"    [{s[-2:]}] ~真实 Moran r={OUT[s]['pcc_moran'][0]:.3f} ρ={OUT[s]['pcc_moran'][1]:.3f}"
          f"   ~噪声天花板 r={OUT[s]['pcc_ceil'][0]:.3f} ρ={OUT[s]['pcc_ceil'][1]:.3f}")
json.dump(OUT, open(f"{RES}/fulltx.json", "w"), indent=2, ensure_ascii=False)
print(f"\n已存 {RES}/fulltx.json + fulltx_P2/P5.npz")

# -*- coding: utf-8 -*-
"""
9 张片上的等价分辨率 —— 回答「σ≈110µm 是指标的性质，还是结直肠肿瘤的组织结构尺度」。

深度线原本只有 P2/P5（同一研究、同一癌种），最致命的反驳是「110µm 只是结肠肿瘤的
腺体/间质块尺度」。本脚本在 9 张片上重跑：
  人 CRC P1/P2/P5（三个病人）· 人癌旁正常 P3 · 人胰腺 · 鼠脑/肾/胚胎/小肠
全部来自 10x 官方原始数据，统一管线（ctx_px=224, grid=1, level 0, 10x 自带坐标）。

评分协议用【空间块 CV + 汇总打分】：第 13 节已证明汇总口径下随机划分与块划分
几乎等价（0.7025 vs 0.6976），且逐折平均会引入高达 0.27 的假象。这是唯一能在
9 张异质切片上统一施加、又不受折聚合假象污染的协议。

天花板一律用可达上限 √c（第 15 节：无噪预测器的相关上限是 √c 而非 c）。
"""
import os, sys, json, argparse, numpy as np, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
import evaluate as E
from effres import build_operator, calibrate_sigma

PREP = "/blue/qsong1/wang.qing/systema4ST/data/prepped"
EMB  = "/blue/qsong1/wang.qing/systema4ST/results/emb9"
RES  = "/blue/qsong1/wang.qing/systema4ST/results"
TISSUE = {  # 组织学分组，用于最终对比
 "Visium_HD_Human_Colon_Cancer_P1": ("人 CRC P1", "肿瘤"),
 "Visium_HD_Human_Colon_Cancer_P2": ("人 CRC P2", "肿瘤"),
 "Visium_HD_Human_Colon_Cancer_P5": ("人 CRC P5", "肿瘤"),
 "Visium_HD_Human_Colon_Normal_P3": ("人结肠正常", "正常"),
 "Visium_HD_Human_Pancreas":        ("人胰腺", "正常"),
 "Visium_HD_Mouse_Brain":           ("鼠脑", "正常"),
 "Visium_HD_Mouse_Kidney":          ("鼠肾", "正常"),
 "Visium_HD_Mouse_Embryo":          ("鼠胚胎", "正常"),
 "Visium_HD_Mouse_Small_Intestine": ("鼠小肠", "正常")}

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--enc", default="hibou_l")
ap.add_argument("--emb", default=None, help="覆盖嵌入路径（用于图像来源对照）")
ap.add_argument("--h5ad", default=None, help="覆盖 h5ad 路径（Xenium 走 data/prepped_xen）")
ap.add_argument("--label", default=None, help="报告用的中文名")
ap.add_argument("--outdir", default="nine")
ap.add_argument("--hvg", type=int, default=50); ap.add_argument("--grid", type=int, default=16)
ap.add_argument("--tmax", type=int, default=2048); ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--tag", default="")
# 邻接半径必须随分箱尺度走：29µm 是为 16µm 方栅格定的（4 正交 + 4 对角 22.6µm）。
# 8µm 沿用 29µm 会把二、三环也连进来；64µm 则一个邻居都连不上。
ap.add_argument("--cut_um", type=float, default=None, help="默认 1.8 × 分箱边长")
ap.add_argument("--pitch_um", type=float, default=16.0)
ap.add_argument("--train", default=None,
                help="逗号分隔的训练片；给出则走跨片协议（与片内块 CV 口径不同，不可混比）")
a_ = ap.parse_args()
cps = [1]
while cps[-1] < a_.tmax: cps.append(cps[-1]*2)

N = a_.name
a = ad.read_h5ad(a_.h5ad or f"{PREP}/{N}_16um.h5ad")
C = sparse.csr_matrix(a.X)
Y = np.log1p(np.asarray(C.todense(), np.float32))
X = np.nan_to_num(np.load(a_.emb or f"{EMB}/emb_{a_.enc}_{N}.npy").astype(np.float32))
assert X.shape[0] == Y.shape[0], f"嵌入 {X.shape[0]} vs 表达 {Y.shape[0]}"
xy = np.asarray(a.obsm["pxl"], np.float64) / float(a.uns["px_per_um"])
n = len(Y)
print(f"[{N}] n={n} 基因={Y.shape[1]} 嵌入={X.shape[1]}维 px/µm={a.uns['px_per_um']:.4f}", flush=True)

CUT = a_.cut_um if a_.cut_um else 1.8 * a_.pitch_um
print(f"  邻接半径 {CUT:.1f}µm (分箱 {a_.pitch_um:.0f}µm)", flush=True)
W = build_operator(xy, k=8, cut_um=CUT)
A = (W > 0).astype(np.float32); A = ((A + A.T) > 0).astype(np.float32)
A.setdiag(0); A.eliminate_zeros()
sig = calibrate_sigma(W, xy, cps)

if a_.train:
    # ── 跨片协议：在别的切片上训练，整片预测（与 110µm 那条线同口径）──
    TR = a_.train.split(",")
    Xs, Ys = [], []
    for t in TR:
        at = ad.read_h5ad(f"{PREP}/{t}_16um.h5ad")
        assert list(at.var_names) == list(a.var_names), f"{t} 与 {N} 基因空间不一致"
        Ys.append(np.log1p(np.asarray(sparse.csr_matrix(at.X).todense(), np.float32)))
        Xs.append(np.nan_to_num(np.load(f"{EMB}/emb_{a_.enc}_{t}.npy").astype(np.float32)))
        del at
    Xtr, Ytr = np.concatenate(Xs), np.concatenate(Ys); del Xs, Ys
    gidx = E.topk_hvg(Ytr, a_.hvg)                 # 基因在训练片上选，无泄漏
    y = Y[:, gidx]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    P = Ridge(alpha=1e4).fit((Xtr-mu)/sd, Ytr[:, gidx]).predict((X-mu)/sd).astype(np.float32)
    del Xtr, Ytr
    cov = np.ones(n, bool)
    pcc_g = E.per_gene_pcc(P, y); pcc = float(np.nanmean(pcc_g))
    print(f"  跨片（训练={'+'.join(t[-2:] for t in TR)}）PCC={pcc:.4f}", flush=True)
else:
    _run_block = True

# ── 空间块 CV + 汇总打分 ──
g = a_.grid
if not a_.train:
    qx = np.quantile(xy[:,0], np.linspace(0,1,g+1)); qx[-1] += 1
    qy = np.quantile(xy[:,1], np.linspace(0,1,g+1)); qy[-1] += 1
    fold = (np.searchsorted(qx, xy[:,0], "right")-1)*g + (np.searchsorted(qy, xy[:,1], "right")-1)
    gidx = E.topk_hvg(Y, a_.hvg)
    y = Y[:, gidx]
    P = np.full_like(y, np.nan)
    for f in np.unique(fold):
        te = fold == f
        # 块的物理尺寸不随分箱变（都是片宽的 1/g），但粗分箱下同一块里的 bin 数按 (16/pitch)^2 减少。
        # 阈值 100 是为 16µm 定的；64µm 时每块只剩约 45 个 bin，沿用会把所有块跳过 → 覆盖率 0。
        MIN_TE = max(20, int(100 * (16.0 / a_.pitch_um) ** 2))
        if te.sum() < MIN_TE or (~te).sum() < 2000: continue
        mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-8
        m = Ridge(alpha=1e4).fit((X[~te]-mu)/sd, y[~te])
        P[te] = m.predict((X[te]-mu)/sd).astype(np.float32)
    cov = np.isfinite(P[:,0])
    pcc_g = E.per_gene_pcc(P[cov], y[cov])
    pcc = float(np.nanmean(pcc_g))
    print(f"  块{g}×{g} 汇总 PCC={pcc:.4f}  覆盖={cov.mean():.3f}  每块最少 {max(20, int(100*(16.0/a_.pitch_um)**2))} 点", flush=True)

# ── 阶梯 + 等价 σ ──
lad, cur, t = {}, y.copy(), 0
for cp in cps:
    while t < cp: cur = W @ cur; t += 1
    lad[cp] = float(np.nanmean(E.per_gene_pcc(cur, y)))
pts = sorted((sig[c], lad[c]) for c in cps)
eq, flag = np.nan, "右删失"
if pcc >= pts[0][1]: eq, flag = pts[0][0], "低于最细档"
else:
    for (s0,v0),(s1,v1) in zip(pts, pts[1:]):
        if v0 >= pcc >= v1: eq, flag = s0 + (v0-pcc)/max(v0-v1,1e-12)*(s1-s0), "ok"; break

# ── 噪声天花板（可达上限 √c）──
rng = np.random.default_rng(0); ch = []
Cg = C[:, gidx].tocsr()
for _ in range(a_.reps):
    h = rng.binomial(Cg.data.astype(np.int64), 0.5).astype(np.float32)
    f_ = lambda d: np.log1p(np.asarray(sparse.csr_matrix((d, Cg.indices, Cg.indptr), shape=Cg.shape).todense(), np.float32))
    ch.append(E.per_gene_pcc(f_(h), f_(Cg.data - h)))
c_half = float(np.nanmean(np.mean(ch,0))); c = 2*c_half/(1+c_half); ceil = float(np.sqrt(max(c,0)))

# ── Moran ──
# 图必须限制到被 CV 覆盖的子集：少数过小的块被跳过，cov 覆盖率 ~99.8%，
# 若仍用全图 A 会维度不符（首轮 39033276-280 即因此失败）。
Ac = A[cov][:, cov].tocsr()
def moran(M):
    Mc = M - M.mean(0)
    return (M.shape[0]/Ac.sum()) * (Mc*(Ac@Mc)).sum(0) / ((Mc**2).sum(0)+1e-12)
I_t = moran(y[cov]); I_p = moran(P[cov])
r_pm = pearsonr(I_t, pcc_g)[0]; rho_pm = spearmanr(I_t, pcc_g)[0]

out = dict(name=N, label_cn=(a_.label or N), protocol=("跨片:"+a_.train if a_.train else f"片内块{g}x{g}"), label=TISSUE.get(N, (N,"?"))[0], group=TISSUE.get(N, (N,"?"))[1],
           n=n, n_gene=int(Y.shape[1]), px_per_um=float(a.uns["px_per_um"]),
           pcc=pcc, eq_sigma=float(eq), eq_flag=flag, ceiling=ceil, c_full=float(c),
           skill=float(pcc/ceil) if ceil > 0 else np.nan,
           counts_median=float(np.median(np.asarray(C.sum(1)))),
           moran_true=float(np.nanmean(I_t)), moran_pred=float(np.nanmean(I_p)),
           moran_diff=float(np.nanmedian(I_p - I_t)),
           pcc_moran_r=float(r_pm), pcc_moran_rho=float(rho_pm),
           sigma_um={str(k): sig[k] for k in cps}, ladder={str(k): lad[k] for k in cps})
os.makedirs(f"{RES}/{a_.outdir}", exist_ok=True)
json.dump(out, open(f"{RES}/{a_.outdir}/{N}{a_.tag}.json", "w"), indent=2, ensure_ascii=False)
print(f"\n=== {a_.label or TISSUE.get(N,(N,))[0]} ===")
print(f"  PCC={pcc:.4f}  等价σ={eq:.0f}µm ({flag})  可达上限√c={ceil:.3f}  归一技能={pcc/ceil:.3f}")
print(f"  真实Moran={np.nanmean(I_t):.3f} 预测Moran={np.nanmean(I_p):.3f} 差={np.nanmedian(I_p-I_t):+.3f}")
print(f"  PCC~Moran r={r_pm:.3f} ρ={rho_pm:.3f}")

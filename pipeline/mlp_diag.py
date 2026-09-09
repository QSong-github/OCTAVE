# -*- coding: utf-8 -*-
"""MLP 头为何普遍低于岭回归 —— 分辨「优化不足(欠拟合)」与「拟合到样本身份(跨样本过拟合)」。
同一份 StandardScaler→PCA-256 输入、同一折、同 50 基因、同 log1p 目标。变体：
  A ridge                     参照（官方头）
  B mlp_asrun                 relu, Adam 1e-3, 早停(训练集内随机 10%)    ← 全量跑的那个
  C mlp_identity              同 B 但 activation=identity ⇒ 线性模型用 Adam 训练。若远低于 A ⇒ 优化不足，是我的设置有问题
  D mlp_lr1e-4_patience30     同 B 但 lr 1e-4、耐心 30、max_iter 500
  E mlp_no_target_std         同 B 但目标不标准化
  F mlp_fixed100              同 B 但不早停，固定 100 轮
  G mlp_groupval              早停验证集 = **整片留出的训练样本**（跨样本），而非样本内随机 10%
每个变体同时报训练集 PCC 与测试 PCC：训练 PCC 低 ⇒ 欠拟合；训练高测试低 ⇒ 过拟合。"""
import os, sys, json, glob, copy, warnings
import numpy as np, anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings("ignore")
B = "/path/to/he2st/HEST/eval/bench_data"
EMB = "/path/to/systema4ST/results/hest_emb"
ENCS = sys.argv[1].split(",")
PAIRS = [[c, "0"] for c in sorted(os.listdir(B)) if os.path.isdir(os.path.join(B, c, "adata"))]

def pcc(P, Y):
    P = P - P.mean(0); Y = Y - Y.mean(0)
    num = (P * Y).sum(0); den = np.sqrt((P ** 2).sum(0) * (Y ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return float(np.nanmean(r))

def load(c, sid, enc, genes):
    z = np.load(os.path.join(EMB, f"{sid}_{enc}.npz"), allow_pickle=True)
    X, bc = z["X"], z["bc"].tolist()
    a = ad.read_h5ad(os.path.join(B, c, "adata", f"{sid}.h5ad"))
    pos = {b: i for i, b in enumerate(a.obs_names.astype(str))}
    sub = a[np.array([pos[b] for b in bc])]
    gi = [list(sub.var_names.astype(str)).index(g) for g in genes]
    Y = np.asarray(sub.X.todense() if sparse.issparse(sub.X) else sub.X, np.float64)[:, gi]
    return X.astype(np.float32), np.log1p(Y).astype(np.float32)

def rd(c, f):
    return [l.split(",")[0] for l in open(os.path.join(B, c, "splits", f)).read().splitlines()[1:] if l.strip()]

def mlp(seed=0, **kw):
    base = dict(hidden_layer_sizes=(512,), activation="relu", solver="adam", alpha=1e-4, batch_size=256,
                learning_rate_init=1e-3, max_iter=200, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=10, random_state=seed)
    base.update(kw); return MLPRegressor(**base)

def fit_predict(kind, Ztr, Ytr, Zte_list, groups):
    """返回 (训练集预测, [测试预测...], 备注)"""
    if kind in RIDGE_A:
        # 与官方头完全相同（lsqr、无截距），只改 α。官方 α=100/(D·G)≈0.008，近乎最小二乘。
        r = Ridge(solver="lsqr", alpha=RIDGE_A[kind], random_state=0, fit_intercept=False, max_iter=1000).fit(Ztr, Ytr)
        return r.predict(Ztr), [r.predict(Z) for Z in Zte_list], ""
    if kind == "A":
        r = Ridge(solver="lsqr", alpha=100.0 / (Ztr.shape[1] * Ytr.shape[1]), random_state=0,
                  fit_intercept=False, max_iter=1000).fit(Ztr, Ytr)
        return r.predict(Ztr), [r.predict(Z) for Z in Zte_list], ""
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-6
    std = (kind != "E")
    T = (Ytr - mu) / sd if std else Ytr
    inv = (lambda P: P * sd + mu) if std else (lambda P: P)
    if kind == "B": m = mlp()
    elif kind == "C": m = mlp(activation="identity")
    elif kind == "H1": m = mlp(alpha=1e-2)
    elif kind == "H2": m = mlp(alpha=1e-1)
    elif kind == "H3": m = mlp(alpha=1.0)
    elif kind == "H4": m = mlp(alpha=10.0)
    elif kind == "H5": m = mlp(alpha=3.0)
    elif kind == "H6": m = mlp(alpha=30.0)
    elif kind == "H7": m = mlp(alpha=100.0)
    elif kind == "H8": m = mlp(alpha=300.0)
    elif kind == "H9": m = mlp(alpha=1000.0)
    elif kind == "S3": m = mlp(hidden_layer_sizes=(64,), alpha=10.0)
    elif kind == "S1": m = mlp(hidden_layer_sizes=(64,))
    elif kind == "S2": m = mlp(hidden_layer_sizes=(64,), alpha=1e-1)
    elif kind == "F": m = mlp(early_stopping=False, max_iter=100)
    elif kind == "K":
        # RBF 核岭回归：闭式解、无随机性的非线性头。训练集超过 8000 点时随机子采样（备注标明）。
        from sklearn.kernel_ridge import KernelRidge
        rng = np.random.RandomState(0); idx = np.arange(len(Ztr))
        if len(idx) > 8000: idx = rng.choice(idx, 8000, replace=False)
        kr = KernelRidge(kernel="rbf", alpha=1.0, gamma=1.0 / Ztr.shape[1]).fit(Ztr[idx], T[idx])
        return inv(kr.predict(Ztr)), [inv(kr.predict(Z)) for Z in Zte_list], ("子采样 8000" if len(idx) < len(Ztr) else "")
    elif kind == "G":
        if len(set(groups)) < 3:
            return None, None, "训练样本 <3 片，跳过"
        # 跨样本早停：留出最后一个训练样本整片做验证
        hold = groups[-1]; msk = np.array([g == hold for g in groups])
        Zin, Tin, Zv, Yv = Ztr[~msk], T[~msk], Ztr[msk], Ytr[msk]
        m = mlp(early_stopping=False, max_iter=1); best, bad, keep, ep = -9, 0, None, 0
        for ep in range(200):
            m.partial_fit(Zin, Tin)
            v = pcc(inv(m.predict(Zv)), Yv)
            if v > best + 1e-4: best, bad, keep = v, 0, copy.deepcopy(m)
            else: bad += 1
            if bad >= 10: break
        m = keep; note = f"轮数={ep+1} 留出片验证PCC={best:.3f}"
        return inv(m.predict(Ztr)), [inv(m.predict(Z)) for Z in Zte_list], note
    m.fit(Ztr, T)
    note = f"轮数={m.n_iter_}" + (f" 验证R2={m.best_validation_score_:.3f}" if getattr(m, "early_stopping", False) else "")
    return inv(m.predict(Ztr)), [inv(m.predict(Z)) for Z in Zte_list], note

RIDGE_A = {"R1": 1e-1, "R2": 1.0, "R3": 10.0, "R4": 100.0, "R5": 1e3, "R6": 1e4, "R7": 1e5}
NAMES = {"R1": "ridge_a1e-1", "R2": "ridge_a1", "R3": "ridge_a10", "R4": "ridge_a100", "R5": "ridge_a1e3", "R6": "ridge_a1e4", "R7": "ridge_a1e5", "H8": "mlp_a300", "H9": "mlp_a1000", "A": "ridge", "C": "mlp_identity", "B": "mlp_a1e-4", "H1": "mlp_a1e-2", "H2": "mlp_a1e-1", "H3": "mlp_a1", "H4": "mlp_a10",
         "H5": "mlp_a3", "H6": "mlp_a30", "H7": "mlp_a100", "S3": "mlp_h64_a10", "S1": "mlp_h64", "S2": "mlp_h64_a1e-1", "F": "mlp_fixed100", "G": "mlp_groupval", "K": "krr_rbf"}
ORDER = ["A", "C", "B", "H1", "H2", "H3", "H4", "S1", "S2", "F", "G", "K"]
if len(sys.argv) > 2: ORDER = sys.argv[2].split(",")
AGG = {}
for enc in ENCS:
    for c, k in PAIRS:
        genes = json.load(open(os.path.join(B, c, "var_50genes.json")))["genes"]
        tr, te = rd(c, f"train_{k}.csv"), rd(c, f"test_{k}.csv")
        D = {s: load(c, s, enc, genes) for s in tr + te}
        Xtr = np.concatenate([D[s][0] for s in tr]); Ytr = np.concatenate([D[s][1] for s in tr])
        groups = sum([[s] * len(D[s][0]) for s in tr], [])
        pipe = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=min(256, Xtr.shape[1], Xtr.shape[0] - 1), random_state=0))]).fit(Xtr)
        Ztr = pipe.transform(Xtr); Zte = [pipe.transform(D[s][0]) for s in te]
        print(f"\n═══ {enc}  {c}/fold{k}  训练 {len(tr)} 片 n={len(Ztr)}  测试 {len(te)} 片", flush=True)
        print(f"  {'变体':<14s} {'训练PCC':>8s} {'测试PCC':>8s}   备注")
        for kind in ORDER:
            Ptr, Pte, note = fit_predict(kind, Ztr, Ytr, Zte, groups)
            if Ptr is None:
                print(f"  {NAMES[kind]:<14s} {'—':>8s} {'—':>8s}   {note}", flush=True); continue
            tr_p = pcc(Ptr, Ytr); te_p = float(np.mean([pcc(P, D[s][1]) for P, s in zip(Pte, te)]))
            AGG.setdefault(enc, {}).setdefault(kind, []).append(te_p)
            print(f"  {NAMES[kind]:<14s} {tr_p:8.4f} {te_p:8.4f}   {note}", flush=True)

print("\n══════ 跨队列汇总（fold0，测试 PCC 的队列均值；括号内 = 高于 ridge 的队列数）══════")
for enc in AGG:
    A = AGG[enc]["A"]; print(f"\n{enc}  ridge 均值 {np.mean(A):.4f}  (n 队列 = {len(A)})")
    for kind in ORDER[1:]:
        v = AGG[enc].get(kind, [])
        if not v: continue
        pa = A[:len(v)] if kind != "G" else None
        wins = sum(1 for x, y in zip(v, A) if x > y) if kind != "G" else "n/a"
        print(f"  {NAMES[kind]:<14s} 均值 {np.mean(v):.4f}  Δ vs ridge {np.mean(v)-np.mean(A[:len(v)]) if kind!='G' else float('nan'):+.4f}  高于 ridge {wins}/{len(v)}")

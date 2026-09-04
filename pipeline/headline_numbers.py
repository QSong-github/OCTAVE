# -*- coding: utf-8 -*-
"""从 results/hest_blocks_summary.json + results/encoder_params.json 算出正文/表 1 引用的全部数字。
口径与 mk_tables.py 表 1 一致：K=20、留一队列选 α 的分母（blk_k20_sel）；K=50/200 只报占比中位。"""
import json, os, sys, numpy as np
def _spearman_perm(x, y, nperm=200000, seed=0):
    # 不依赖 scipy：秩相关 + 置换检验（双侧）
    def rank(a):
        o = np.argsort(a); r = np.empty(len(a)); r[o] = np.arange(len(a)); return r
    rx, ry = rank(x), rank(y); rho = np.corrcoef(rx, ry)[0, 1]
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(nperm):
        cnt += abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(rho) - 1e-12
    return rho, (cnt + 1) / (nperm + 1)
R = os.environ.get("S4ST_RESULTS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
S = json.load(open(os.path.join(R, "hest_blocks_summary.json")))
P = json.load(open(os.path.join(R, "encoder_params.json")))
rows = [d for d in S if d["enc"] in P] if "--all" not in sys.argv else S
miss = [d["enc"] for d in S if d["enc"] not in P]
n = len(rows)
def col(key, f): return np.array([d[key][f] for d in rows])
sh = col("blk_k20_sel", "share"); dm = col("blk_k20_sel", "cohort_mean"); t = col("blk_k20_sel", "t"); P_ = col("blk_k20_sel", "P")
pos = int((dm > 0).sum()); t2 = int((np.abs(t) >= 2).sum()); sig = int((P_ < 0.05).sum())
prm = np.array([P[d["enc"]]["params"] for d in rows], float)
try:
    from scipy.stats import spearmanr as _sp; rho, pv = _sp(prm, sh)   # 与论文原口径一致（t 近似）
except Exception:
    rho, pv = _spearman_perm(prm, sh)                                   # 本机 scipy 坏时退回置换检验
small = np.median(sh[prm < 1e8]); big = np.median(sh[prm > 6e8])
print(f"编码器数 n={n}" + (f"（缺参数量、未计入：{' '.join(miss)}）" if miss else ""))
print(f"参数量范围 {prm.min()/1e6:.0f}M – {prm.max()/1e6:.0f}M（{prm.max()/prm.min():.0f}×）；最小 {rows[int(prm.argmin())]['enc']}，最大 {rows[int(prm.argmax())]['enc']}")
print(f"K=20 选α 占比中位 {np.median(sh):.0f}% ({sh.min():.0f}–{sh.max():.0f})；配对差为正 {pos}/{n}；|t|≥2 {t2}/{n}；符号检验 P<0.05 {sig}/{n}")
for k in ("blk_k50_sel", "blk_k200_sel"):
    s2 = col(k, "share"); print(f"{k}: 占比中位 {np.median(s2):.0f}% ({s2.min():.0f}–{s2.max():.0f})")
print(f"Spearman ρ(参数量, 占比) = {rho:.2f}, P = {pv:.4f}；<100M 中位 {small:.0f}% (n={int((prm<1e8).sum())})，>600M 中位 {big:.0f}% (n={int((prm>6e8).sum())})")
neg = [(d["enc"], round(float(d["blk_k20_sel"]["cohort_mean"]), 4), int(d["blk_k20_sel"]["won"])) for d in rows if d["blk_k20_sel"]["cohort_mean"] <= 0 or abs(d["blk_k20_sel"]["t"]) < 2]
print("未达 |t|≥2 或差≤0 的：", neg)
mod = col("blk_k20_sel", "mod"); print(f"模型分范围 {mod.min():.4f}–{mod.max():.4f}（最佳 {rows[int(mod.argmax())]['enc']}，最低 {rows[int(mod.argmin())]['enc']}）")

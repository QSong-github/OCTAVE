# -*- coding: utf-8 -*-
"""归一化旋钮（log1p 官方目标 vs counts-per-10k）在全部有 cp10k 结果的编码器上的统计：
下降幅度、逐编码器/逐队列方向一致性、编码器排序的 Spearman、不一致对数、最大名次变动。"""
import json, glob, os, sys, itertools, numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
enc = sorted(os.path.basename(f)[len("hest_cp10k_"):-5] for f in glob.glob(f"{R}/hest_cp10k_*.json"))
enc = [e for e in enc if os.path.exists(f"{R}/hest_reported_pcc_{e}.json")]
A = {e: json.load(open(f"{R}/hest_reported_pcc_{e}.json")) for e in enc}
B = {e: json.load(open(f"{R}/hest_cp10k_{e}.json"))["per_sample_pcc"] for e in enc}
ids = sorted(set.intersection(*[set(k for k, v in A[e].items() if isinstance(v, dict) and "pcc" in v) & set(B[e]) for e in enc]))
coh = {i: A[enc[0]][i]["cohort"] for i in ids}
a = np.array([[A[e][i]["pcc"] for i in ids] for e in enc]); b = np.array([[B[e][i] for i in ids] for e in enc])
am, bm = a.mean(1), b.mean(1)
drop_enc = 1 - bm / am
print(f"编码器 {len(enc)}，样本 {len(ids)}")
print("下降幅度：逐编码器均值相对下降 中位 %.1f%% 均值 %.1f%%（范围 %.1f–%.1f%%）；总均值相对下降 %.1f%%" % (100*np.median(drop_enc), 100*drop_enc.mean(), 100*drop_enc.min(), 100*drop_enc.max(), 100*(1-bm.mean()/am.mean())))
print("方向：cp10k 更低的编码器 %d/%d" % (int((bm < am).sum()), len(enc)))
cohs = sorted(set(coh.values())); cd = {c: float(np.mean([b[:, j] - a[:, j] for j, i in enumerate(ids) if coh[i] == c])) for c in cohs}
print("队列：cp10k 更低的队列 %d/%d；各队列均差 %s" % (sum(v < 0 for v in cd.values()), len(cohs), {c: round(v, 3) for c, v in cd.items()}))
def rank(x):
    o = np.argsort(-x); r = np.empty(len(x), int); r[o] = np.arange(1, len(x)+1); return r
ra, rb = rank(am), rank(bm)
d = ra - rb; n = len(enc)
rho = 1 - 6 * float((d ** 2).sum()) / (n * (n * n - 1))
disc = sum(1 for i, j in itertools.combinations(range(n), 2) if (am[i] - am[j]) * (bm[i] - bm[j]) < 0)
print("排序：Spearman ρ = %.3f（无并列公式）；不一致对 %d / %d；最大名次变动 %d 位；变动≥1 位的编码器 %d" % (rho, disc, n*(n-1)//2, int(np.abs(d).max()), int((d != 0).sum())))
if "--list" in sys.argv:
    for e, x, y, r1, r2 in sorted(zip(enc, am, bm, ra, rb), key=lambda t: t[3]): print("  %-18s 官方 %.4f cp10k %.4f 名次 %2d→%2d" % (e, x, y, r1, r2))

# -*- coding: utf-8 -*-
"""57 编码器最细带 β1 排名的稳定性：按标本（8 个）自助重采样，重算每个编码器的 β1（标本级中位再取中位），
统计 (1) 每个编码器 β1 的自助区间；(2) 标量排名与最细带排名之间的 231 对反转里有多少在 ≥95% 的重采样中方向不变；
(3) 自助重采样之间最细带排名的 Spearman；(4) DINOv3-H+ 对 KEEP 的 β1 差的自助区间。输入 results/blocks_xen_bands_{enc}/*.json（ridge 的 band_pcc、pcc）。"""
import json, glob, os, numpy as np
def spearmanr(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return (np.corrcoef(ra, rb)[0, 1], None)
R = "results"; M = json.load(open(f"{R}/xen_multi_rank.json")); encs = sorted(M["encoders"])
SPEC = ["Human_Breast_Biomarkers_S1", "Human_Breast_Biomarkers_S2", "Human_Breast_Biomarkers_S3", "Human_Breast_Biomarkers_S4", "Xenium_Prime_Cervical", "Xenium_Prime_Ovarian", "Xenium_V1_Human_Kidney", "Xenium_V1_Human_Ovary"]
sp = lambda n: next(s for s in SPEC if n.startswith(s))
def load(e):
    d = f"{R}/blocks_xen_bands_{'base' if e == 'hibou_l' else e}"; per = {}
    for f in glob.glob(d + "/*.json"):
        j = json.load(open(f)); per.setdefault(sp(j["name"]), []).append((j["pred"]["ridge"]["pcc"], j["pred"]["ridge"]["band_pcc"]))
    return {s: (float(np.median([a for a, b in v])), float(np.median([b for a, b in v]))) for s, v in per.items()}
D = {e: load(e) for e in encs}; assert all(len(D[e]) == 8 for e in encs), {e: len(D[e]) for e in encs if len(D[e]) != 8}
def agg(spec_list):
    pcc = np.array([np.median([D[e][s][0] for s in spec_list]) for e in encs]); b1 = np.array([np.median([D[e][s][1] for s in spec_list]) for e in encs]); return pcc, b1
pcc0, b10 = agg(SPEC)
# 反转对：标量顺序与最细带顺序相反
pairs = [(i, j) for i in range(len(encs)) for j in range(i + 1, len(encs))]
rev0 = [(i, j) for i, j in pairs if (pcc0[i] - pcc0[j]) * (b10[i] - b10[j]) < 0]
rng = np.random.default_rng(0); B = 2000; sign = np.zeros((B, len(rev0))); rk = np.zeros((B, len(encs))); b1s = np.zeros((B, len(encs))); dk = np.zeros(B)
iA, iK = encs.index("dinov3_vith16"), encs.index("keep")
for b in range(B):
    ss = [SPEC[k] for k in rng.integers(0, 8, 8)]; p, q = agg(ss); b1s[b] = q; rk[b] = np.argsort(np.argsort(-q))
    sign[b] = [np.sign(q[i] - q[j]) for i, j in rev0]; dk[b] = q[iK] - q[iA]
sign0 = np.array([np.sign(b10[i] - b10[j]) for i, j in rev0])
stable = (sign == sign0).mean(0); rho = [spearmanr(rk[b], np.argsort(np.argsort(-b10)))[0] for b in range(B)]
ci = np.percentile(b1s, [2.5, 97.5], axis=0); width = ci[1] - ci[0]
out = dict(n_enc=len(encs), n_pairs=len(pairs), n_rev=len(rev0), rev_stable95=int((stable >= 0.95).sum()), rev_stable80=int((stable >= 0.80).sum()), rev_stable_median=float(np.median(stable)),
           spearman_boot_vs_full=dict(med=float(np.median(rho)), lo=float(np.percentile(rho, 2.5)), hi=float(np.percentile(rho, 97.5))),
           b1_ci_width=dict(med=float(np.median(width)), max=float(width.max())), keep_minus_dinov3h=dict(point=float(b10[iK] - b10[iA]), lo=float(np.percentile(dk, 2.5)), hi=float(np.percentile(dk, 97.5)), frac_pos=float((dk > 0).mean())),
           top1_stable=float((rk[:, int(np.argmax(b10))] == 0).mean()), B=B)
json.dump(out, open(f"{R}/beta1_rank_boot.json", "w"), indent=1); print(json.dumps(out, indent=1))

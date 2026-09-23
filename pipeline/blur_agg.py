#!/usr/bin/env python
"""模糊轴的裁决：下游读数的下降轨迹更像 PCC 还是更像 β₁。

对每个区域，把 PCC、β₁、各下游读数都对 t=0 归一，得到「保留比例」曲线。
两个判据：
  ① 轨迹距离：|下游 − PCC| 与 |下游 − β₁| 谁更小（逐 t 求和）
  ② 等效损失：下游掉到 X 时，PCC 说掉了多少、β₁ 说掉了多少
逐区域算 → 样本内中位 → 8 样本精确符号检验。
"""
import glob, json, re
import numpy as np
from math import comb
R = "/path/to/project/results"
KEYS = ["svg_top_jaccard", "svg_rank_rho", "hotspot_jaccard",
        "coloc_preserve", "hotspot_recall_selectivity"]


def spec(n):
    m = re.match(r"(Human_Breast_Biomarkers_S\d)_(Top|Mid|Bot)$", n)
    if "Human_Lung_Cancer_FFPE" in n: return "Lung"   # v1 与 Prime 5K 同一供体同一组织块，按一个标本计
    return m.group(1) if m else n


def signp(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


D = [json.load(open(f)) for f in sorted(glob.glob(R + "/blur_ds_*.json"))]
if not D:
    raise SystemExit("尚无 blur_ds_*.json")
TS = sorted((int(t) for t in D[0]["levels"]), key=int)
print("区域 %d 个 / 样本 %d 个；模糊档 %s\n" % (len(D), len({spec(d["name"]) for d in D}), TS))

# 先看轴本身有没有区分力：PCC 与 β₁ 的下降幅度是否明显不同
pk, bk = [], []
for d in D:
    L = d["levels"]
    pk.append(L[str(TS[-1])]["pcc"] / L["0"]["pcc"])
    bk.append(L[str(TS[-1])]["beta1"] / L["0"]["beta1"])
print("t=0 → t=%d 的保留比例：PCC %.3f [%.3f, %.3f]   β₁ %.3f [%.3f, %.3f]"
      % (TS[-1], np.median(pk), min(pk), max(pk), np.median(bk), min(bk), max(bk)))
if np.median(pk) - np.median(bk) < 0.05:
    print("⚠ 两者下降幅度接近，该轴区分力不足\n")
else:
    print("⇒ 两把尺子在这条轴上明显分离，检验有区分力\n")

print("%-28s %10s %10s %10s   %s" % ("下游读数", "末档保留", "更像PCC", "更像β₁", "样本层级符号检验"))
res = {}
for k in KEYS:
    win_p, win_b, keep = {}, {}, {}
    for d in D:
        L = d["levels"]
        if L["0"].get(k) is None or abs(L["0"][k]) < 1e-9:
            continue
        base = L["0"][k]
        ds = np.array([L[str(t)][k] / base for t in TS], float)
        pc = np.array([L[str(t)]["pcc"] / L["0"]["pcc"] for t in TS], float)
        b1 = np.array([L[str(t)]["beta1"] / L["0"]["beta1"] for t in TS], float)
        if not np.isfinite(ds).all():
            continue
        s = spec(d["name"])
        keep.setdefault(s, []).append(ds[-1])
        win_p.setdefault(s, []).append(float(np.abs(ds - pc).sum()))
        win_b.setdefault(s, []).append(float(np.abs(ds - b1).sum()))
    S = sorted(set(win_p) & set(win_b))
    if len(S) < 5:
        print("%-28s （有效样本 %d 个）" % (k, len(S))); continue
    dp = np.array([np.median(win_p[s]) for s in S])
    db = np.array([np.median(win_b[s]) for s in S])
    kb = int((db < dp).sum())          # β₁ 更贴近的样本数
    P = signp(kb, len(S))
    res[k] = dict(n=len(S), keep_last=float(np.median([np.median(keep[s]) for s in S])),
                  dist_pcc=float(np.median(dp)), dist_beta=float(np.median(db)),
                  n_beta_closer=kb, P=P)
    print("%-28s %10.3f %10.3f %10.3f   β₁更近 %d/%d  P=%.4f"
          % (k, res[k]["keep_last"], np.median(dp), np.median(db), kb, len(S), P))

if res:
    nb = sum(1 for v in res.values() if v["n_beta_closer"] > v["n"] / 2 and v["P"] < 0.05)
    np_ = sum(1 for v in res.values() if v["n_beta_closer"] < v["n"] / 2 and v["P"] < 0.05)
    print("\n显著更像 β₁ 的读数: %d/%d；显著更像 PCC 的: %d/%d" % (nb, len(res), np_, len(res)))
    json.dump(res, open(R + "/blur_verdict.json", "w"), indent=1)
    print("-> %s/blur_verdict.json" % R)

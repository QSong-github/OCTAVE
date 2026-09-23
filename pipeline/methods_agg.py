# -*- coding: utf-8 -*-
"""轨道 1：把上游项目已测的已发表方法放进本文表 2 的框架。

口径必须与表 2 一致：报告分数用逐样本均值；队列统计量用「队列内逐样本配对差的
中位数」，与符号检验同源。方法与编码器共用同一批 72 个样本、同一 HEST 划分。
"""
import glob, json, os
import numpy as np
from math import comb
R = "/path/to/project/results"


def signp(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load_flat(pat):
    """把 {sample: {cohort, pcc}} 或按队列分片的一组文件合成一个字典。"""
    out = {}
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f))
        for s, v in d.items():
            if isinstance(v, dict) and "pcc" in v:
                out[s] = v
    return out


M = {"HisToGene": load_flat(R + "/histogene_matched.json"),
     "Hist2ST": load_flat(R + "/h2st2_matched_*.json"),
     "BLEEP": load_flat(R + "/bleep_hest.json")}
# 参照：15 个编码器的 ridge 分数，与各自 k=50 的匹配检索地板
ENC = sorted(os.path.basename(p)[len("hest_floor_"):] for p in glob.glob(R + "/hest_floor_*")
             if os.path.isdir(p) and not os.path.basename(p)[len("hest_floor_"):][0].isdigit()
             and "_k10_" not in p and "_k200_" not in p and "_k800_" not in p)
import re
ENC = [e for e in ENC if not re.match(r"^k\d+_", e)]
E = {}
for e in ENC:
    ps = R + "/hest_effres_ps_%s.json" % e
    fs = sorted(glob.glob(R + "/hest_floor_%s/*.json" % e))
    if not os.path.exists(ps) or len(fs) != 10: continue
    mod = json.load(open(ps))["per_sample_pcc"]
    fl, coh = {}, {}
    for f in fs:
        d = json.load(open(f))
        for s, v in d["samples"].items():
            fl[s] = v["pcc"]; coh[s] = d["cohort"]
    E[e] = dict(model=mod, floor=fl, cohort=coh)
print("编码器 %d 个；方法样本数 %s" % (len(E), {k: len(v) for k, v in M.items()}))
COH = list(E.values())[0]["cohort"]
enc_mean = {e: float(np.mean(list(v["model"].values()))) for e, v in E.items()}
flo_mean = {e: float(np.mean(list(v["floor"].values()))) for e, v in E.items()}
print("\n编码器 ridge 逐样本均值 %.4f–%.4f；匹配检索地板 %.4f–%.4f"
      % (min(enc_mean.values()), max(enc_mean.values()), min(flo_mean.values()), max(flo_mean.values())))

OUT = {}
print("\n%-11s %5s %9s %14s %14s %s" % ("方法", "n", "逐样本均值", "低于几个编码器", "低于几个地板", "队列层级 vs 最佳编码器"))
best = max(enc_mean, key=enc_mean.get)
for name, d in M.items():
    if not d: print("%-11s （无数据）" % name); continue
    ids = sorted(set(d) & set(E[best]["model"]))
    mv = float(np.mean([d[s]["pcc"] for s in ids]))
    n_below_enc = sum(1 for e in enc_mean if mv < enc_mean[e])
    n_below_flo = sum(1 for e in flo_mean if mv < flo_mean[e])
    # 与最佳编码器的队列层级配对比较
    dif = {s: d[s]["pcc"] - E[best]["model"][s] for s in ids}
    by = {}
    for s in ids: by.setdefault(COH.get(s, d[s].get("cohort", "?")), []).append(dif[s])
    med = {c: float(np.median(v)) for c, v in by.items()}
    won = sum(1 for v in med.values() if v > 0)
    g = np.array(list(med.values())); sem = float(g.std(ddof=1) / np.sqrt(len(g)))
    OUT[name] = dict(n_samples=len(ids), mean_sample=mv, n_below_encoders=n_below_enc,
                     n_encoders=len(enc_mean), n_below_floors=n_below_flo, n_floors=len(flo_mean),
                     vs_best_encoder=best, cohort_mean=float(g.mean()), cohort_sem=sem,
                     cohorts_won=won, n_cohorts=len(med), P=signp(won, len(med)),
                     by_cohort={c: float(np.median(v)) for c, v in by.items()})
    print("%-11s %5d %9.4f %14s %14s   %+.4f ± %.4f，胜 %d/%d，P=%.4f"
          % (name, len(ids), mv, "%d/%d" % (n_below_enc, len(enc_mean)),
             "%d/%d" % (n_below_flo, len(flo_mean)), g.mean(), sem, won, len(med), signp(won, len(med))))
OUT["_reference"] = dict(encoder_mean=enc_mean, floor_mean=flo_mean, best_encoder=best)
json.dump(OUT, open(R + "/methods_vs_floor.json", "w"), indent=1)
print("\n-> %s/methods_vs_floor.json" % R)

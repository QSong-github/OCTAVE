# -*- coding: utf-8 -*-
"""五个已发表方法放进本文表 2 的框架。口径与表 2 一致：
报告分数用逐样本均值；队列统计量用队列内逐样本配对差的中位数，与符号检验同源。"""
import glob, json, os, re
import numpy as np
from math import comb
_here = os.path.dirname(os.path.abspath(__file__))
for R in (os.environ.get("OCTAVE_RESULTS"), "/path/to/systema4ST/results",
          os.path.join(_here, "results"), os.path.join(os.path.dirname(_here), "results")):
    if R and os.path.isdir(R):
        break


def signp(k, n):
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load_flat(pat):
    out = {}
    for f in sorted(glob.glob(pat)):
        for s, v in json.load(open(f)).items():
            if isinstance(v, dict) and "pcc" in v:
                out[s] = v
    return out


M = {"HisToGene": load_flat(R + "/histogene_matched.json"),
     "Hist2ST":   load_flat(R + "/h2st2_matched_*.json"),
     "BLEEP":     load_flat(R + "/bleep_hest.json"),
     "HECLIP":    load_flat(R + "/heclip_hest.json"),
     "HGGEP":     load_flat(R + "/hggep_hest.json")}
import os as _os
if _os.path.exists(R + "/thitogene_hest.json"):          # 2026-09-03 扩集；折文件合并后才存在
    M["THItoGene"] = load_flat(R + "/thitogene_hest.json")

# 2026-09-14 扩集：DeepSpot（Nonchev et al. 2025）。作者未指定默认配置（README 并列 UNI / H-optimus-0 / Phikon；
# notebook 训 10 epoch，论文设置反推约 500 步），故跑了两个基座 × 两种训练长度。
# 2026-09-18 起每个配置单独成行，与其它六个方法同口径（各一次运行）；逐样本取四者最好的"信封"只写进正文一句，
# 键 DeepSpot:best_of_4，制表脚本跳过它。
DS_LABEL = {"hoptimus0": "DeepSpot, H-optimus-0, 10 epochs", "hoptimus0_steps500": "DeepSpot, H-optimus-0, 500 steps",
            "uni_v1": "DeepSpot, UNI, 10 epochs", "uni_v1_steps500": "DeepSpot, UNI, 500 steps"}
_ds = {}
for _f in sorted(glob.glob(R + "/deepspot_hest_*.json")):
    _tag = os.path.basename(_f)[len("deepspot_hest_"):-len(".json")]
    M["DeepSpot:" + _tag] = load_flat(_f)
    for _s, _v in json.load(open(_f)).items():
        if isinstance(_v, dict) and "pcc" in _v and (_s not in _ds or _v["pcc"] > _ds[_s]["pcc"]):
            _ds[_s] = _v
if _ds:
    M["DeepSpot:best_of_4"] = _ds

# 2026-09-03：参照编码器 = 所有有留一队列选 α（LOCO）ridge 结果的编码器，与表 1 的分母同口径（不再限于有地板目录的 30 个）
ENC = [os.path.basename(p)[len("hest_rsel_ps_"):-len(".json")] for p in glob.glob(R + "/hest_rsel_ps_*.json")]
ENC = sorted(e for e in ENC if not re.match(r"^k\d+_", e))
ENC = [e for e in ENC if e != "omiclip_raw"]   # omiclip 的未归一化变体，同一模型，不占两行
E = {}
for e in ENC:
    ps = R + "/hest_rsel_ps_%s.json" % e
    mod = json.load(open(ps))["per_sample_pcc"]
    fs = sorted(glob.glob(R + "/hest_floor_%s/*.json" % e))
    fl, coh = {}, {}
    if len(fs) == 10:                      # 匹配检索地板只有原 30 个有；正文已不用，仅留档
        for f in fs:
            d = json.load(open(f))
            for s, v in d["samples"].items():
                fl[s] = v["pcc"]; coh[s] = d["cohort"]
    E[e] = dict(model=mod, floor=fl, cohort=coh)
COH = next(v["cohort"] for v in E.values() if v["cohort"])
enc_mean = {e: float(np.mean(list(v["model"].values()))) for e, v in E.items()}
flo_mean = {e: float(np.mean(list(v["floor"].values()))) for e, v in E.items() if v["floor"]}
best = max(enc_mean, key=enc_mean.get)
worst_floor = min(flo_mean, key=flo_mean.get)
print("参照：%d 个编码器 ridge 逐样本均值 %.4f–%.4f；匹配检索地板 %.4f–%.4f"
      % (len(E), min(enc_mean.values()), max(enc_mean.values()),
         min(flo_mean.values()), max(flo_mean.values())))
print("      最佳编码器 %s (%.4f)；最低地板 %s (%.4f)\n"
      % (best, enc_mean[best], worst_floor, flo_mean[worst_floor]))

OUT = {}
print("%-11s %4s %9s %14s %14s   %-26s %s"
      % ("方法", "n", "逐样本均值", "低于编码器", "低于地板", "vs 最佳编码器(队列层级)", "P"))
for name, d in M.items():
    if not d:
        print("%-11s （无结果）" % name); continue
    ids = sorted(set(d) & set(E[best]["model"]))
    mv = float(np.mean([d[s]["pcc"] for s in ids]))
    nbe = sum(1 for e in enc_mean if mv < enc_mean[e])
    nbf = sum(1 for e in flo_mean if mv < flo_mean[e])
    dif = {s: d[s]["pcc"] - E[best]["model"][s] for s in ids}
    by = {}
    for s in ids:
        by.setdefault(COH.get(s, d[s].get("cohort", "?")), []).append(dif[s])
    med = np.array([np.median(v) for v in by.values()])
    won = int((med > 0).sum()); P = signp(won, len(med))
    sem = float(med.std(ddof=1) / np.sqrt(len(med)))
    OUT[name] = dict(n_samples=len(ids), mean_sample=mv,
                     n_below_encoders=nbe, n_encoders=len(enc_mean),
                     n_below_floors=nbf, n_floors=len(flo_mean),
                     vs_best_encoder=best, cohort_mean=float(med.mean()), cohort_sem=sem,
                     cohorts_won=won, n_cohorts=len(med), P=P)
    print("%-11s %4d %9.4f %14s %14s   %+.4f ± %.4f, 胜 %d/%d   %.4f"
          % (name, len(ids), mv, "%d/%d" % (nbe, len(enc_mean)),
             "%d/%d" % (nbf, len(flo_mean)), med.mean(), sem, won, len(med), P))
OUT["_reference"] = dict(encoder_mean=enc_mean, floor_mean=flo_mean, best_encoder=best)
json.dump(OUT, open(R + "/methods_vs_floor.json", "w"), indent=1)
ok = [k for k in OUT if k != "_reference"]
allb = all(OUT[k]["n_below_floors"] == OUT[k]["n_floors"] for k in ok)
print("\n%d 个方法全部低于全部 %d 条检索地板: %s" % (len(ok), len(flo_mean), allb))
if not allb:
    print("  例外:", {k: "%d/%d" % (OUT[k]["n_below_floors"], OUT[k]["n_floors"])
                     for k in ok if OUT[k]["n_below_floors"] != OUT[k]["n_floors"]})
print("-> %s/methods_vs_floor.json" % R)

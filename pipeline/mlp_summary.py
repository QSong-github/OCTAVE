import json, glob, sys, numpy as np
tag = sys.argv[1]                      # mlp | mlp10
K = json.load(open("results/k_sensitivity.json"))["50"]; KM = json.load(open("results/k_sensitivity_%s.json" % tag))["50"]
C = json.load(open("results/cohort_spread.json"));   CM = json.load(open("results/cohort_spread_%s.json" % tag))
rel = lambda d, e: (d[e]["rel_ok"] if d[e].get("rel_ok") is not None else d[e]["rel"])
P = lambda d, e: (d[e].get("P_ok") if d[e].get("P_ok") is not None else d[e]["P"])
rows = []
for e in sorted(KM):
    ms = [json.load(open(f))["mean_pcc"] for f in sorted(glob.glob("results/mlp_seeds/hest_%s_ps_%s_s*.json" % (tag, e)))]
    rows.append((e, K[e]["floor"] + K[e]["gap"], KM[e]["floor"] + KM[e]["gap"], np.std(ms, ddof=1) if len(ms) > 1 else float("nan"), len(ms),
                 KM[e]["floor"], rel(KM, e), CM[e]["cohort_mean"], CM[e]["cohort_sem"], abs(CM[e]["t"]), CM[e]["cohorts_won"], abs(C[e]["t"]), rel(K, e)))
rows.sort(key=lambda r: -r[2])
print("══ %s：30 编码器（seed 0 进表；种子 sd 来自全部种子的逐样本均值 PCC）══" % tag)
print("%-14s %8s %8s %7s %3s %8s %8s %8s %16s %6s %6s %8s" % ("编码器", "ridge", tag, "种子sd", "ns", "地板", "余量%ridge", "余量%" + tag, "队列均值±sem", "|t|", "赢", "|t|ridge"))
for r in rows:
    print("%-14s %8.4f %8.4f %7.4f %3d %8.4f %+8.1f %+8.1f %+8.4f±%.4f %6.2f %3d/10 %8.2f" % (r[0], r[1], r[2], r[3], r[4], r[5], r[12], r[6], r[7], r[8], r[9], r[10], r[11]))
d = [r[2] - r[1] for r in rows]
print("\n%s−ridge 逐编码器：中位 %+.4f，范围 %+.4f … %+.4f，为正的 %d/%d" % (tag, np.median(d), min(d), max(d), sum(1 for x in d if x > 0), len(d)))
print("%s 线：低于自身地板 %d/%d；|t|≥2 的 %d/%d（其中余量为正的 %d）；队列层级显著 %d/%d %s" % (
    tag, sum(1 for r in rows if r[6] < 0), len(rows),
    sum(1 for r in rows if r[9] >= 2), len(rows), sum(1 for r in rows if r[9] >= 2 and r[7] > 0),
    sum(1 for e in KM if P(KM, e) < 0.05), len(KM), [e for e in KM if P(KM, e) < 0.05]))

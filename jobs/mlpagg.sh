#!/bin/bash
#SBATCH -J mlpagg
#SBATCH --qos=YOUR_QOS --partition=hpg-default -c 2 --mem=16G -t 1:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
python3 -u k_sens_mlp.py | tail -45
python3 -u cohort_spread_mlp.py | tail -6
python3 - <<'PY'
import json, glob, numpy as np
print("\n══ 种子离散度（3 种子的逐样本均值 PCC）══")
K=json.load(open("results/k_sensitivity.json"))["50"]; KM=json.load(open("results/k_sensitivity_mlp.json"))["50"]
C=json.load(open("results/cohort_spread.json")); CM=json.load(open("results/cohort_spread_mlp.json"))
rows=[]
for e in sorted(KM):
    ms=[json.load(open(f))["mean_pcc"] for f in sorted(glob.glob("results/mlp_seeds/hest_mlp_ps_%s_s*.json"%e))]
    rows.append((e, K[e]["floor"]+K[e]["gap"], KM[e]["floor"]+KM[e]["gap"], np.std(ms,ddof=1), KM[e]["floor"],
                 (KM[e]["rel_ok"] if KM[e].get("rel_ok") is not None else KM[e]["rel"]), CM[e]["cohort_mean"], CM[e]["cohort_sem"], abs(CM[e]["t"]), CM[e]["cohorts_won"], abs(C[e]["t"])))
rows.sort(key=lambda r:-r[2])
print("%-14s %8s %8s %7s %8s %7s %16s %6s %6s %s"%("编码器","ridge","mlp","种子sd","地板","余量%","队列均值±sem","|t|mlp","赢","|t|ridge"))
for r in rows: print("%-14s %8.4f %8.4f %7.4f %8.4f %+7.1f %+8.4f±%.4f %6.2f %3d/10 %6.2f"%r)
print("\nMLP 线：低于自身地板 %d/%d；|t|≥2 的 %d/%d %s；队列层级显著(P<0.05) %d/%d %s"%(
  sum(1 for r in rows if r[5]<0),len(rows),
  sum(1 for r in rows if r[8]>=2),len(rows),[r[0] for r in rows if r[8]>=2],
  sum(1 for e in KM if (KM[e].get("P_ok") if KM[e].get("P_ok") is not None else KM[e]["P"])<0.05),len(KM),
  [e for e in KM if (KM[e].get("P_ok") if KM[e].get("P_ok") is not None else KM[e]["P"])<0.05]))
d=[r[2]-r[1] for r in rows]; print("MLP−ridge 逐编码器：中位 %+.4f，范围 %+.4f … %+.4f，为正的 %d/%d"%(np.median(d),min(d),max(d),sum(1 for x in d if x>0),len(d)))
PY

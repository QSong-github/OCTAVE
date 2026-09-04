#!/bin/bash
cd /blue/qsong1/wang.qing/systema4ST
say(){ echo "[$(date +%H:%M:%S)] $*"; }
python3 add_mlp.py
mkdir -p results/mlp_seeds
ENC=$(ls results/hest_effres_ps_*.json | sed 's#.*/hest_effres_ps_##; s#\.json##' | grep -v '^omiclip_raw$' | tr '\n' ' ')
NE=$(echo $ENC | wc -w); echo "编码器 $NE 个: $ENC"
[ "$NE" = 30 ] || { echo "编码器数不是 30"; exit 1; }

say "冒烟：ciga seed0"
cat > jobs/mlpsmoke.sh <<'SH'
#!/bin/bash
#SBATCH -J mlpsmoke
#SBATCH --qos=qsong1 --partition=hpg-default -c 2 --mem=16G -t 2:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export OMP_NUM_THREADS=2
python -u src/hest_effres_ps.py --encoder ciga --skip_sigma --head mlp --seed 0 --out results/mlp_seeds/hest_mlp_ps_ciga_s0.json
python - <<'PY'
import json
m=json.load(open("results/mlp_seeds/hest_mlp_ps_ciga_s0.json")); r=json.load(open("results/hest_effres_ps_ciga.json"))
print("冒烟对照 ciga: mlp 均值 %.4f (n=%d)  ridge 均值 %.4f" % (m["mean_pcc"], len(m["per_sample_pcc"]), r["mean_pcc"]))
assert len(m["per_sample_pcc"])==72
PY
SH
J=$(sbatch --parsable jobs/mlpsmoke.sh)
while [ "$(squeue -j $J -h | wc -l)" -gt 0 ]; do sleep 30; done
grep -E "冒烟对照|mlp 训练" logs/mlpsmoke_${J}.out | tail -4
[ -s results/mlp_seeds/hest_mlp_ps_ciga_s0.json ] || { tail -20 logs/mlpsmoke_${J}.out; exit 1; }

say "全量：30 编码器 × 3 种子"
cat > jobs/mlpall.sh <<SH
#!/bin/bash
#SBATCH -J mlpall
#SBATCH --qos=qsong1 --partition=hpg-default --array=0-89 -c 2 --mem=16G -t 4:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export OMP_NUM_THREADS=2
E=($ENC); I=\$SLURM_ARRAY_TASK_ID; N=\${E[\$((I/3))]}; S=\$((I%3))
O=results/mlp_seeds/hest_mlp_ps_\${N}_s\${S}.json
[ -s "\$O" ] && exit 0
python -u src/hest_effres_ps.py --encoder "\$N" --skip_sigma --head mlp --seed \$S --out "\$O"
SH
sbatch jobs/mlpall.sh >/dev/null
while [ "$(squeue -u wang.qing -r -h -n mlpall | wc -l)" -gt 0 ]; do sleep 60; done
echo "  产出 $(ls results/mlp_seeds/*.json | wc -l)/90"
[ "$(ls results/mlp_seeds/*.json | wc -l)" -ge 90 ] || { grep -l -i "error\|Traceback" logs/mlpall_*.out | head -3 | xargs -I{} tail -5 {}; exit 1; }

say "汇总：seed0 进 MLP 版 k_sens / cohort_spread，其余种子只算离散度"
for e in $ENC; do cp results/mlp_seeds/hest_mlp_ps_${e}_s0.json results/hest_mlp_ps_${e}.json; done
sed -e 's#results/hest_effres_ps_#results/hest_mlp_ps_#g' -e 's#results/k_sensitivity.json#results/k_sensitivity_mlp.json#' k_sens.py > k_sens_mlp.py
sed -e 's#hest_effres_ps_#hest_mlp_ps_#g' -e 's#/k_sensitivity.json#/k_sensitivity_mlp.json#' -e 's#/cohort_spread.json#/cohort_spread_mlp.json#' cohort_spread.py > cohort_spread_mlp.py
grep -c "hest_mlp_ps_" k_sens_mlp.py cohort_spread_mlp.py
cat > jobs/mlpagg.sh <<'SH'
#!/bin/bash
#SBATCH -J mlpagg
#SBATCH --qos=qsong1 --partition=hpg-default -c 2 --mem=16G -t 1:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
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
SH
J=$(sbatch --parsable jobs/mlpagg.sh)
while [ "$(squeue -j $J -h | wc -l)" -gt 0 ]; do sleep 20; done
cat logs/mlpagg_${J}.out

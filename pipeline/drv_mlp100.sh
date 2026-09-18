#!/bin/bash
cd /path/to/systema4ST
say(){ echo "[$(date +%H:%M:%S)] $*"; }
python3 add_alpha.py >/dev/null
ENC=$(ls results/hest_effres_ps_*.json | sed 's#.*/hest_effres_ps_##; s#\.json##' | grep -v '^omiclip_raw$' | tr '\n' ' ')
[ "$(echo $ENC | wc -w)" = 30 ] || { echo "编码器数不是 30"; exit 1; }
say "全量：30 编码器 × 3 种子，α=100"
cat > jobs/mlp100.sh <<SH
#!/bin/bash
#SBATCH -J mlp100
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-89 -c 2 --mem=16G -t 4:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST; export OMP_NUM_THREADS=2
E=($ENC); I=\$SLURM_ARRAY_TASK_ID; N=\${E[\$((I/3))]}; S=\$((I%3))
O=results/mlp_seeds/hest_mlp100_ps_\${N}_s\${S}.json
[ -s "\$O" ] && exit 0
python -u src/hest_effres_ps.py --encoder "\$N" --skip_sigma --head mlp --mlp_alpha 100 --seed \$S --out "\$O"
SH
sbatch jobs/mlp100.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n mlp100 | wc -l)" -gt 0 ]; do sleep 60; done
echo "  产出 $(ls results/mlp_seeds/hest_mlp100_ps_*.json | wc -l)/90"
[ "$(ls results/mlp_seeds/hest_mlp100_ps_*.json | wc -l)" -ge 90 ] || { grep -l "Traceback" logs/mlp100_*.out | head -2 | xargs -I{} tail -5 {}; exit 1; }
for e in $ENC; do cp results/mlp_seeds/hest_mlp100_ps_${e}_s0.json results/hest_mlp100_ps_${e}.json; done
sed -e 's#results/hest_effres_ps_#results/hest_mlp100_ps_#g' -e 's#results/k_sensitivity.json#results/k_sensitivity_mlp100.json#' k_sens.py > k_sens_mlp100.py
sed -e 's#hest_effres_ps_#hest_mlp100_ps_#g' -e 's#/k_sensitivity.json#/k_sensitivity_mlp100.json#' -e 's#/cohort_spread.json#/cohort_spread_mlp100.json#' cohort_spread.py > cohort_spread_mlp100.py
cat > jobs/mlp100agg.sh <<'SH'
#!/bin/bash
#SBATCH -J mlp100agg
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=16G -t 1:00:00
#SBATCH -o /path/to/systema4ST/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/systema4ST
python3 -u k_sens_mlp100.py | grep -E "^\s+k=|跨度|移动|超参|翻转" 
python3 -u cohort_spread_mlp100.py | tail -3
python3 mlp_summary.py mlp100
echo; echo "──── 对照：首轮 α=1e-4 ────"; python3 mlp_summary.py mlp | tail -3
SH
J=$(sbatch --parsable jobs/mlp100agg.sh)
while [ "$(squeue -j $J -h | wc -l)" -gt 0 ]; do sleep 20; done
cat logs/mlp100agg_${J}.out

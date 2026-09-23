#!/bin/bash
cd /path/to/project
say(){ echo "[$(date +%H:%M:%S)] $*"; }
python3 add_ralpha.py; mkdir -p results/ridge_alpha
ENC=$(ls results/hest_effres_ps_*.json | sed 's#.*/hest_effres_ps_##; s#\.json##' | grep -v '^omiclip_raw$' | tr '\n' ' ')
[ "$(echo $ENC | wc -w)" = 30 ] || { echo "编码器数不是 30"; exit 1; }
say "岭回归 α 网格：30 编码器 × 7 α"
cat > jobs/ralpha.sh <<SH
#!/bin/bash
#SBATCH -J ralpha
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION --array=0-209 -c 2 --mem=16G -t 2:00:00
#SBATCH -o /path/to/project/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project; export OMP_NUM_THREADS=2
E=($ENC); A=(0.1 1 10 100 1000 10000 100000); I=\$SLURM_ARRAY_TASK_ID; N=\${E[\$((I/7))]}; AL=\${A[\$((I%7))]}
O=results/ridge_alpha/hest_ra_\${N}_a\${AL}.json
[ -s "\$O" ] && exit 0
python -u src/hest_effres_ps.py --encoder "\$N" --skip_sigma --ridge_alpha \$AL --out "\$O"
SH
sbatch jobs/ralpha.sh >/dev/null
while [ "$(squeue -u $USER -r -h -n ralpha | wc -l)" -gt 0 ]; do sleep 45; done
echo "  产出 $(ls results/ridge_alpha/*.json | wc -l)/210"
[ "$(ls results/ridge_alpha/*.json | wc -l)" -ge 210 ] || { grep -l Traceback logs/ralpha_*.out | head -2 | xargs -I{} tail -5 {}; exit 1; }
say "留一队列选 α 并汇总"
sed -e 's#results/hest_effres_ps_#results/hest_rsel_ps_#g' -e 's#results/k_sensitivity.json#results/k_sensitivity_rsel.json#' k_sens.py > k_sens_rsel.py
sed -e 's#hest_effres_ps_#hest_rsel_ps_#g' -e 's#/k_sensitivity.json#/k_sensitivity_rsel.json#' -e 's#/cohort_spread.json#/cohort_spread_rsel.json#' cohort_spread.py > cohort_spread_rsel.py
cat > jobs/rselagg.sh <<SH
#!/bin/bash
#SBATCH -J rselagg
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION -c 2 --mem=16G -t 1:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project
python3 ridge_loco.py $(echo $ENC | tr ' ' ',')
python3 -u k_sens_rsel.py | grep -E "^\s+k=|跨度|移动|超参|翻转"
python3 -u cohort_spread_rsel.py | tail -3
python3 mlp_summary.py rsel
SH
J=$(sbatch --parsable jobs/rselagg.sh)
while [ "$(squeue -j $J -h | wc -l)" -gt 0 ]; do sleep 20; done
cat logs/rselagg_${J}.out

#!/bin/bash
#SBATCH -J waivds
#SBATCH --qos=qsong1 --partition=hpg-default
#SBATCH --array=0-81 -c 2 --mem=12G -t 8:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export OMP_NUM_THREADS=2
E=(phaet mascaret); COH=(CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM); KS=(10 50 200 800)
I=$SLURM_ARRAY_TASK_ID
if [ "$I" -ge 80 ]; then
  N=${E[$((I-80))]}
  [ -s "results/hest_effres_ps_${N}.json" ] && exit 0
  exec python -u src/hest_effres_ps.py --encoder "$N" --out "results/hest_effres_ps_${N}.json"
fi
N=${E[$((I/40))]}; J=$((I%40)); K=${KS[$((J/10))]}; C=${COH[$((J%10))]}
if [ "$K" = "50" ]; then OUT=results/hest_floor_${N}; else OUT=results/hest_floor_k${K}_${N}; fi
mkdir -p $OUT; [ -s "$OUT/$C.json" ] && exit 0
python -u src/hest_floor.py --encoder "$N" --cohort "$C" --k "$K" --out "$OUT/$C.json"

#!/bin/bash
#SBATCH -J muskfloor
#SBATCH --qos=qsong1 --partition=hpg-default
#SBATCH --array=0-39 -c 2 --mem=12G -t 8:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
COH=(CCRCC COAD HCC IDC LUNG LYMPH_IDC PAAD PRAD READ SKCM); KS=(10 50 200 800)
I=$SLURM_ARRAY_TASK_ID; K=${KS[$((I/10))]}; C=${COH[$((I%10))]}
if [ "$K" = "50" ]; then OUT=results/hest_floor_musk; else OUT=results/hest_floor_k${K}_musk; fi
mkdir -p $OUT; [ -s "$OUT/$C.json" ] && exit 0
python -u src/hest_floor.py --encoder musk --cohort "$C" --k "$K" --out "$OUT/$C.json"

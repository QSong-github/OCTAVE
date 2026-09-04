#!/bin/bash
#SBATCH -J muskps
#SBATCH --qos=qsong1 --partition=hpg-default
#SBATCH -c 4 --mem=48G -t 8:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -eu
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
python -u src/hest_effres_ps.py --encoder musk --out results/hest_effres_ps_musk.json

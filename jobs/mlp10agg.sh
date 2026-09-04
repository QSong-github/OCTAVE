#!/bin/bash
#SBATCH -J mlp10agg
#SBATCH --qos=qsong1 --partition=hpg-default -c 2 --mem=16G -t 1:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
python3 -u k_sens_mlp10.py | grep -E "^\s+k=|跨度|移动|超参|翻转" 
python3 -u cohort_spread_mlp10.py | tail -3
python3 mlp_summary.py mlp10
echo; echo "──── 对照：首轮 α=1e-4 ────"; python3 mlp_summary.py mlp | tail -3

#!/bin/bash
#SBATCH --job-name=panelsweep
#SBATCH --qos=YOUR_QOS
#SBATCH --array=0-3
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%A_%a.out
# 基因面板敏感性: top-20/50/100/200 HVG。审稿人必问的一条 ——
# "110µm 是不是只在 top-50 HVG 这个特定面板上成立?"
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=12
HVGS=(20 50 100 200)
H=${HVGS[$SLURM_ARRAY_TASK_ID]}
python -u src/tower_sweep.py --tower hibou_l --hvg $H --outdir results/panel_sweep_hvg$H

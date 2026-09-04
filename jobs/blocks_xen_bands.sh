#!/bin/bash
#SBATCH --job-name=blkband
#SBATCH --qos=qsong1
#SBATCH --partition=hpg-default
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%A_%a.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
mapfile -t N < <(ls results/xenium/*.json | xargs -n1 basename | sed 's/.json//')
echo "节点 $(hostname)  区域 ${N[$SLURM_ARRAY_TASK_ID]}"
python -u src/blocks_xen_bands.py --name "${N[$SLURM_ARRAY_TASK_ID]}"

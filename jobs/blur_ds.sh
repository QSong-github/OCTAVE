#!/bin/bash
#SBATCH --job-name=blurds
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/path/to/project/logs/%x_%A_%a.out
# 第四步的决定性检验：同一批 bin、同一真值下，已训模型 vs 块预言机的
# 六个下游读数落差，与标量落差(约22%)、最细带落差(约80%)比对。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1 OMP_NUM_THREADS=8
mapfile -t N < <(ls results/blocks_xen_bands/*.json | xargs -n1 basename | sed 's/\.json$//')
echo "节点 $(hostname)  区域 ${N[$SLURM_ARRAY_TASK_ID]}"
python -u src/blur_downstream.py --name "${N[$SLURM_ARRAY_TASK_ID]}"

#!/bin/bash
#SBATCH --job-name=blkxen
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%A_%a.out
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
mapfile -t N < <(ls results/xenium/*.json | xargs -n1 basename | sed 's/.json//')
echo "节点 $(hostname)  区域 ${N[$SLURM_ARRAY_TASK_ID]}"
python -u src/blocks_xen.py --name "${N[$SLURM_ARRAY_TASK_ID]}"

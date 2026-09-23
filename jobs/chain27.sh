#!/bin/bash
#SBATCH -J chain27
#SBATCH --qos=YOUR_QOS --partition=YOUR_CPU_PARTITION
#SBATCH -c 4 --mem=32G -t 02:00:00
#SBATCH -o /path/to/project/logs/%x_%j.out
set -eu
source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
echo "════════ k 敏感性（27 编码器）"
python3 -u k_sens.py
echo; echo "════════ 名次稳定性"
python3 -u k_rank.py
echo; echo "════════ 队列层级误差棒"
python3 -u cohort_spread.py

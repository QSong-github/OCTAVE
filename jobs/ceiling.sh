#!/bin/bash
#SBATCH --job-name=ceiling
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
# 不申请 GPU: hest 环境在 GPU 分区上 numcodecs/blosc 导入失败(见 effres_38035239)
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16
python -u src/ceiling.py --hvg 50 --tmax 2048 --reps 3

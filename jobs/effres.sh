#!/bin/bash
#SBATCH --job-name=effres
#SBATCH --qos=YOUR_QOS
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/systema4ST/logs/%x_%j.out
# 不申请 GPU: hest 环境在 GPU 分区上 numcodecs/blosc 导入失败,
# 而 align.py 的 MLP 头本就有 CPU 回退 (dev = cuda if available else cpu)。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/systema4ST
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16
python -u src/effres.py --tower hibou_l --hvg 50 --tmax 2048

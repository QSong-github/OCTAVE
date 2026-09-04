#!/bin/bash
#SBATCH --job-name=effres
#SBATCH --qos=qsong1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
# 不申请 GPU: hest 环境在 GPU 分区(c0605a-s4)上 numcodecs/blosc 导入失败,
# 而 align.py 的 MLP 头本就有 CPU 回退 (dev = cuda if available else cpu)。
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16
python -u src/effres.py --tower hibou_l --hvg 50 --tmax 2048

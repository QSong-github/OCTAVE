#!/bin/bash
#SBATCH --job-name=ruler
#SBATCH --qos=qsong1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
# CPU 分区: hest 环境在 GPU 节点上 numcodecs/blosc 导入失败(见 effres_38035239)
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=32
python -u src/ruler.py --tower hibou_l --hvg 50 --tmax 2048 --tstar 128 --nclust 20

#!/bin/bash
#SBATCH --job-name=fig1icl
#SBATCH --qos=qsong1
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
# 32 线程：与 jobs/ruler_fold.sh 一致，否则 KMeans 落入不同局部解，
# 重算的 D_domImg_k20 与 ruler_fold.json 对不上（脚本内已断言）。
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
python -u fig1_iclr.py

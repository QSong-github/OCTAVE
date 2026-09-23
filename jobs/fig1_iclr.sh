#!/bin/bash
#SBATCH --job-name=fig1icl
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/path/to/project/logs/%x_%j.out
# 32 线程：与 jobs/ruler_fold.sh 一致，否则 KMeans 落入不同局部解，
# 重算的 D_domImg_k20 与 ruler_fold.json 对不上（脚本内已断言）。
set -e
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate hest
cd /path/to/project
export PYTHONDONTWRITEBYTECODE=1 TQDM_DISABLE=1
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32
python -u fig1_iclr.py

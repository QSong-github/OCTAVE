#!/bin/bash
#SBATCH --job-name=fig1v2
#SBATCH --qos=qsong1 --partition=hpg-default --cpus-per-task=32 --mem=64G --time=3:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export MPLBACKEND=Agg OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
python -u fig1_iclr_v2.py

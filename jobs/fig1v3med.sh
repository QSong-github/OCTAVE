#!/bin/bash
#SBATCH --job-name=fig1v3med
#SBATCH --qos=qsong1 --partition=hpg-default --cpus-per-task=16 --mem=64G --time=3:00:00
#SBATCH --output=/blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export MPLBACKEND=Agg OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 FIG1_SEED=0 FIG1_MODE=median FIG1_TAG=med
python -u fig1_iclr_v3.py

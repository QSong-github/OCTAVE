#!/bin/bash
#SBATCH -J fig2only
#SBATCH --qos=qsong1 --partition=hpg-default -c 4 --mem=16G -t 0:40:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export MPLBACKEND=Agg OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cp -f figures/Fig2_scale.pdf figures/Fig2_scale_nospan.pdf
python -u fig2_scale.py

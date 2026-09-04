#!/bin/bash
#SBATCH -J refig3
#SBATCH --qos=qsong1 --partition=hpg-default -c 32 --mem=64G -t 1:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export MPLBACKEND=Agg OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
python -u make_figs_new.py 3
pdftotext figures/Fig3_evaluation_protocol.pdf - | grep -i "peaks\|best at\|fixed grid" | head -4

#!/bin/bash
#SBATCH -J refig5
#SBATCH --qos=qsong1 --partition=hpg-default -c 32 --mem=64G -t 1:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
export MPLBACKEND=Agg OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
python -u make_figs_new.py 6
cp -f figures/Fig6_protocol_knobs.pdf figures/Fig5_protocol_knobs.pdf
pdftotext figures/Fig5_protocol_knobs.pdf - | grep -i "encoder\|normali\|cp10k\|counts" | head -6

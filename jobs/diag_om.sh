#!/bin/bash
#SBATCH -J diagom
#SBATCH --qos=qsong1 --partition=hpg-default -c 8 --mem=32G -t 2:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export OMP_NUM_THREADS=8
python -u diag_openmidnight.py

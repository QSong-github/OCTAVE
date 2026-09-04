#!/bin/bash
#SBATCH -J agg27
#SBATCH --qos=qsong1 --partition=hpg-default
#SBATCH -c 2 --mem=16G -t 00:30:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -eu
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST && python3 -u agg27.py

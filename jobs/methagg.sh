#!/bin/bash
#SBATCH -J methagg
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/methagg.out
#SBATCH -e /blue/qsong1/wang.qing/systema4ST/logs/methagg.out
#SBATCH -t 00:15:00 -c 1 --mem=8G
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST && python3 -u methods_agg.py

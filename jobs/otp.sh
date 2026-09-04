#!/bin/bash
#SBATCH -J otp
#SBATCH --qos=qsong1 --partition=hpg-default
#SBATCH -c 4 --mem=64G -t 01:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -eu
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST && python3 -u omi_trunk_probe.py

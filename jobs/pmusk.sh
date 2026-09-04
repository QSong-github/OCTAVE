#!/bin/bash
#SBATCH -J pmusk
#SBATCH --qos=qsong1 --partition=hpg-b200,hpg-rtx6000
#SBATCH --gres=gpu:1 -c 6 --mem=64G -t 01:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -eu
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST
pip install --no-cache-dir --no-deps einops 2>&1 | tail -1
python -u probe_musk.py

#!/bin/bash
#SBATCH -J thitosmoke
#SBATCH --qos=qsong1 --partition=hpg-b200,hpg-rtx6000,hpg-turin --gres=gpu:1 -c 6 --mem=32G -t 4:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export PYTHONDONTWRITEBYTECODE=1
python -u src/thitogene_hest.py --cohorts SKCM --fold 0 --out results/thito_smoke.json

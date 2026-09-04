#!/bin/bash
#SBATCH -J probecfg
#SBATCH --qos=qsong1 --partition=hpg-default -c 4 --mem=24G -t 3:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export HF_TOKEN=$(cat /blue/qsong1/wang.qing/.cache/huggingface/token); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN; unset HF_HUB_OFFLINE
python -u probe_cfg.py "$@"

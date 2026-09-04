#!/bin/bash
#SBATCH -J probeb2
#SBATCH --qos=qsong1 --partition=hpg-default -c 4 --mem=16G -t 1:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate hest
cd /blue/qsong1/wang.qing/systema4ST; export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export HF_TOKEN=$(cat /blue/qsong1/wang.qing/.cache/huggingface/token); export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN; unset HF_HUB_OFFLINE
python -u probe_b2.py

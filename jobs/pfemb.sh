#!/bin/bash
#SBATCH -J pfemb
#SBATCH --qos=qsong1 --partition=hpg-b200,hpg-rtx6000,hpg-turin --gres=gpu:1 -c 6 --mem=16G -t 12:00:00
#SBATCH -o /blue/qsong1/wang.qing/systema4ST/logs/%x_%j.out
set -e
source /blue/qsong1/wang.qing/miniconda3/etc/profile.d/conda.sh; conda activate tfpf
cd /blue/qsong1/wang.qing/systema4ST; unset HF_HUB_OFFLINE
python -u pathfound_embed.py
